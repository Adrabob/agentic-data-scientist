"""JSON-backed persistent memory for the agent.

Stores per-dataset experience keyed by a SHA256 fingerprint of shape + columns
+ target (see `tools.data_profiler.dataset_fingerprint`). Records now carry a compact
`feature_summary` block so the memory can support **similarity-based
retrieval** when an exact fingerprint lookup misses.

Similarity metric (see `find_similar_datasets`):
    Normalised Manhattan distance over 5 dimensions:
        log10(1 + n_rows)         scaled by 7.0  (≈ 10M rows)
        log10(1 + n_cols)         scaled by 3.0  (≈ 1k cols)
        n_numeric_ratio           span 1.0
        n_categorical_ratio       span 1.0
        log10(1 + imbalance_ratio) scaled by 2.0  (≈ 100:1)
    similarity_score = 1 / (1 + distance)  — in (0, 1]
"""

import json
import math
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional


# Per-dimension "typical span" used to normalise Manhattan distance. These are
# not learned from data; they describe the order-of-magnitude range each
# dimension is expected to cover for the datasets CE888 students encounter.
_FEATURE_SCALES: Dict[str, float] = {
    "log_rows": 7.0,
    "log_cols": 3.0,
    "numeric_ratio": 1.0,
    "categorical_ratio": 1.0,
    "log_imbalance": 2.0,
}


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def build_feature_summary(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Compact, similarity-friendly profile snapshot.

    Extracts the minimum set of numeric / boolean features needed by
    `find_similar_datasets`. Safe to call on partial profiles; missing keys
    resolve to neutral defaults.
    """
    shape = profile.get("shape", {}) or {}
    n_rows = int(shape.get("rows", 0) or 0)
    n_cols = int(shape.get("cols", 0) or 0)

    feature_types = profile.get("feature_types", {}) or {}
    n_numeric = len(feature_types.get("numeric", []) or [])
    n_categorical = len(feature_types.get("categorical", []) or [])
    total_features = max(1, n_numeric + n_categorical)

    missing = profile.get("missing_pct", {}) or {}
    has_missing = any(float(v or 0) > 0 for v in missing.values())

    imb = float(profile.get("imbalance_ratio") or 1.0)

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_numeric": n_numeric,
        "n_categorical": n_categorical,
        "n_numeric_ratio": round(n_numeric / total_features, 4),
        "n_categorical_ratio": round(n_categorical / total_features, 4),
        "imbalance_ratio": imb,
        "has_missing": bool(has_missing),
        "is_classification": bool(profile.get("is_classification", True)),
    }


def _vectorise(summary: Dict[str, Any]) -> Dict[str, float]:
    """Project a feature_summary dict onto the 5-D space used for distance."""
    return {
        "log_rows": math.log10(1 + max(0, int(summary.get("n_rows", 0) or 0))),
        "log_cols": math.log10(1 + max(0, int(summary.get("n_cols", 0) or 0))),
        "numeric_ratio": float(summary.get("n_numeric_ratio", 0.0) or 0.0),
        "categorical_ratio": float(summary.get("n_categorical_ratio", 0.0) or 0.0),
        "log_imbalance": math.log10(
            1 + max(0.0, float(summary.get("imbalance_ratio", 1.0) or 1.0))
        ),
    }


def _manhattan_distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    va = _vectorise(a)
    vb = _vectorise(b)
    total = 0.0
    for key, scale in _FEATURE_SCALES.items():
        if scale <= 0:
            continue
        total += abs(va[key] - vb[key]) / scale
    return total


class JSONMemory:
    """Lightweight persistent memory for the agent.

    Stores `{fingerprint: record}` pairs under the `datasets` key. Each record
    is written by the orchestrator and contains:
        - `last_seen`        (ISO timestamp)
        - `target`           (target column name)
        - `shape`            ({"rows": int, "cols": int})
        - `best_model`       (name)
        - `best_metrics`     (dict from evaluate_best)
        - `feature_summary`  (from build_feature_summary) 
        - `applied_hints`    (plan["hints"] snapshot)    
        - `top_3_models`     ([(name, bal_acc), ...])    
        - `reflection_issues` (severity-tagged list)     
    """

    def __init__(self, path: str = "agent_memory.json"):
        self.path = path
        self.data: Dict[str, Any] = {"datasets": {}, "notes": []}
        self._load()

    # --- IO ------------------------------------------------------------------

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except Exception:
            backup = self.path + ".bak"
            shutil.copy(self.path, backup)
            self.data = {
                "datasets": {},
                "notes": [
                    {"ts": now_iso(), "msg": f"Memory reset; backup at {backup}"}
                ],
            }

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    # --- CRUD ----------------------------------------------------------------

    def get_dataset_record(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        return self.data.get("datasets", {}).get(fingerprint)

    def upsert_dataset_record(
        self, fingerprint: str, record: Dict[str, Any]
    ) -> None:
        self.data.setdefault("datasets", {})[fingerprint] = record
        self.save()

    def add_note(self, msg: str) -> None:
        self.data.setdefault("notes", []).append({"ts": now_iso(), "msg": msg})
        self.save()

    # --- Similarity retrieval ---

    def find_similar_datasets(
        self,
        current_summary: Dict[str, Any],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return the top-k prior records closest to `current_summary`.

        Records without a `feature_summary` (legacy schema or manual notes)
        are excluded. The returned list is annotated with:
            - `fingerprint`       (store key)
            - `distance`          (normalised Manhattan, lower = more similar)
            - `similarity_score`  (1 / (1 + distance), higher = more similar)
        Sorted by `distance` ascending; ties broken by `last_seen` descending
        so the more recent record wins on a tie.
        """
        datasets = self.data.get("datasets", {}) or {}
        if not datasets:
            return []

        ranked: List[Dict[str, Any]] = []
        for fp, record in datasets.items():
            fs = record.get("feature_summary")
            if not fs:
                continue
            dist = _manhattan_distance(current_summary, fs)
            entry = dict(record)
            entry["fingerprint"] = fp
            entry["distance"] = round(dist, 6)
            entry["similarity_score"] = round(1.0 / (1.0 + dist), 6)
            ranked.append(entry)

        ranked.sort(
            key=lambda r: (r["distance"], -_ts_to_sort_key(r.get("last_seen"))),
        )
        return ranked[: max(1, int(top_k))]


def _ts_to_sort_key(ts: Optional[str]) -> float:
    """Convert an ISO timestamp to a sortable float. Missing → 0.0."""
    if not ts:
        return 0.0
    try:
        # Strip trailing 'Z' for fromisoformat on Python 3.10.
        raw = ts.rstrip("Z")
        return datetime.fromisoformat(raw).timestamp()
    except Exception:
        return 0.0
