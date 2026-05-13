"""
Planner Agent

Generates a conditional execution plan from a dataset profile. The plan is a
structured dict (not a plain list) so downstream components can consume both
human-readable reasoning and machine-actionable hints:

    {
      "steps":     List[str],        # tagged step labels, e.g. "build_preprocessor[drop_ids]"
      "rationale": Dict[str, str],   # branch_tag -> short reason, rendered in the report
      "hints":     Dict[str, Any],   # actionable recommendations for modelling / preprocessor
    }

Branches fire based on signals extracted from the EDA profile
(`tools.data_profiler.run_eda`) and, optionally, a memory hint from a prior run on the
same dataset fingerprint.
"""

from typing import Any, Dict, List, Optional

# Base, ordered list of pipeline stages. Branches suffix-tag entries in place
# rather than appending many new step names so the plan stays scannable.
_BASE_STEPS: List[str] = [
    "profile_dataset",
    "build_preprocessor",
    "select_models",
    "train_models",
    "evaluate",
    "reflect",
    "write_report",
]


def _empty_hints() -> Dict[str, Any]:
    """Return a fresh hints dict with all keys initialised to safe defaults."""
    return {
        "drop_columns": [],
        "log_transform_cols": [],
        "target_encode_cols": [],
        "cast_fix_cols": [],
        "resample_strategy": None,
        "impute_strategy": {},
        "priority_models": [],
        "cv_folds": 5,
        "plan_template": "default",
        "replan_attempt": 0,
    }


def create_plan(
    dataset_profile: Dict[str, Any],
    memory_hint: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate a conditional execution plan from a dataset profile.

    Parameters
    ----------
    dataset_profile
        Dict produced by `tools.data_profiler.run_eda`. Expected top-level keys:
        `shape`, `feature_types`, `imbalance_ratio`, `missing_pct`,
        `is_classification`, `notes`. Nested diagnostics under
        `data_quality_and_features` and `target_analysis` are used when
        present.
    memory_hint
        Optional record from `agents.memory.JSONMemory.get_dataset_record`
        for the same dataset fingerprint. May contain a `best_model` name.

    Returns
    -------
    Dict with keys `steps` (tagged step list), `rationale`
    (branch -> reason), and `hints` (machine-consumable recommendations).
    """
    steps: List[str] = list(_BASE_STEPS)
    rationale: Dict[str, str] = {}
    hints: Dict[str, Any] = _empty_hints()

    signals = _extract_signals(dataset_profile)

    # Accumulate tags per step so we can suffix them at the end in one pass.
    tags: Dict[str, List[str]] = {s: [] for s in steps}

    # --- Data-quality branches ------------------------------------------------

    # drop_ids: IDs and useless (single-value) columns carry no predictive signal.
    ids_and_useless = _dedupe(signals["ids"] + signals["useless"])
    if ids_and_useless:
        hints["drop_columns"] = ids_and_useless
        rationale["drop_ids"] = (
            f"Dropping {len(ids_and_useless)} id/useless column(s): "
            f"{', '.join(ids_and_useless)}."
        )
        tags["build_preprocessor"].append("drop_ids")

    # log_transform: skewed numeric columns benefit from a log1p transform.
    if signals["skewed"]:
        hints["log_transform_cols"] = list(signals["skewed"])
        rationale["log_transform"] = (
            f"{len(signals['skewed'])} skewed numeric column(s) flagged for "
            f"log-transform: {', '.join(signals['skewed'])}."
        )
        tags["build_preprocessor"].append("log_transform")

    # target_encode: high-cardinality categoricals would explode under one-hot.
    if signals["high_card"]:
        hints["target_encode_cols"] = list(signals["high_card"])
        rationale["target_encode"] = (
            f"{len(signals['high_card'])} high-cardinality categorical(s) "
            f"flagged for target-encoding: {', '.join(signals['high_card'])}."
        )
        tags["build_preprocessor"].append("target_encode")

    # cast_fix: mixed-dtype columns need explicit casting before modelling.
    if signals["mixed"]:
        hints["cast_fix_cols"] = list(signals["mixed"])
        rationale["cast_fix"] = (
            f"{len(signals['mixed'])} column(s) have inconsistent dtypes "
            f"(int/float mix): {', '.join(signals['mixed'])}."
        )
        tags["build_preprocessor"].append("cast_fix")

    # --- Missing-data branches ------------------------------------------------

    severe_missing = [
        c for c, r in signals["missing_ratios"].items() if float(r) > 0.40
    ]
    moderate_missing = [
        c for c, r in signals["missing_ratios"].items() if 0.05 <= float(r) <= 0.40
    ]
    if severe_missing:
        hints["impute_strategy"]["drop_col"] = severe_missing
        rationale["missing_severe"] = (
            f"{len(severe_missing)} column(s) missing >40% of values; "
            f"recommending drop: {', '.join(severe_missing)}."
        )
        tags["build_preprocessor"].append("missing_severe")
    if moderate_missing:
        hints["impute_strategy"].setdefault("numeric", "median")
        hints["impute_strategy"].setdefault("categorical", "mode")
        rationale["missing_moderate"] = (
            f"{len(moderate_missing)} column(s) missing 5%–40%; "
            f"standard median/mode imputation."
        )
        tags["build_preprocessor"].append("missing_moderate")

    # --- Imbalance branch -----------------------------------------------------

    imb_status = signals["imb_status"]
    if imb_status in ("imbalanced", "severe"):
        hints["resample_strategy"] = (
            "class_weight" if imb_status == "imbalanced" else "smote"
        )
        rationale["imbalance_resample"] = (
            f"Target imbalance status = '{imb_status}'; "
            f"strategy = {hints['resample_strategy']}."
        )
        tags["train_models"].append("imbalance_resample")

    # --- Plan-template branches (size-driven) --------------------------------

    if signals["rows"] and signals["rows"] < 1000:
        hints["plan_template"] = "small"
        hints["cv_folds"] = 10
        hints["priority_models"] = _dedupe(
            hints["priority_models"] + ["LogisticRegression", "RandomForest"]
        )
        rationale["small_dataset"] = (
            f"Only {signals['rows']} rows; favouring regularised/tree models "
            f"and using 10-fold CV."
        )
        tags["select_models"].append("small_dataset")

    if signals["cols"] > 100:
        hints["plan_template"] = "high_dim"
        hints["priority_models"] = _dedupe(
            hints["priority_models"] + ["RandomForest", "GradientBoosting"]
        )
        rationale["high_dim"] = (
            f"{signals['cols']} columns; prioritising tree-based models."
        )
        tags["select_models"].append("high_dim")

    # --- Memory-hint branch ---------------------------------------------------

    if memory_hint and memory_hint.get("best_model"):
        prior_best = memory_hint["best_model"]
        # Prepend prior best so it is tried first, but keep template-driven
        # priorities afterwards. _dedupe keeps first occurrence.
        hints["priority_models"] = _dedupe(
            [prior_best] + hints["priority_models"]
        )
        hint_source = str(memory_hint.get("hint_source", "exact"))
        if hint_source == "similar":
            score = float(memory_hint.get("similarity_score", 0.0))
            rationale_tag = "memory_priority_similar"
            rationale[rationale_tag] = (
                f"Nearest prior dataset (similarity={score:.3f}) preferred "
                f"'{prior_best}'; trying it first."
            )
            tags["select_models"].append("memory_priority_similar")
        else:
            rationale_tag = "memory_priority_exact"
            rationale[rationale_tag] = (
                f"Prior run on this dataset fingerprint preferred "
                f"'{prior_best}'; trying it first."
            )
            tags["select_models"].append("memory_priority_exact")

    # Apply accumulated tags back onto the step list.
    steps = [
        f"{s}[{','.join(tags[s])}]" if tags.get(s) else s for s in steps
    ]

    return {"steps": steps, "rationale": rationale, "hints": hints}


def _dedupe(items: List[str]) -> List[str]:
    """Return items with duplicates removed, preserving first occurrence."""
    seen: Dict[str, None] = {}
    for x in items:
        if x not in seen:
            seen[x] = None
    return list(seen.keys())


def _extract_signals(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise EDA signals into a flat dict, guarding every nested access."""
    dq = profile.get("data_quality_and_features", {}) or {}
    numeric_stats = dq.get("numeric_stats", {}) or {}
    categorical_stats = dq.get("categorical_stats", {}) or {}
    missing_values = dq.get("missing_values", {}) or {}
    target_analysis = profile.get("target_analysis", {}) or {}
    shape = profile.get("shape", {}) or {}
    return {
        "rows": int(shape.get("rows", 0)),
        "cols": int(shape.get("cols", 0)),
        "skewed": list(numeric_stats.get("skewed_cols", []) or []),
        "outliers": list(numeric_stats.get("outlier_cols", []) or []),
        "mixed": [
            c.get("column")
            for c in (numeric_stats.get("mixed_dtype_cols", []) or [])
            if c.get("column")
        ],
        "high_card": [
            c.get("column")
            for c in (categorical_stats.get("high_cardinality_cols", []) or [])
            if c.get("column")
        ],
        "ids": list(categorical_stats.get("potential_ids", []) or []),
        "useless": list(categorical_stats.get("useless_cols", []) or []),
        "missing_cols": list(missing_values.get("cols", []) or []),
        "missing_ratios": dict(missing_values.get("ratios", {}) or {}),
        "imb_status": target_analysis.get("imbalance_status", "balanced"),
        "imb_ratio": profile.get("imbalance_ratio"),
    }
