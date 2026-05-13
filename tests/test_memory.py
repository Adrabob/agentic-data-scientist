"""Unit tests for agents.memory: richer storage + similarity retrieval."""
from __future__ import annotations

from typing import Any, Dict

import pytest

from agents.memory import (
    JSONMemory,
    build_feature_summary,
    now_iso,
)


# -------
# Helpers
# -------


def _record(
    n_rows: int = 1000,
    n_cols: int = 10,
    n_numeric: int = 7,
    n_categorical: int = 3,
    imb: float = 1.0,
    best: str = "LogisticRegression",
    last_seen: str = "2026-04-16T00:00:00Z",
) -> Dict[str, Any]:
    total = max(1, n_numeric + n_categorical)
    return {
        "last_seen": last_seen,
        "target": "y",
        "shape": {"rows": n_rows, "cols": n_cols},
        "best_model": best,
        "best_metrics": {"model": best, "balanced_accuracy": 0.8},
        "feature_summary": {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "n_numeric": n_numeric,
            "n_categorical": n_categorical,
            "n_numeric_ratio": n_numeric / total,
            "n_categorical_ratio": n_categorical / total,
            "imbalance_ratio": imb,
            "has_missing": False,
            "is_classification": True,
        },
    }


def _profile(
    n_rows: int = 500,
    n_cols: int = 8,
    numeric: int = 5,
    categorical: int = 3,
    imb: float = 1.5,
) -> Dict[str, Any]:
    return {
        "shape": {"rows": n_rows, "cols": n_cols},
        "feature_types": {
            "numeric": [f"n{i}" for i in range(numeric)],
            "categorical": [f"c{i}" for i in range(categorical)],
        },
        "missing_pct": {"n0": 0.0},
        "imbalance_ratio": imb,
        "is_classification": True,
    }


# ---------------------
# build_feature_summary
# ---------------------


def test_build_feature_summary_extracts_core_fields():
    profile = _profile(n_rows=1000, n_cols=10, numeric=7, categorical=3, imb=2.0)
    fs = build_feature_summary(profile)
    assert fs["n_rows"] == 1000
    assert fs["n_cols"] == 10
    assert fs["n_numeric"] == 7
    assert fs["n_categorical"] == 3
    assert fs["n_numeric_ratio"] == pytest.approx(0.7, rel=0.02)
    assert fs["imbalance_ratio"] == pytest.approx(2.0)
    assert fs["is_classification"] is True


def test_build_feature_summary_handles_empty_feature_types():
    profile = {"shape": {"rows": 0, "cols": 0}, "feature_types": {}}
    fs = build_feature_summary(profile)
    # n_numeric_ratio falls back to 0 when total_features is 0 (division guard).
    assert fs["n_numeric_ratio"] == 0.0
    assert fs["n_categorical_ratio"] == 0.0


# ---------------
# CRUD roundtrips
# ---------------


def test_upsert_and_get_roundtrip(tmp_path):
    mem = JSONMemory(path=str(tmp_path / "mem.json"))
    mem.upsert_dataset_record("fp1", _record(best="RandomForest"))
    got = mem.get_dataset_record("fp1")
    assert got is not None
    assert got["best_model"] == "RandomForest"

    # Reload from disk to confirm persistence.
    mem2 = JSONMemory(path=str(tmp_path / "mem.json"))
    assert mem2.get_dataset_record("fp1")["best_model"] == "RandomForest"


def test_get_returns_none_for_missing_fingerprint(tmp_path):
    mem = JSONMemory(path=str(tmp_path / "mem.json"))
    assert mem.get_dataset_record("does-not-exist") is None


# ---------------------
# find_similar_datasets
# ---------------------


def test_find_similar_handles_empty_store(tmp_path):
    mem = JSONMemory(path=str(tmp_path / "mem.json"))
    assert mem.find_similar_datasets(build_feature_summary(_profile())) == []


def test_find_similar_returns_topk_sorted_by_distance(tmp_path):
    mem = JSONMemory(path=str(tmp_path / "mem.json"))
    # Three prior records at increasing sizes; query matches the smallest.
    mem.upsert_dataset_record("fp_small",
                              _record(n_rows=500, n_cols=10, n_numeric=7,
                                      n_categorical=3, imb=1.0,
                                      best="LogReg"))
    mem.upsert_dataset_record("fp_mid",
                              _record(n_rows=50_000, n_cols=30,
                                      n_numeric=20, n_categorical=10,
                                      imb=1.0, best="RF"))
    mem.upsert_dataset_record("fp_huge",
                              _record(n_rows=5_000_000, n_cols=300,
                                      n_numeric=200, n_categorical=100,
                                      imb=30.0, best="GB"))

    query = build_feature_summary(_profile(n_rows=600, n_cols=11,
                                           numeric=8, categorical=3, imb=1.1))
    results = mem.find_similar_datasets(query, top_k=2)

    assert len(results) == 2
    assert results[0]["fingerprint"] == "fp_small"
    # Returned list is distance-sorted ascending.
    assert results[0]["distance"] <= results[1]["distance"]
    # And similarity_score matches the inverse relationship.
    assert results[0]["similarity_score"] >= results[1]["similarity_score"]


def test_similarity_score_higher_for_closer_shape(tmp_path):
    mem = JSONMemory(path=str(tmp_path / "mem.json"))
    mem.upsert_dataset_record(
        "close",
        _record(n_rows=1000, n_cols=10, n_numeric=7, n_categorical=3, imb=1.0),
    )
    mem.upsert_dataset_record(
        "far",
        _record(n_rows=10_000_000, n_cols=500, n_numeric=400,
                n_categorical=100, imb=50.0),
    )

    query = build_feature_summary(
        _profile(n_rows=1100, n_cols=11, numeric=7, categorical=3, imb=1.05)
    )
    results = mem.find_similar_datasets(query, top_k=2)
    scores = {r["fingerprint"]: r["similarity_score"] for r in results}
    assert scores["close"] > scores["far"]
    # The close record should be comfortably above the 0.7 threshold used
    # by the orchestrator to decide whether to apply a similar-hint.
    assert scores["close"] > 0.7


def test_legacy_record_without_feature_summary_is_skipped(tmp_path):
    mem = JSONMemory(path=str(tmp_path / "mem.json"))
    mem.upsert_dataset_record(
        "legacy",
        {
            "last_seen": now_iso(),
            "target": "y",
            "shape": {"rows": 100, "cols": 5},
            "best_model": "LogReg",
            "best_metrics": {"model": "LogReg"},
        },
    )
    mem.upsert_dataset_record("modern", _record(n_rows=100, n_cols=5,
                                                n_numeric=3, n_categorical=2))

    results = mem.find_similar_datasets(
        build_feature_summary(_profile(n_rows=100, n_cols=5))
    )
    fingerprints = [r["fingerprint"] for r in results]
    assert "legacy" not in fingerprints
    assert "modern" in fingerprints
