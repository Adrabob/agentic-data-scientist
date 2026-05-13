"""Unit tests for the orchestrator helpers and error-isolation behaviour."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd
import pytest

from agentic_data_scientist import AgenticDataScientist, _step_enabled


# --------------------------------------------------------------------------- #
# _step_enabled                                                                #
# --------------------------------------------------------------------------- #


def test_step_enabled_defaults_true_for_unknown_step():
    assert _step_enabled("train_models", {}) is True
    assert _step_enabled("evaluate", {"drop_columns": ["x"]}) is True


def test_step_enabled_smote_only_when_hint_smote():
    assert _step_enabled("smote_resample", {"resample_strategy": "smote"}) is True
    assert _step_enabled("smote_resample", {"resample_strategy": "class_weight"}) is False
    assert _step_enabled("smote_resample", {}) is False


def test_step_enabled_class_weight_only_when_hint_class_weight():
    assert _step_enabled("class_weight", {"resample_strategy": "class_weight"}) is True
    assert _step_enabled("class_weight", {"resample_strategy": "smote"}) is False


def test_step_enabled_strips_inline_tags():
    # Planner emits e.g. "build_preprocessor[drop_ids,cast_fix]" — base name only.
    assert _step_enabled("drop_ids[extra]", {"drop_columns": ["patient_id"]}) is True
    assert _step_enabled("drop_ids[extra]", {}) is False


def test_step_enabled_regularise_gated_on_template():
    assert _step_enabled("regularise_models", {"plan_template": "regularised"}) is True
    assert _step_enabled("regularise_models", {"plan_template": "default"}) is False


# --------------------------------------------------------------------------- #
# train_models per-model error isolation                                       #
# --------------------------------------------------------------------------- #


class _BoomEstimator:
    """An sklearn-compatible estimator that raises on fit — used to test isolation."""

    def fit(self, X, y):
        raise RuntimeError("intentional failure for test")

    def predict(self, X):
        raise RuntimeError("intentional failure for test")

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {}

    def set_params(self, **kwargs):
        return self


def _tiny_dataset() -> pd.DataFrame:
    return pd.DataFrame({
        "a": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
              1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
        "b": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
              11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "y": (["pos"] * 10) + (["neg"] * 10),
    })


def _minimal_profile(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "shape": {"rows": df.shape[0], "cols": df.shape[1]},
        "columns": list(df.columns),
        "target": "y",
        "target_dtype": "object",
        "is_classification": True,
        "feature_types": {"numeric": ["a", "b"], "categorical": []},
        "n_unique_by_col": {c: df[c].nunique() for c in df.columns},
        "missing_pct": {c: 0.0 for c in df.columns},
        "imbalance_ratio": 1.0,
        "class_counts": df["y"].value_counts().to_dict(),
        "notes": [],
    }


def test_train_models_isolates_single_failing_candidate(tmp_path):
    from sklearn.linear_model import LogisticRegression
    from tools.modelling import build_preprocessor, train_models

    df = _tiny_dataset()
    profile = _minimal_profile(df)
    preprocessor = build_preprocessor(profile)

    candidates = [
        ("BoomModel", _BoomEstimator()),
        ("LogReg", LogisticRegression(max_iter=500)),
    ]

    results = train_models(
        df=df,
        target="y",
        preprocessor=preprocessor,
        candidates=candidates,
        seed=0,
        test_size=0.3,
        output_dir=str(tmp_path),
        verbose=False,
        cv_folds=3,
    )

    surviving = [r["name"] for r in results["results"]]
    assert "LogReg" in surviving
    assert "BoomModel" not in surviving
    assert any(f.get("model") == "BoomModel" for f in results["failed"])
    assert results["best"]["name"] == "LogReg"


def test_train_models_raises_when_every_candidate_fails(tmp_path):
    from tools.modelling import build_preprocessor, train_models

    df = _tiny_dataset()
    profile = _minimal_profile(df)
    preprocessor = build_preprocessor(profile)

    candidates = [
        ("Boom1", _BoomEstimator()),
        ("Boom2", _BoomEstimator()),
    ]

    with pytest.raises(RuntimeError, match="All 2 candidate models failed"):
        train_models(
            df=df,
            target="y",
            preprocessor=preprocessor,
            candidates=candidates,
            seed=0,
            test_size=0.3,
            output_dir=str(tmp_path),
            verbose=False,
            cv_folds=3,
        )


# --------------------------------------------------------------------------- #
# End-to-end smoke using AgenticDataScientist.run()                            #
# --------------------------------------------------------------------------- #


def test_agent_run_produces_all_artefacts_including_run_log(tmp_path):
    mem_path = tmp_path / "memory.json"
    agent = AgenticDataScientist(memory_path=str(mem_path), verbose=False)

    out_dir = agent.run(
        data_path="data/example_dataset.csv",
        target="auto",
        output_root=str(tmp_path / "outputs"),
        seed=42,
        test_size=0.2,
        max_replans=0,
    )

    expected = [
        "report.md", "eda_summary.json", "plan.json", "metrics.json",
        "reflection.json", "confusion_matrix.png", "run.log",
    ]
    for name in expected:
        assert os.path.exists(os.path.join(out_dir, name)), f"missing {name}"

    log_text = open(os.path.join(out_dir, "run.log"), encoding="utf-8").read()
    assert "Loading dataset" in log_text
    assert "Candidate models" in log_text
