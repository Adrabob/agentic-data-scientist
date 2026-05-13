"""Unit tests for agents.planner.create_plan conditional branches."""
from typing import Any, Dict, List, Optional

import pytest

from agents.planner import create_plan, _extract_signals


def _make_profile(
    *,
    rows: int = 5000,
    cols: int = 10,
    numeric: Optional[List[str]] = None,
    categorical: Optional[List[str]] = None,
    imbalance_ratio: Optional[float] = 1.0,
    imbalance_status: str = "balanced",
    missing_ratios: Optional[Dict[str, float]] = None,
    skewed_cols: Optional[List[str]] = None,
    outlier_cols: Optional[List[str]] = None,
    mixed_dtype_cols: Optional[List[Dict[str, Any]]] = None,
    high_cardinality_cols: Optional[List[Dict[str, Any]]] = None,
    potential_ids: Optional[List[str]] = None,
    useless_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a minimal profile dict that trips only the branches we specify."""
    numeric = numeric or []
    categorical = categorical or []
    return {
        "shape": {"rows": rows, "cols": cols},
        "columns": numeric + categorical,
        "target": "label",
        "target_dtype": "int64",
        "is_classification": True,
        "feature_types": {"numeric": numeric, "categorical": categorical},
        "n_unique_by_col": {},
        "missing_pct": {},
        "imbalance_ratio": imbalance_ratio,
        "class_counts": {},
        "notes": [],
        "data_quality_and_features": {
            "missing_values": {
                "has_missing": bool(missing_ratios),
                "cols": list((missing_ratios or {}).keys()),
                "ratios": missing_ratios or {},
            },
            "numeric_stats": {
                "skewed_cols": skewed_cols or [],
                "outlier_cols": outlier_cols or [],
                "mixed_dtype_cols": mixed_dtype_cols or [],
            },
            "categorical_stats": {
                "high_cardinality_cols": high_cardinality_cols or [],
                "potential_ids": potential_ids or [],
                "useless_cols": useless_cols or [],
            },
        },
        "target_analysis": {
            "target_col": "label",
            "problem_type": "classification",
            "imbalance_status": imbalance_status,
        },
    }


def test_default_plan_balanced_clean():
    plan = create_plan(_make_profile())
    assert plan["rationale"] == {}
    assert plan["hints"]["drop_columns"] == []
    assert plan["hints"]["priority_models"] == []
    assert plan["hints"]["plan_template"] == "default"
    assert all(
        "[" not in s for s in plan["steps"]
    ), "No branches fired — steps should not be tagged."


def test_extract_signals_missing_nested_keys():
    signals = _extract_signals({})
    assert signals["skewed"] == []
    assert signals["ids"] == []
    assert signals["missing_ratios"] == {}
    assert signals["imb_status"] == "balanced"


def test_imbalance_severe_triggers_smote_hint():
    profile = _make_profile(imbalance_ratio=12.0, imbalance_status="severe")
    plan = create_plan(profile)
    assert plan["hints"]["resample_strategy"] == "smote"
    assert "imbalance_resample" in plan["rationale"]


def test_imbalance_moderate_triggers_class_weight():
    profile = _make_profile(imbalance_ratio=4.0, imbalance_status="imbalanced")
    plan = create_plan(profile)
    assert plan["hints"]["resample_strategy"] == "class_weight"


def test_high_cardinality_triggers_target_encode_hint():
    profile = _make_profile(
        categorical=["city"],
        high_cardinality_cols=[{"column": "city", "nunique": 500, "ratio": 0.9}],
    )
    plan = create_plan(profile)
    assert plan["hints"]["target_encode_cols"] == ["city"]
    assert "target_encode" in plan["rationale"]


def test_skewed_triggers_log_transform_hint():
    profile = _make_profile(numeric=["income"], skewed_cols=["income"])
    plan = create_plan(profile)
    assert plan["hints"]["log_transform_cols"] == ["income"]
    assert "log_transform" in plan["rationale"]


def test_mixed_dtype_triggers_cast_fix_hint():
    profile = _make_profile(
        numeric=["price"],
        mixed_dtype_cols=[{"column": "price", "int_like_ratio": 0.92}],
    )
    plan = create_plan(profile)
    assert plan["hints"]["cast_fix_cols"] == ["price"]
    assert "cast_fix" in plan["rationale"]


def test_potential_ids_added_to_drop_columns():
    profile = _make_profile(
        categorical=["patient_id", "flag"],
        potential_ids=["patient_id"],
        useless_cols=["flag"],
    )
    plan = create_plan(profile)
    assert set(plan["hints"]["drop_columns"]) == {"patient_id", "flag"}
    assert "drop_ids" in plan["rationale"]


def test_missing_over_40_marks_drop_col():
    profile = _make_profile(missing_ratios={"sparse_col": 0.85})
    plan = create_plan(profile)
    assert "sparse_col" in plan["hints"]["impute_strategy"].get("drop_col", [])
    assert "missing_severe" in plan["rationale"]


def test_missing_moderate_sets_imputation_defaults():
    profile = _make_profile(missing_ratios={"col_a": 0.15})
    plan = create_plan(profile)
    assert plan["hints"]["impute_strategy"].get("numeric") == "median"
    assert plan["hints"]["impute_strategy"].get("categorical") == "mode"
    assert "missing_moderate" in plan["rationale"]


def test_small_dataset_sets_template_and_cv_folds():
    profile = _make_profile(rows=200, cols=8)
    plan = create_plan(profile)
    assert plan["hints"]["plan_template"] == "small"
    assert plan["hints"]["cv_folds"] == 10
    assert "LogisticRegression" in plan["hints"]["priority_models"]
    assert "small_dataset" in plan["rationale"]


def test_high_dim_prefers_tree_models():
    profile = _make_profile(rows=5000, cols=250)
    plan = create_plan(profile)
    assert plan["hints"]["plan_template"] == "high_dim"
    assert "RandomForest" in plan["hints"]["priority_models"]
    assert "high_dim" in plan["rationale"]


def test_memory_hint_promotes_best_model_to_priority():
    profile = _make_profile()
    memory_hint = {"best_model": "GradientBoosting", "best_metrics": {}}
    plan = create_plan(profile, memory_hint=memory_hint)
    assert plan["hints"]["priority_models"][0] == "GradientBoosting"
    # Default (no hint_source) → exact tag.
    assert "memory_priority_exact" in plan["rationale"]


def test_memory_hint_similar_source_uses_similar_tag():
    profile = _make_profile()
    memory_hint = {
        "best_model": "RandomForest",
        "hint_source": "similar",
        "similarity_score": 0.82,
    }
    plan = create_plan(profile, memory_hint=memory_hint)
    assert "memory_priority_similar" in plan["rationale"]
    assert "memory_priority_exact" not in plan["rationale"]
    assert plan["hints"]["priority_models"][0] == "RandomForest"
