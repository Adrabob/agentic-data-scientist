"""Unit tests for agents.reflector helpers and main reflect() flow."""
from typing import Any, Dict, List, Optional

import pytest

from agents.reflector import (
    analyze_per_class_performance,
    apply_replan_strategy,
    compare_models_statistically,
    detect_overfitting,
    reflect,
    should_replan,
)


def _metrics(
    name: str,
    *,
    bal_acc: float = 0.7,
    f1_macro: float = 0.65,
    train_f1: Optional[float] = None,
    cv_scores: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Build a metrics dict in the shape `train_models` produces."""
    out: Dict[str, Any] = {
        "model": name,
        "accuracy": bal_acc,
        "balanced_accuracy": bal_acc,
        "f1_macro": f1_macro,
        "precision_macro": f1_macro,
        "recall_macro": f1_macro,
        "cv_scores": list(cv_scores) if cv_scores is not None else [],
    }
    if train_f1 is not None:
        out["train_metrics"] = {
            "accuracy": train_f1,
            "balanced_accuracy": train_f1,
            "f1_macro": train_f1,
        }
    return out


def _profile(imb: float = 1.0) -> Dict[str, Any]:
    return {
        "shape": {"rows": 1000, "cols": 10},
        "imbalance_ratio": imb,
        "notes": [],
        "feature_types": {"numeric": [], "categorical": []},
    }


# ------------------
# detect_overfitting
# ------------------


def test_overfit_detector_flags_large_gap():
    res = detect_overfitting({"f1_macro": 0.95}, {"f1_macro": 0.60})
    assert res["status"] == "overfit"
    assert res["gap"] == pytest.approx(0.35)


def test_overfit_detector_healthy_when_close():
    res = detect_overfitting({"f1_macro": 0.72}, {"f1_macro": 0.70})
    assert res["status"] == "healthy"


def test_overfit_detector_underfit_when_test_higher():
    res = detect_overfitting({"f1_macro": 0.50}, {"f1_macro": 0.70})
    assert res["status"] == "underfit"


# -----------------------------
# analyze_per_class_performance
# -----------------------------


def test_per_class_analysis_finds_worst_class():
    y_test = ["a"] * 10 + ["b"] * 10 + ["c"] * 10
    y_pred = ["a"] * 9 + ["b"] + ["b"] * 10 + ["a"] * 10  # 'c' collapses
    res = analyze_per_class_performance(y_test, y_pred)
    assert res["n_classes"] == 3
    assert res["worst_class"] == "c"
    assert res["worst_f1"] < 0.1


def test_per_class_analysis_handles_binary():
    y_test = ["yes"] * 5 + ["no"] * 5
    y_pred = ["yes"] * 5 + ["no"] * 5
    res = analyze_per_class_performance(y_test, y_pred)
    assert res["n_classes"] == 2
    assert res["worst_f1"] == pytest.approx(1.0)
    assert res["f1_gap"] == pytest.approx(0.0)


# ----------------------------
# compare_models_statistically
# ----------------------------


def test_statistical_comparison_skips_when_no_cv_scores():
    am = [_metrics("A"), _metrics("B")]  # cv_scores empty
    res = compare_models_statistically(am)
    assert res["best_model"] == "A"
    assert res["comparisons"][0]["test"] == "skipped"


def test_statistical_comparison_runs_wilcoxon_with_folds():
    am = [
        _metrics("A", cv_scores=[0.80, 0.82, 0.78, 0.81, 0.79]),
        _metrics("B", cv_scores=[0.60, 0.63, 0.59, 0.61, 0.62]),
    ]
    res = compare_models_statistically(am)
    comp = res["comparisons"][0]
    assert comp["test"] == "wilcoxon"
    assert comp["mean_diff"] > 0
    assert comp["n_folds"] == 5


def test_statistical_comparison_skips_identical_scores():
    am = [
        _metrics("A", cv_scores=[0.7, 0.7, 0.7]),
        _metrics("B", cv_scores=[0.7, 0.7, 0.7]),
    ]
    res = compare_models_statistically(am)
    assert res["comparisons"][0]["reason"] == "identical_scores"


# ---------
# reflect()
# ---------


def test_reflect_returns_required_keys():
    am = [_metrics("LogReg", train_f1=0.7), _metrics("DummyMostFrequent")]
    out = reflect(_profile(), am[0], am)
    for k in (
        "status", "best_model", "issues", "suggestions",
        "structured_issues", "statistical_comparison",
        "per_class", "overfit", "replan_recommended",
    ):
        assert k in out


def test_weak_vs_dummy_flags_high_severity_issue():
    am = [
        _metrics("LogReg", bal_acc=0.51, f1_macro=0.50, train_f1=0.52),
        _metrics("DummyMostFrequent", bal_acc=0.50, f1_macro=0.50),
    ]
    out = reflect(_profile(), am[0], am)
    cats = [i["category"] for i in out["structured_issues"]]
    assert "weak_signal" in cats
    assert any(i["severity"] == "high" for i in out["structured_issues"])
    assert out["replan_recommended"] is True


def test_reflect_overfit_drives_replan():
    am = [_metrics("RF", bal_acc=0.7, f1_macro=0.65, train_f1=0.95)]
    out = reflect(_profile(), am[0], am)
    assert out["overfit"]["status"] == "overfit"
    assert out["replan_recommended"] is True


# ------
# Replan
# ------


def test_should_replan_returns_recommended_flag():
    assert should_replan({"replan_recommended": True}) is True
    assert should_replan({"replan_recommended": False}) is False


def test_apply_replan_overfit_prepends_logreg():
    plan = {"steps": ["select_models"], "rationale": {}, "hints": {}}
    refl = {
        "issues": [],
        "overfit": {"status": "overfit", "gap": 0.30},
        "per_class": None,
    }
    new_plan, _ = apply_replan_strategy(plan, _profile(), refl)
    assert new_plan["hints"]["priority_models"][0] == "LogisticRegression"
    assert new_plan["hints"]["plan_template"] == "regularised"
    assert "replan_regularise" in new_plan["rationale"]


def test_apply_replan_class_collapse_sets_class_weight():
    plan = {"steps": [], "rationale": {}, "hints": {}}
    refl = {
        "issues": [],
        "overfit": None,
        "per_class": {"n_classes": 3, "worst_class": "rare", "worst_f1": 0.1},
    }
    new_plan, _ = apply_replan_strategy(plan, _profile(), refl)
    assert new_plan["hints"]["resample_strategy"] == "class_weight"
    assert "replan_class_weight" in new_plan["rationale"]


def test_apply_replan_legacy_list_upgraded():
    legacy = ["step1", "step2"]
    refl = {"issues": [], "overfit": None, "per_class": None}
    new_plan, _ = apply_replan_strategy(legacy, _profile(), refl)
    assert isinstance(new_plan, dict)
    assert new_plan["steps"] == ["step1", "step2"]
    assert new_plan["hints"]["replan_attempt"] == 1


def test_apply_replan_class_weight_flips_to_smote():
    plan = {
        "steps": [],
        "rationale": {},
        "hints": {"resample_strategy": "class_weight"},
    }
    refl = {"issues": [], "overfit": None, "per_class": None}
    new_plan, _ = apply_replan_strategy(plan, _profile(), refl)
    assert new_plan["hints"]["resample_strategy"] == "smote"
