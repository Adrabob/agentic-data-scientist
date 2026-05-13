"""Reflector agent.

Analyses a finished training run and emits:
  - structured issues tagged by severity (high / medium / low)
  - statistical comparison of candidates (Wilcoxon over per-fold CV f1_macro)
  - per-class performance breakdown for the best model
  - train-vs-test overfit check for the best model
  - a replan recommendation that the orchestrator can act on

Replan strategies in `apply_replan_strategy` mutate the plan dict's `hints`
so that the next loop iteration in `agentic_data_scientist.run()` actually
trains a different shortlist.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sklearn.metrics import classification_report

from scipy.stats import wilcoxon


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def detect_overfitting(
    train_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    threshold: float = 0.15,
) -> Dict[str, Any]:
    """Compare train vs. test macro F1 and label the gap.

    `gap > threshold`  → overfit
    `gap < -0.05`      → underfit (test exceeds train; suspicious)
    otherwise          → healthy
    """
    train_f1 = float(train_metrics.get("f1_macro", 0.0))
    test_f1 = float(test_metrics.get("f1_macro", 0.0))
    gap = train_f1 - test_f1
    if gap > threshold:
        status = "overfit"
    elif gap < -0.05:
        status = "underfit"
    else:
        status = "healthy"
    return {
        "status": status,
        "gap": float(gap),
        "train_f1": train_f1,
        "test_f1": test_f1,
        "threshold": float(threshold),
    }


def analyze_per_class_performance(y_test, y_pred) -> Dict[str, Any]:
    """Per-class F1 breakdown using sklearn's classification_report."""
    y_test_list = list(y_test) if hasattr(y_test, "tolist") else y_test
    y_pred_list = list(y_pred) if hasattr(y_pred, "tolist") else y_pred

    report = classification_report(
        y_test_list, y_pred_list, output_dict=True, zero_division=0
    )

    per_class = {
        k: v for k, v in report.items()
        if isinstance(v, dict) and k not in ("macro avg", "weighted avg")
    }

    if not per_class:
        return {"per_class_f1": {}, "n_classes": 0}

    f1_by_class = {str(k): float(v.get("f1-score", 0.0)) for k, v in per_class.items()}
    sorted_f1 = sorted(f1_by_class.items(), key=lambda kv: kv[1])
    worst_class, worst_f1 = sorted_f1[0]
    best_f1 = sorted_f1[-1][1]

    return {
        "per_class_f1": f1_by_class,
        "n_classes": len(f1_by_class),
        "worst_class": str(worst_class),
        "worst_f1": float(worst_f1),
        "best_f1": float(best_f1),
        "f1_gap": float(best_f1 - worst_f1),
        "problematic_classes": [k for k, v in f1_by_class.items() if v < 0.5],
    }


def compare_models_statistically(all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pairwise Wilcoxon between the best model and every other candidate.

    Expects each entry in `all_metrics` to have a `cv_scores: List[float]`
    field (added by `tools.modelling.train_models`). Pairs with fewer than
    3 folds are reported as `test: "skipped"` rather than fabricated.
    """
    if not all_metrics:
        return {"best_model": None, "comparisons": []}

    best = all_metrics[0]
    best_name = best.get("model")
    best_cv = list(best.get("cv_scores") or [])

    comparisons: List[Dict[str, Any]] = []
    for other in all_metrics[1:]:
        other_name = other.get("model")
        other_cv = list(other.get("cv_scores") or [])

        if len(best_cv) < 3 or len(other_cv) < 3 or len(best_cv) != len(other_cv):
            comparisons.append({
                "vs": other_name,
                "test": "skipped",
                "reason": "insufficient_or_mismatched_folds",
                "n_folds": min(len(best_cv), len(other_cv)),
            })
            continue

        diffs = np.asarray(best_cv) - np.asarray(other_cv)
        if np.allclose(diffs, 0.0):
            comparisons.append({
                "vs": other_name,
                "test": "skipped",
                "reason": "identical_scores",
                "mean_diff": 0.0,
                "n_folds": len(best_cv),
            })
            continue

        try:
            stat, p = wilcoxon(best_cv, other_cv)
            comparisons.append({
                "vs": other_name,
                "test": "wilcoxon",
                "statistic": float(stat),
                "p_value": float(p),
                "significant": bool(p < 0.05),
                "mean_diff": float(np.mean(diffs)),
                "n_folds": len(best_cv),
            })
        except Exception as exc:
            comparisons.append({
                "vs": other_name,
                "test": "skipped",
                "reason": f"wilcoxon_error: {exc}",
            })

    return {"best_model": best_name, "comparisons": comparisons}


# --------------------------------------------------------------------------- #
# Main reflect()                                                               #
# --------------------------------------------------------------------------- #


_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _add_issue(
    issues: List[Dict[str, Any]],
    severity: str,
    category: str,
    message: str,
    suggestion: str,
) -> None:
    issues.append({
        "severity": severity,
        "category": category,
        "message": message,
        "suggestion": suggestion,
    })


def reflect(
    dataset_profile: Dict[str, Any],
    evaluation: Dict[str, Any],
    all_metrics: List[Dict[str, Any]],
    best_predictions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Analyse results and return a structured reflection.

    Args:
        dataset_profile: EDA profile dict.
        evaluation: best model's metrics dict (from `evaluate_best`).
        all_metrics: per-model metrics; each entry should carry `cv_scores`
            and `train_metrics` for stat tests + overfit detection.
        best_predictions: `{"y_test": ..., "y_pred": ...}` for the best model
            so per-class analysis can run. Optional for backward-compat.

    Returns:
        Dict with keys: status, best_model, issues, suggestions,
        structured_issues, statistical_comparison, per_class, overfit,
        replan_recommended.
    """
    best_model = evaluation.get("model")
    bal_acc = float(evaluation.get("balanced_accuracy", 0.0))
    f1_macro = float(evaluation.get("f1_macro", 0.0))
    imb = float(dataset_profile.get("imbalance_ratio") or 1.0)

    structured: List[Dict[str, Any]] = []

    best_entry = next(
        (m for m in all_metrics if m.get("model") == best_model), None
    )

    # 1. Dummy baseline gap
    dummy = next((m for m in all_metrics if "Dummy" in str(m.get("model", ""))), None)
    if dummy is not None:
        gap_dummy = bal_acc - float(dummy.get("balanced_accuracy", 0.0))
        if gap_dummy < 0.05:
            _add_issue(
                structured,
                "high",
                "weak_signal",
                f"Best model only {gap_dummy:.3f} better than baseline. "
                "Weak signal or pipeline issues.",
                "Check for target leakage, verify target quality, "
                "or improve feature engineering.",
            )

    # 2. Overfit detection on best model (needs train_metrics)
    overfit_info: Optional[Dict[str, Any]] = None
    if best_entry is not None and best_entry.get("train_metrics"):
        overfit_info = detect_overfitting(best_entry["train_metrics"], evaluation)
        if overfit_info["status"] == "overfit":
            sev = "high" if overfit_info["gap"] > 0.20 else "medium"
            _add_issue(
                structured,
                sev,
                "overfit",
                f"Train-test macro F1 gap = {overfit_info['gap']:.3f} "
                f"(train={overfit_info['train_f1']:.3f}, "
                f"test={overfit_info['test_f1']:.3f}).",
                "Use a simpler/regularised model or reduce feature dimensionality.",
            )
        elif overfit_info["status"] == "underfit":
            _add_issue(
                structured,
                "medium",
                "underfit",
                f"Test exceeds train (gap={overfit_info['gap']:.3f}); "
                "split or trivial baseline?",
                "Inspect baseline, stratification, and target distribution.",
            )

    # 3. Per-class performance (multiclass / when predictions available)
    per_class: Optional[Dict[str, Any]] = None
    if best_predictions is not None and best_predictions.get("y_pred") is not None:
        per_class = analyze_per_class_performance(
            best_predictions["y_test"], best_predictions["y_pred"]
        )
        if per_class.get("n_classes", 0) > 1:
            if per_class.get("worst_f1", 1.0) < 0.3:
                _add_issue(
                    structured,
                    "high",
                    "class_collapse",
                    f"Worst class '{per_class['worst_class']}' "
                    f"F1={per_class['worst_f1']:.3f} (collapsed).",
                    "Apply class_weight, threshold tuning, or oversampling "
                    "for the under-performing class.",
                )
            elif per_class.get("f1_gap", 0.0) > 0.30:
                _add_issue(
                    structured,
                    "medium",
                    "class_gap",
                    f"Per-class F1 gap = {per_class['f1_gap']:.3f} "
                    f"(best={per_class['best_f1']:.3f}, "
                    f"worst={per_class['worst_f1']:.3f}).",
                    "Inspect precision/recall per class; consider rebalancing.",
                )

    # 4. Macro F1 threshold
    if f1_macro < 0.60:
        sev = "high" if f1_macro < 0.40 else "medium"
        _add_issue(
            structured,
            sev,
            "low_f1",
            f"Macro F1 = {f1_macro:.3f} (below 0.60).",
            "Try different models, tune hyperparameters, or improve preprocessing.",
        )

    # 5. Imbalance hint (low severity, planner already handles)
    if imb >= 3.0:
        _add_issue(
            structured,
            "low",
            "imbalance",
            f"Dataset imbalance ratio = {imb:.2f}.",
            "Consider class_weight, threshold tuning, or SMOTE.",
        )

    # 6. Statistical comparison between candidates
    stat_comparison = compare_models_statistically(all_metrics)
    no_significance = False
    real_comparisons = [
        c for c in stat_comparison.get("comparisons", []) if c.get("test") == "wilcoxon"
    ]
    if real_comparisons:
        any_significant = any(c.get("significant") for c in real_comparisons)
        if not any_significant:
            no_significance = True
            _add_issue(
                structured,
                "low",
                "no_stat_difference",
                "Best model is not significantly better than alternatives "
                "(all Wilcoxon p > 0.05).",
                "Treat the ranking as a tie; favour simpler / faster model.",
            )

    # Sort by severity (high → low) for stable rendering downstream
    structured.sort(key=lambda x: _SEVERITY_ORDER.get(x.get("severity"), 99))

    issues_flat = [f"[{i['severity']}] {i['message']}" for i in structured]
    suggestions_flat = [i["suggestion"] for i in structured]

    has_high = any(i["severity"] == "high" for i in structured)
    status = "needs_attention" if (has_high or len(structured) >= 3) else "ok"

    # Replan when there's something material to fix:
    #   - a high-severity issue, OR
    #   - overfit detected on the best model.
    # `no_significance` alone is too weak — it would force a replan even on
    # near-perfect runs where two models simply tie.
    replan_recommended = bool(
        has_high
        or (overfit_info is not None and overfit_info["status"] == "overfit")
    )

    return {
        "status": status,
        "best_model": best_model,
        "issues": issues_flat,
        "suggestions": suggestions_flat,
        "structured_issues": structured,
        "statistical_comparison": stat_comparison,
        "per_class": per_class,
        "overfit": overfit_info,
        "replan_recommended": replan_recommended,
    }



# Replan                                 #

def should_replan(reflection: Dict[str, Any]) -> bool:
    """Decide whether to trigger replanning based on reflection.

    Diminishing-returns gating against `max_replans` is enforced by the
    orchestrator, so this only signals 'a useful replan would be possible'.
    """
    return bool(reflection.get("replan_recommended", False))


def _prepend_unique(names: List[str], head: List[str]) -> List[str]:
    merged: List[str] = []
    for n in head + list(names):
        if n not in merged:
            merged.append(n)
    return merged


def apply_replan_strategy(
    plan: Any,
    dataset_profile: Dict[str, Any],
    reflection: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Modify the plan and dataset profile based on reflection.

    Returns a (plan_dict, profile) tuple. Legacy list plans are upgraded
    in-place to the structured `{steps, rationale, hints}` shape.
    """
    if isinstance(plan, dict):
        new_plan: Dict[str, Any] = {
            "steps": list(plan.get("steps", [])),
            "rationale": dict(plan.get("rationale", {})),
            "hints": dict(plan.get("hints", {})),
        }
    else:
        new_plan = {
            "steps": list(plan or []),
            "rationale": {},
            "hints": {},
        }

    hints = new_plan["hints"]
    rationale = new_plan["rationale"]
    new_profile = dict(dataset_profile)

    # Track replan iteration depth for telemetry / report.
    hints["replan_attempt"] = int(hints.get("replan_attempt", 0)) + 1

    # Strategy 1 — weak signal vs dummy → escalate to stronger trees.
    issues = reflection.get("issues", []) or []
    weak_signal = any(
        "only" in str(i) and "better than baseline" in str(i) for i in issues
    )
    if weak_signal:
        hints["priority_models"] = _prepend_unique(
            hints.get("priority_models", []) or [],
            ["GradientBoosting", "RandomForest"],
        )
        rationale["replan_escalate"] = (
            "Weak signal vs dummy baseline — escalating to stronger tree models."
        )
        if "escalate_models" not in new_plan["steps"]:
            new_plan["steps"].append("escalate_models")

    # Strategy 2 — class_weight already tried → flip to SMOTE hint.
    if hints.get("resample_strategy") == "class_weight":
        hints["resample_strategy"] = "smote"
        rationale["replan_resample"] = (
            "class_weight already applied — flipping resample strategy to SMOTE."
        )

    # Strategy 3 — overfit detected → regularise via LogisticRegression.
    overfit = reflection.get("overfit") or {}
    if overfit.get("status") == "overfit":
        hints["plan_template"] = "regularised"
        hints["priority_models"] = _prepend_unique(
            hints.get("priority_models", []) or [],
            ["LogisticRegression"],
        )
        rationale["replan_regularise"] = (
            f"Overfit detected (train-test F1 gap={overfit.get('gap', 0.0):.3f}) "
            "— prepending LogisticRegression for regularisation."
        )
        if "regularise_models" not in new_plan["steps"]:
            new_plan["steps"].append("regularise_models")

    # Strategy 4 — class collapse → apply class_weight if not already set.
    per_class = reflection.get("per_class") or {}
    if (
        per_class.get("n_classes", 0) > 1
        and per_class.get("worst_f1", 1.0) < 0.3
        and hints.get("resample_strategy") is None
    ):
        hints["resample_strategy"] = "class_weight"
        rationale["replan_class_weight"] = (
            f"Worst-class F1={per_class.get('worst_f1', 0.0):.3f} "
            f"on class '{per_class.get('worst_class')}' "
            "— applying class_weight."
        )

    notes = list(new_profile.get("notes", []))
    notes.append(f"Replan attempt #{hints['replan_attempt']} after reflection.")
    new_profile["notes"] = notes

    return new_plan, new_profile
