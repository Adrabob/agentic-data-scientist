import os
import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: str, title: str) -> None:
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)

    thresh = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(int(cm[i, j]), "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def evaluate_best(training_payload: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    best = training_payload["best"]
    all_metrics = training_payload["all_metrics"]

    y_test = best["y_test"]
    y_pred = best["y_pred"]

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted([str(x) for x in y_test.dropna().unique().tolist()])
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, labels, cm_path, f"Confusion Matrix: {best['name']}")

    # Classification report
    cls_report = classification_report(y_test, y_pred, zero_division=0)

    return {
        "best_metrics": best["metrics"],
        "all_metrics": all_metrics,
        "confusion_matrix_path": cm_path,
        "classification_report": cls_report,
    }


def write_markdown_report(
    out_path: str,
    ctx: Any,
    fingerprint: str,
    dataset_profile: Dict[str, Any],
    plan: Any,
    eval_payload: Dict[str, Any],
    reflection: Dict[str, Any],
) -> None:
    best = eval_payload["best_metrics"]

    def short_list(xs: List[str], n: int = 12) -> str:
        return ", ".join(xs[:n]) + (" ..." if len(xs) > n else "")

    # `plan` may be a legacy List[str] or the new structured dict with
    # keys {steps, rationale, hints}. Accept both for backward compatibility.
    if isinstance(plan, dict):
        plan_steps = plan.get("steps", [])
        plan_rationale = plan.get("rationale", {}) or {}
        plan_hints = plan.get("hints", {}) or {}
    else:
        plan_steps = list(plan or [])
        plan_rationale = {}
        plan_hints = {}

    if plan_rationale:
        rationale_block = "\n".join(
            [f"- **{tag}**: {reason}" for tag, reason in plan_rationale.items()]
        )
    else:
        rationale_block = "- (no conditional branches fired)"

    hints_block = json.dumps(plan_hints, indent=2) if plan_hints else "{}"

    # --- Reflection subsections (issues / stat tests / per-class / overfit) -- #
    structured = (reflection or {}).get("structured_issues") or []
    if structured:
        issues_block = "\n".join(
            f"- **[{i.get('severity', '?')}] {i.get('category', '')}** — "
            f"{i.get('message', '')}  \n  *Suggestion:* {i.get('suggestion', '')}"
            for i in structured
        )
    else:
        suggestions = (reflection or {}).get("suggestions", []) or []
        issues_block = (
            "\n".join(f"- {s}" for s in suggestions)
            if suggestions else "- (no issues detected)"
        )

    stat_comp = (reflection or {}).get("statistical_comparison") or {}
    comparisons = stat_comp.get("comparisons", []) if stat_comp else []
    if comparisons:
        rows = ["| vs | test | n_folds | p_value | mean_diff | significant |",
                "|---|---|---|---|---|---|"]
        for c in comparisons:
            p = c.get("p_value")
            md = c.get("mean_diff")
            rows.append(
                f"| {c.get('vs')} | {c.get('test')} | {c.get('n_folds', '-')} | "
                f"{f'{p:.4f}' if isinstance(p, float) else (c.get('reason') or '-')} | "
                f"{f'{md:+.4f}' if isinstance(md, float) else '-'} | "
                f"{'yes' if c.get('significant') else 'no'} |"
            )
        stat_block = "\n".join(rows)
    else:
        stat_block = "- (no candidates to compare)"

    per_class = (reflection or {}).get("per_class")
    if per_class and per_class.get("per_class_f1"):
        pc_lines = [
            f"- `{cls}` → F1 = {f1:.3f}"
            for cls, f1 in per_class["per_class_f1"].items()
        ]
        worst = per_class.get("worst_class")
        if worst is not None:
            pc_lines.append(
                f"- **Worst class:** `{worst}` "
                f"(F1={per_class.get('worst_f1', 0.0):.3f}); "
                f"gap to best = {per_class.get('f1_gap', 0.0):.3f}"
            )
        per_class_block = "\n".join(pc_lines)
    else:
        per_class_block = "- (binary or no predictions available)"

    overfit = (reflection or {}).get("overfit")
    if overfit:
        overfit_block = (
            f"- Status: **{overfit.get('status')}**  \n"
            f"- Train F1: {overfit.get('train_f1', 0.0):.3f}  \n"
            f"- Test F1: {overfit.get('test_f1', 0.0):.3f}  \n"
            f"- Gap: {overfit.get('gap', 0.0):+.3f} "
            f"(threshold = {overfit.get('threshold', 0.15):.2f})"
        )
    else:
        overfit_block = "- (train metrics not available for best model)"

    numeric = dataset_profile.get("feature_types", {}).get("numeric", [])
    categorical = dataset_profile.get("feature_types", {}).get("categorical", [])
    notes = dataset_profile.get("notes", [])

    md = f"""# Agentic Data Scientist Report

**Run ID:** `{ctx.run_id}`  
**Started (UTC):** {ctx.started_at}  
**Dataset:** `{ctx.data_path}`  
**Target:** `{ctx.target}`  
**Fingerprint:** `{fingerprint}`  

## Dataset Profile
- Rows: **{dataset_profile["shape"]["rows"]}**
- Columns: **{dataset_profile["shape"]["cols"]}**
- Classification: **{dataset_profile.get("is_classification")}**
- Imbalance ratio: **{dataset_profile.get("imbalance_ratio")}**

**Feature Types**
- Numeric ({len(numeric)}): {short_list(numeric)}
- Categorical ({len(categorical)}): {short_list(categorical)}

**Notes**
{chr(10).join([f"- {n}" for n in notes]) if notes else "- (none)"}

## Plan
{chr(10).join([f"- {t}" for t in plan_steps])}

## Planner Rationale
{rationale_block}

## Applied Hints
```json
{hints_block}
```

## Results (Best Model)
**Model:** `{best.get("model")}`

- Accuracy: **{best.get("accuracy"):.3f}**
- Balanced accuracy: **{best.get("balanced_accuracy"):.3f}**
- Macro F1: **{best.get("f1_macro"):.3f}**
- Macro Precision: **{best.get("precision_macro"):.3f}**
- Macro Recall: **{best.get("recall_macro"):.3f}**

Top metrics (all candidates):
```json
{json.dumps(eval_payload.get("all_metrics", []), indent=2)}
```

## Reflection

### Issues (by severity)
{issues_block}

### Statistical Comparison (Wilcoxon over CV f1_macro)
{stat_block}

### Per-class Performance
{per_class_block}

### Overfit Check
{overfit_block}

# Artefacts
- Confusion matrix: {eval_payload.get("confusion_matrix_path")}

"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
