# CE888 Agentic Data Scientist — Technical Report

**Student:** Arda Kaya (2501529)
**Module:** CE888 Data Science & Decision Making, University of Essex, 2025/26

---

## 1. Introduction

Modern data science workflows involve a recurring cycle: loading data, profiling it for quality issues, selecting appropriate models, evaluating results, and iterating when performance is unsatisfactory. This report presents an offline, rule-based agentic system that automates this cycle for tabular classification tasks. The term "agentic" refers to the system's capacity to make autonomous decisions at each stage — choosing which columns to drop, which models to prioritise, whether to replan after reflection — without human intervention once a dataset and target column are provided.

The system is implemented entirely in Python using scikit-learn, pandas, NumPy, SciPy, and standard library modules. Crucially, it uses no large language models, no external APIs, and no AutoML frameworks. Every decision is driven by explicit heuristics derived from the dataset's statistical profile. This design choice ensures full transparency: every branch the agent takes can be traced to a measurable signal (e.g., imbalance ratio exceeding a threshold, skewness beyond a cutoff, or a class collapsing to zero F1).

The system was evaluated on five diverse classification datasets ranging from a 20-row synthetic sanity check to a 3,276-row water quality dataset with nearly 50% missingness. Results demonstrate that the agent adapts its behaviour meaningfully across datasets — triggering mixed-dtype repairs for WineQuality, ID-column removal for cancer risk factors, and overfit-driven replans for water potability — while avoiding unnecessary interventions on clean, balanced data like the mobile price dataset.

The remainder of this report describes the architecture (Section 2), dataset understanding pipeline (Section 3), conditional planning logic (Section 4), modelling and evaluation strategy (Section 5), reflection and re-planning mechanisms (Section 6), memory and learning subsystem (Section 7), ethical considerations and limitations (Section 8), and conclusions with future work (Section 9).

## 2. System Architecture

The system follows a sequential pipeline with an iterative replan loop. The entry point is a CLI (`run_agent.py`) that parses arguments and delegates to the `AgenticDataScientist` orchestrator class. The orchestrator coordinates six functional stages:

```
run_agent.py (CLI)
  └─> AgenticDataScientist.run()
       ├─ Stage 1: EDA Profiling        (tools/data_profiler.py)
       ├─ Stage 2: Memory Lookup        (agents/memory.py)
       ├─ Stage 3: Planning             (agents/planner.py)
       └─ REPLAN LOOP (max_replans):
            ├─ Stage 4: Preprocessing + Training   (tools/modelling.py)
            ├─ Stage 5: Evaluation                 (tools/evaluation.py)
            ├─ Stage 6: Reflection                 (agents/reflector.py)
            └─ if should_replan → mutate plan → continue loop
```

**Stage 1 — EDA Profiling.** The `run_eda` function reads the CSV, cleans column names (lowercasing, deduplication, dropping pandas-generated `Unnamed:` columns), detects mixed-dtype columns using a pattern-based heuristic, optionally repairs them by rounding and casting to nullable `Int64`, and produces a comprehensive profile dictionary. This profile contains both flat pipeline-compatible keys (used downstream by the planner and modelling modules) and nested structures (for the EDA visualisations and markdown report). The profiler also generates distribution plots saved as PNG artefacts.

**Stage 2 — Memory Lookup.** Before planning, the orchestrator computes a SHA-256 fingerprint of the dataset (over its shape, target column, and sorted column names) and queries a JSON-backed memory store. An exact fingerprint match retrieves the prior run's best model and applied hints. If no exact match exists, similarity-based retrieval searches for the nearest prior dataset using a normalised Manhattan distance over five dimensions (Section 7). The resulting memory hint, if any, is passed to the planner.

**Stage 3 — Planning.** The conditional planner evaluates ten boolean branches over the EDA profile (Section 4) and emits a structured plan containing tagged steps, human-readable rationale, and a machine-consumable hints dictionary. The hints dictionary drives all downstream behaviour — from which columns to drop to which models to prioritise.

**Stages 4-5 — Training and Evaluation.** The modelling module builds a scikit-learn `ColumnTransformer` preprocessor (median imputation + standard scaling for numeric features; mode imputation + one-hot encoding for categoricals), selects candidate models based on dataset size and imbalance, trains each with a stratified train/test split, and computes per-fold cross-validation scores for statistical testing. Failed candidates are isolated via try/except so a single model's error does not abort the run. The evaluation module ranks candidates by balanced accuracy (then macro F1 as tiebreaker) and persists a confusion matrix PNG.

**Stage 6 — Reflection.** The reflector analyses the training results through four lenses: overfit detection (train-vs-test F1 gap), per-class performance analysis, statistical comparison of candidates via the Wilcoxon signed-rank test, and comparison against the dummy baseline. Each finding is tagged with a severity level (high/medium/low) and an actionable suggestion. If any high-severity issue is detected and the replan budget remains, the reflector mutates the plan's hints and the loop continues with a new iteration.

**Orchestrator hardening.** The orchestrator uses Python's `logging` module for structured, timestamped logging to both console and a per-run `run.log` file. A retry mechanism catches full pipeline failures: on the first failure, it simplifies the configuration (LogisticRegression + Dummy only, dropping all categorical columns to avoid one-hot expansion issues) and retries once. A `_step_enabled` helper maps each plan step to its gating hint, logging which steps are active or skipped per iteration.

Each run produces seven artefacts in `outputs/<timestamp>/`: `report.md`, `eda_summary.json`, `plan.json`, `metrics.json`, `reflection.json`, `confusion_matrix.png`, and `run.log`.

## 3. Dataset Understanding

The EDA module extracts a rich set of signals that directly inform the planner's conditional branches. Each signal category addresses a specific class of data-quality or structural issue that, if left unhandled, would degrade model performance or cause pipeline failures.

**Missing values.** For each column, the profiler computes the percentage of null entries. Columns with 5-40% missingness trigger the `missing_moderate` planner branch (median/mode imputation); columns exceeding 40% trigger `missing_severe` (recommending column removal, though the current implementation retains all columns with imputation). In the water potability dataset, `ph` has 49% missingness and `Sulfate` has 24%, exercising both thresholds.

**Skewness and outliers.** Numeric columns are tested for skewness (threshold: |skew| > 2.0) and outliers (values beyond 1.5x IQR). Highly skewed columns are flagged for log-transform in the planner's `log_transform` branch. While the current preprocessor does not execute the transform (it remains a hint for future extension), the signal is documented in the plan rationale and markdown report.

**High cardinality and ID columns.** Columns where the number of unique values exceeds 90% of the row count are flagged as potential identifiers. The planner's `drop_ids` branch adds these to the `drop_columns` hint, and the preprocessor filters them out before fitting. This prevents the model from memorising row-specific information. In the cancer risk factors dataset, `Patient_ID` is correctly identified and removed.

**Imbalance detection.** The profiler computes the ratio between the largest and smallest class counts. An imbalance ratio of 3-10 triggers `class_weight="balanced"` in candidate models; a ratio above 10 additionally hints at SMOTE resampling. The cancer risk factors dataset (ratio 15.4:1 across three classes) and WineQuality (ratio 60.7:1 across six classes) both activate this branch.

**Mixed-dtype detection.** A pattern-based heuristic flags numeric columns where 80-99% of non-null values are integer-like. This catches a common data ingestion issue: a column like WineQuality's `quality` contains values such as 3.0, 4.0, 5.0 alongside 3.5 and 4.5, causing pandas to infer float64 when the column is semantically ordinal integer. Without repair, `analyze_target` would classify this as a regression problem, crashing the classification pipeline. The `apply_mixed_dtype_cast_fix` function rounds and casts such columns to nullable `Int64`, resolving the issue transparently. In the WineQuality run, three columns are repaired: `free_sulfur_dioxide` (94.2% int-like), `total_sulfur_dioxide` (94.3%), and `quality` (94.1%).

**Target inference.** The `analyze_target` function resolves the target column through a three-step cascade: (1) exact match against the user-supplied name, (2) normalised match (lowercasing + special-character stripping) to handle the common case where `clean_column_names` has transformed the original name, and (3) automatic detection using a priority keyword list (`label`, `target`, `class`, `outcome`, `y`) with fallback to the rightmost column. Step 2 was added to fix a bug where `--target Risk_Level` silently failed after column cleaning lowercased it to `risk_level`.

## 4. Planning Logic

The planner evaluates ten conditional branches over the EDA profile and emits a structured plan:

```python
{
  "steps":     ["profile_dataset", "build_preprocessor[tags]", ...],
  "rationale": {"tag": "human-readable reason", ...},
  "hints":     {"drop_columns": [...], "priority_models": [...], ...}
}
```

The hints dictionary is the machine-consumable output that drives all downstream decisions. Every key is always present with a sensible default (empty list, `None`, or `"default"`), so downstream code never needs to check for missing keys.

### Branch Table

| Branch Tag | Trigger Condition | Effect on Hints |
|---|---|---|
| `drop_ids` | High-cardinality potential ID columns detected | `drop_columns` populated |
| `log_transform` | Skewed numeric columns (skew > 2.0) | `log_transform_cols` populated |
| `target_encode` | High-cardinality categoricals | `target_encode_cols` populated |
| `cast_fix` | Mixed-dtype columns detected | `cast_fix_cols` populated |
| `imbalance_resample` | Imbalance status is "imbalanced" or "severe" | `resample_strategy` set to `class_weight` or `smote` |
| `missing_severe` | Any column >40% missing | `impute_strategy` updated |
| `missing_moderate` | Any column 5-40% missing | `impute_strategy` updated |
| `small_dataset` | Rows < 1000 | `cv_folds=10`, priority LogReg/RF |
| `high_dim` | Columns > 100 | `plan_template="high_dim"`, priority RF/GB |
| `memory_priority` | Memory hint with prior best model | Best model prepended to `priority_models` |

### Worked Example 1: WineQuality

The WineQuality dataset (1,459 rows, 12 columns, 6-class target `quality`) triggers three branches:

1. **`log_transform`** — 5 skewed columns flagged: `residual_sugar`, `chlorides`, `free_sulfur_dioxide`, `total_sulfur_dioxide`, `sulphates`.
2. **`imbalance_resample`** — Imbalance ratio 60.7 (class 3 has 24 samples vs. class 5 with 1,457 in the full wine dataset) triggers `resample_strategy = "class_weight"`.
3. **`cast_fix`** — 3 float-mostly-int columns detected (repaired before planning, but documented in rationale).

The plan emits `priority_models = []` (no memory hint available on first run), so the default candidate ordering applies. After the first iteration's reflection detects class collapse (worst-class F1 = 0.0) and mild overfit, the replan loop fires twice: first flipping `resample_strategy` from `class_weight` to `smote` (hint only), then prepending `LogisticRegression` to `priority_models` under the `regularised` template.

### Worked Example 2: cancer_risk_factors

The cancer risk factors dataset (2,000 rows, 21 columns, 3-class target `Risk_Level`) triggers:

1. **`drop_ids`** — `Patient_ID` flagged and added to `drop_columns`.
2. **`log_transform`** — 3 skewed columns flagged.
3. **`target_encode`** — `Patient_ID` also flagged as high-cardinality categorical (redundant with drop, but the branch fires independently).
4. **`imbalance_resample`** — Ratio ~15 triggers `class_weight`.

The resulting model (GradientBoosting) achieves balanced accuracy 0.999 with no replan needed — the agent correctly recognises that the dataset is essentially separable once the ID column is removed.

### Memory-Informed Planning

On a second run of any dataset, the fingerprint-based memory lookup retrieves the prior best model and prepends it to `priority_models` via the `memory_priority_exact` branch. For novel datasets, similarity-based retrieval (Section 7) can fire the `memory_priority_similar` branch. In the water potability run, the nearest prior dataset (similarity score 0.844) was the example dataset, whose best model (LogisticRegression) was promoted — a reasonable choice given the overfit-prone nature of the data.

## 5. Modelling and Evaluation

### Candidate Selection

The modelling module selects candidates from a fixed pool based on dataset characteristics:

- **DummyMostFrequent** — always included as a baseline.
- **LogisticRegression** — always included; receives `class_weight="balanced"` when the imbalance ratio exceeds 3.
- **RandomForest** — always included (max 200 estimators); `class_weight="balanced"` when imbalanced.
- **GradientBoosting** — included when rows <= 50,000.
- **SVC (RBF kernel)** — included when rows <= 20,000 and columns <= 200; `class_weight="balanced"` when imbalanced.

The `priority_models` hint from the planner reorders this list (without adding models not in the candidate pool). This ensures that when the reflector prepends LogisticRegression during a regularisation replan, it is trained first but all other viable candidates still run.

### Training and Cross-Validation

Each candidate is wrapped in a scikit-learn `Pipeline` with the `ColumnTransformer` preprocessor and trained on a stratified 80/20 train/test split (configurable via `--test_size`). After fitting, the module computes:

- **Test metrics:** accuracy, balanced accuracy, and macro F1 on the held-out test set.
- **Train metrics:** the same three metrics on the training set (for overfit detection).
- **Per-fold CV scores:** `StratifiedKFold` cross-validation on the training set, with folds clamped to the minimum class count when a class has fewer than 5 samples. These per-fold macro F1 scores feed the Wilcoxon test in the reflector.

Per-model error isolation ensures that if one candidate fails (e.g., SVC running out of memory on a large one-hot-expanded dataset), the remaining candidates still produce results.

### Evaluation and Ranking

Candidates are ranked by balanced accuracy (descending), with macro F1 as a tiebreaker. Balanced accuracy was chosen over raw accuracy because four of the five evaluation datasets exhibit some degree of class imbalance — raw accuracy would favour majority-class predictors.

### Cross-Dataset Results

| Dataset | n_rows | n_cols | Best Model | Bal. Acc. | F1 Macro | Replans | Top Issue |
|---|---|---|---|---|---|---|---|
| example_dataset | 20 | 6 | LogisticRegression | 1.000 | 1.000 | 0 | - |
| cancer_risk_factors | 2,000 | 21 | GradientBoosting | 0.999 | 0.997 | 0 | [low] imbalance |
| mobile_price_classification | 2,000 | 21 | LogisticRegression | 0.965 | 0.965 | 0 | [low] no_stat_difference |
| WineQuality | 1,459 | 12 | SVC_RBF | 0.388 | 0.321 | 2 | [high] class_collapse |
| water_potability | 3,276 | 10 | RandomForest | 0.601 | 0.593 | 2 | [high] overfit |

The results show clear variation in agent behaviour: clean datasets (example, mobile) require no replanning; moderately challenging datasets (cancer) trigger imbalance handling but no replan; and genuinely difficult datasets (WineQuality, water) push the agent through its full reflect-replan loop.

## 6. Reflection and Re-planning

The reflector analyses training results through four complementary lenses, each producing severity-tagged findings that drive replan decisions.

### 6.1 Overfit Detection

The reflector compares train-set macro F1 against test-set macro F1. A gap exceeding 0.15 flags overfitting; a negative gap exceeding -0.05 flags potential underfitting. In the water potability run, the initial RandomForest achieved train F1 = 1.000 vs. test F1 = 0.593 — a gap of 0.407, triggering a high-severity overfit issue. The replan strategy responded by setting `plan_template = "regularised"` and prepending LogisticRegression (a simpler model less prone to memorisation) to the priority list.

### 6.2 Per-Class Performance Analysis

Using scikit-learn's `classification_report(output_dict=True)`, the reflector identifies the worst-performing class and computes the F1 gap between best and worst classes. A worst-class F1 below 0.3 in a multiclass problem triggers a high-severity `class_collapse` issue. In the WineQuality run, class "3" achieved F1 = 0.000 (complete collapse due to only ~24 samples in that class). This triggered the `replan_class_weight` strategy on the first replan, escalating to `replan_resample` (hinting SMOTE) on the second.

### 6.3 Statistical Comparison (Wilcoxon Signed-Rank Test)

For each pair of (best model, alternative), the reflector extracts per-fold CV macro F1 scores and runs a Wilcoxon signed-rank test — a non-parametric test chosen because: (a) the number of folds is small (typically 5), making normality assumptions unreliable; (b) the test is paired, respecting that each fold uses the same data partition; and (c) it is robust to outlier folds.

A finding of "no significant difference" (p > 0.05) between the best model and an alternative is reported as a low-severity issue with the suggestion to favour the simpler model. Importantly, this finding alone does not trigger a replan — near-perfect runs like cancer_risk_factors often show no significant difference at the top simply because all models perform well, and forcing a replan in such cases would be counterproductive.

In the WineQuality run, no candidate was significantly better than any other (all Wilcoxon p > 0.05), confirming that the dataset's inherent difficulty (6-class ordinal with extreme imbalance) limits all models equally.

### 6.4 Replan Strategy Selection

The `should_replan` function triggers when any high-severity issue exists and the replan budget has not been exhausted. Four replan strategies are available:

1. **Escalate models** — when the best model barely beats the dummy baseline, prepend GradientBoosting and RandomForest to `priority_models`.
2. **Regularise** — when overfit is detected, switch to the `regularised` plan template and prepend LogisticRegression.
3. **Class weight** — when a class collapses and no resample strategy is active, set `resample_strategy = "class_weight"`.
4. **SMOTE flip** — when `class_weight` was already applied but class collapse persists, flip to `resample_strategy = "smote"` (hint only; actual SMOTE execution is a future extension).

These strategies compose across iterations: WineQuality's run applied `class_weight` on replan 1, then `SMOTE flip + regularise` on replan 2, exhausting the budget at `max_replans = 2`.

### 6.5 Reflection Leading to Behaviour Change

The rubric emphasises that reflection must lead to meaningful changes. The cross-dataset results validate this:
- **mobile_price_classification** — no issues above [low] severity; no replan fired. The agent correctly avoids unnecessary interventions.
- **water_potability** — [high] overfit on iteration 1; replan switches to regularised template with LogisticRegression priority. Iteration 2's reflection still flags overfit (inherent to the data) but the budget is exhausted, so the agent accepts the best available result.
- **WineQuality** — [high] class_collapse on iteration 1; replan escalates resample strategy. Iteration 2 adds regularisation. Final best (SVC_RBF, F1 0.321) is honestly reported as poor — the agent does not mask failure.

## 7. Memory and Learning

The memory subsystem uses a JSON-backed store (`agent_memory.json`) that persists across runs. Each record is keyed by the dataset's SHA-256 fingerprint and contains:

- **Core fields:** target column, shape, best model name and metrics, timestamp.
- **Feature summary:** `n_rows`, `n_cols`, `n_numeric`, `n_categorical`, `imbalance_ratio`, `has_missing` — a compact statistical signature of the dataset.
- **Applied hints:** a snapshot of the final `plan["hints"]` dictionary, capturing what the agent decided to do.
- **Top 3 models:** a list of `(model_name, balanced_accuracy)` tuples for quick reference.
- **Reflection issues:** the severity-tagged structured issues from the reflector.

### Exact vs. Similarity Retrieval

When the orchestrator queries memory, it first attempts an exact fingerprint match. If the same dataset has been run before, its prior best model is retrieved and prepended to `priority_models` via the `memory_priority_exact` planner branch. This is a strong signal: the agent has direct experience with this exact data.

When no exact match exists, the system falls back to similarity-based retrieval. The `find_similar_datasets` method computes a normalised Manhattan distance over five dimensions:

1. `log10(1 + n_rows) / 7.0` — normalised row count (log-scaled to handle 20-row vs. 100k-row datasets).
2. `log10(1 + n_cols) / 3.0` — normalised column count.
3. `n_numeric / (n_numeric + n_categorical)` — numeric feature ratio.
4. `n_categorical / (n_numeric + n_categorical)` — categorical feature ratio.
5. `log10(1 + imbalance_ratio) / 2.0` — normalised imbalance severity.

The similarity score is `1 / (1 + distance)`. Only matches above 0.7 are used, tagged with `memory_priority_similar` in the plan rationale. This threshold was chosen empirically: below 0.7, the prior dataset's characteristics diverge enough that its best model recommendation is unreliable.

In the water potability run, the nearest prior dataset was the example dataset (similarity 0.844). Despite the datasets being quite different in difficulty, both share a similar shape profile (low column count, numeric-heavy, mild or no imbalance). The transferred recommendation (LogisticRegression) was a reasonable starting point that the reflector then validated independently.

## 8. Ethics and Limitations

### Transparency

The system's rule-based architecture provides full decision transparency. Every planner branch, replan trigger, and model selection can be traced to a specific numerical threshold in the EDA profile. This contrasts with LLM-based agents where the reasoning chain is opaque. However, the thresholds themselves (e.g., imbalance ratio 3.0 for class_weight, skewness 2.0 for log-transform) are heuristic and may not generalise to all domains.

### Limitations

**Single train/test split.** The primary evaluation uses a single stratified 80/20 split. Cross-validation is used only for statistical comparison between models, not for the primary performance estimate. This means reported metrics have higher variance than a full nested cross-validation setup would produce. For the 20-row example dataset, this is particularly acute — the test set contains only 4 samples.

**Hint-only transforms.** Several planner branches (log-transform, target-encoding, SMOTE resampling) emit hints that are not yet executed by the preprocessor. The system documents the recommendation but does not act on it. This is an honest limitation: the infrastructure is in place for future extension, but the current pipeline only executes column dropping, median/mode imputation, standard scaling, and one-hot encoding.

**Fixed model pool.** The candidate set is hardcoded (Dummy, LogReg, RF, GB, SVC). There is no hyperparameter tuning beyond `class_weight="balanced"`. Datasets that require specialised architectures (e.g., deep trees for highly non-linear boundaries) or fine-tuned regularisation cannot be served optimally.

**Similarity retrieval naivety.** The five-dimensional distance metric treats all dimensions equally and uses Manhattan distance, which may not capture important structural differences (e.g., two datasets with identical shapes but completely different feature distributions). A more sophisticated approach might use learned embeddings or distribution-aware metrics.

### Fairness and Bias

The system does not perform any fairness analysis. Class imbalance handling (via `class_weight` or SMOTE hints) addresses predictive imbalance but does not consider protected attributes or disparate impact. For real-world deployment in domains like healthcare (cancer risk) or environmental justice (water potability), fairness audits would be essential.

## 9. Conclusion and Future Work

This report presented an offline, rule-based agentic data scientist that autonomously profiles datasets, plans workflows, trains and evaluates models, reflects on results, and learns from prior runs. Across five diverse classification datasets, the system demonstrated meaningful adaptation: avoiding unnecessary interventions on clean data, triggering appropriate repairs for data-quality issues, and iteratively replanning when reflection identified overfit or class collapse.

The strongest contributions are the conditional planner (10 branches driven by EDA signals), the multi-lens reflector (overfit detection, per-class analysis, Wilcoxon statistical tests), and the similarity-based memory retrieval. Together, these components show that useful autonomous reasoning is achievable with explicit heuristics — no language model required.

Future work should address the current hint-only limitations by implementing log-transform, target-encoding, and SMOTE execution in the preprocessor. Adding hyperparameter tuning (e.g., grid search over a small parameter space) and nested cross-validation would improve both model performance and metric reliability. Finally, incorporating fairness metrics and distribution-aware similarity measures would make the system more robust for real-world deployment.

*AI Disclosure*: This project involved limited use of AI-assisted tools for general guidance on structure and concepts. All implementation, testing, and final decisions were carried out independently by the author.

## References

1. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, pp. 2825-2830.
2. McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*, pp. 56-61.
3. Virtanen, P. et al. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. *Nature Methods*, 17, pp. 261-272.
4. Wilcoxon, F. (1945). Individual Comparisons by Ranking Methods. *Biometrics Bulletin*, 1(6), pp. 80-83.
5. Cortez, P. et al. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, 47(4), pp. 547-553.
6. Chawla, N.V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, pp. 321-357.


