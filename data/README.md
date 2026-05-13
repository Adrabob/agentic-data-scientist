# `data/` — Dataset catalogue

This folder holds every classification dataset wired into the agentic pipeline for CE888 2025/26. The agent itself does **not** bundle these datasets into the submission zip — they are included for local reproducibility of the cross-dataset experiments in `report/REPORT.md`. Do not commit large external datasets beyond what is listed here.

> Convention: `python run_agent.py --data data/<file> --target <col>` for every entry. `--target auto` is shown when the last-column heuristic or priority-list inference in `tools.data_profiler.analyze_target` resolves the correct column without a user hint.

## Quick reference table

| File | Rows | Cols | Target | n_classes | Challenges | Verified |
|---|---|---|---|---|---|---|
| `example_dataset.csv` | 20 | 6 | `label` (auto) | 2 | Tiny sample, balanced — sanity only | ✅ |
| `demo.csv` | 20 | 6 | `label` (auto) | 2 | Clone of example_dataset used by `test_smoke_run` | ✅ |
| `WineQuality.csv` | 1,699 | 12 | `quality` (auto) | 6 | Mixed-dtype target (float w/ 94% ints), severe imbalance (60:1), skewed numerics | ✅ |
| `cancer_risk_factors.csv` | 2,000 | 21 | `Risk_Level` | 3 | 15:1 imbalance, ID column, mixed num/cat, case-sensitive target name | ✅ |
| `mobile_price_classification.csv` | 2,000 | 21 | `price_range` | 4 | Balanced 4-class, mostly integer-coded categoricals | ✅ |
| `water_potability.csv` | 3,276 | 10 | `Potability` | 2 | 24% missingness on `Sulfate`, 49% on `ph`, mild imbalance | ✅ |
| `mushrooms.csv` | 8,124 | 23 | `class` | 2 | 100% categorical (23 cols) — heavy one-hot expansion; optional stress test | ⚠ |

The "Verified" column marks the five datasets run end-to-end for the report. `mushrooms.csv` is included for completeness and to exercise the executor's retry-with-simpler-config path but is not in the primary result set.

---

## `example_dataset.csv` / `demo.csv`

- **Shape:** 20 × 6.
- **Target:** `label` (binary, 10/10 balanced).
- **Features:** `age` (int), `bmi` (float), `smoker` (bool-as-int), `steps_per_day` (int), `cholesterol` (int).
- **Source:** Hand-crafted synthetic data shipped with the repository for smoke testing.
- **Expected agent behaviour:**
  - EDA: `small_dataset` branch fires (rows < 1000) → `cv_folds=10`, LogReg / RF priority.
  - Planner: no data-quality tags (clean, no missing, low cardinality).
  - Model: LogisticRegression consistently wins with F1=1.0 — the dataset is deliberately separable.
- **Caveat:** Too small to draw any generalisation conclusions. Used only to guarantee the pipeline runs cold and to seed `JSONMemory` in CI.

---

## `WineQuality.csv`

- **Shape:** 1,699 × 12.
- **Target:** `quality`.
- **Features:** 11 numeric physico-chemical measurements (acidity, sugar, chlorides, alcohol, …).
- **Source:** Cortez, P., Cerdeira, A., Almeida, F., Matos, T. & Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties.* Decision Support Systems 47(4). UCI ID 186.
- **Expected agent behaviour:**
  - EDA: `detect_mixed_dtypes` flags `free_sulfur_dioxide`, `total_sulfur_dioxide`, and **`quality`** as Pattern-1 float-mostly-int columns. `apply_mixed_dtype_cast_fix` rounds and casts `quality` to nullable `Int64` so `analyze_target` correctly infers **classification**, not regression.
  - Planner: `cast_fix` + `log_transform` tags on skewed numerics.
  - Reflector: `[high] class_collapse` on the rare classes (3 and 8) — macro F1 drops below 0.4 despite high accuracy. Replan escalates to `class_weight` then (on retry) `smote` hint. No model in the candidate set ever gets both extremes right because the data truly doesn't separate there.

---

## `cancer_risk_factors.csv`

- **Shape:** 2,000 × 21.
- **Target:** `Risk_Level` (3 classes: Medium / Low / High — 1574 / 324 / 102 → 15.4:1 imbalance).
- **Features:** Mix of demographic (age, gender), behavioural (smoking, alcohol, diet), and lab-derived risk scores. `Patient_ID` is a unique identifier. `Overall_Risk_Score` is a continuous float — must **not** be picked as the target.
- **Source:** Public Kaggle dataset "Cancer Risk Factors for Early Detection" (Abhijit M., 2024). Used in the CE888 reference examples.
- **Expected agent behaviour:**
  - EDA: `Patient_ID` flagged as high-cardinality potential ID → `drop_ids` tag.
  - Planner: `imbalance_resample` fires (`class_weight` at ratio ≥ 3, bumps to `smote` hint at ≥ 10).
  - Target resolution: `Risk_Level` is case-sensitive; `clean_column_names` lowercases it, but `analyze_target` now tries the normalised match. Running with `--target auto` picks `Overall_Risk_Score` (continuous) — **always pass `--target Risk_Level` explicitly**.
  - Model: GradientBoosting wins at F1 ≈ 0.997 (tie with RF within Wilcoxon p > 0.05 → `[low] no_stat_difference` reflection only).
- **Caveat:** The per-class F1 gap is tiny because the dataset is essentially separable once the ID is removed — good for stress-testing stats tests, not for demonstrating class collapse.

---

## `mobile_price_classification.csv`

- **Shape:** 2,000 × 21.
- **Target:** `price_range` (4 balanced classes: 0 / 1 / 2 / 3 — 500 each).
- **Features:** 20 phone specs (`battery_power`, `ram`, `px_height`, …). Most integer-coded; `m_dep` and `clock_speed` are floats.
- **Source:** Public Kaggle dataset "Mobile Price Classification" (Abhishek Sharma, 2018).
- **Expected agent behaviour:**
  - EDA: clean — no missing, no IDs, balanced.
  - Planner: no branches other than base template.
  - Model: LogisticRegression wins convincingly (bal_acc ≈ 0.965). RAM and pixel dimensions are close to linearly separable for price.
- **Caveat:** Multiclass with balanced classes — useful for demonstrating that the agent does **not** trigger unnecessary replans when there is no issue. Used as the control in the report's replan-trigger analysis.

---

## `water_potability.csv`

- **Shape:** 3,276 × 10.
- **Target:** `Potability` (binary: 0 = not potable, 1 = potable — 1998 / 1278, mild 1.56:1 imbalance).
- **Features:** 9 numeric water-quality measurements (`ph`, `Hardness`, `Solids`, `Chloramines`, `Sulfate`, `Conductivity`, `Organic_carbon`, `Trihalomethanes`, `Turbidity`).
- **Source:** Public Kaggle dataset "Water Quality" (Aditya Kadiwal, 2021).
- **Expected agent behaviour:**
  - EDA: `ph` (~49% missing) + `Sulfate` (~24% missing) + `Trihalomethanes` (~5% missing) trigger `missing_moderate` / `missing_severe` branches. Median imputation in the preprocessor handles the moderate band; the severe branch merely recommends drop — the column stays in because the orchestrator does not currently act on `impute_strategy.drop_col` (emit-only).
  - Reflector: overfit is frequent on RF → `[high] overfit` → regularise replan prepends LogReg.
  - Model: RandomForest tends to win with bal_acc ≈ 0.6 — genuinely a hard problem; nobody in the literature breaks 0.7 without feature engineering the pipeline does not attempt.
- **Caveat:** Good showcase for the reflect→replan loop. The second iteration often swaps to LogReg under the regularise template; reflection then flags the LogReg run as underfit, and the orchestrator respects `max_replans` to stop.

---

## `mushrooms.csv` *(optional)*

- **Shape:** 8,124 × 23.
- **Target:** `class` (binary: e = edible / p = poisonous, 4208 / 3916, balanced).
- **Features:** 22 purely categorical attributes (cap shape, odor, stalk shape, spore colour, …). One column (`stalk-root`) has "?" sentinels for missingness.
- **Source:** UCI Machine Learning Repository, *Mushroom Data Set*, Schlimmer (1987).
- **Expected agent behaviour:**
  - EDA: every feature is categorical → all 22 flagged for potential target-encoding (some are high-cardinality).
  - Preprocessor: one-hot expansion balloons to ~120 dummy columns. `select_models` drops SVC (cols > 200 guard is not tripped, but training time climbs sharply).
  - Model: LogReg or RF both hit >99% F1. The interest here is stress-testing the **executor retry path** — if one-hot memory blows on a weaker machine, the retry falls back to LogReg + drop-categoricals.
- **Caveat:** Not part of the primary 5-dataset result set because its near-perfect separability makes it a poor case study for reflection / replan behaviour. Included for CI-size / memory-pressure coverage.

---

## Adding a new dataset

1. Drop the CSV into `data/`.
2. Try `python run_agent.py --data data/<new>.csv --target auto` first — the priority-list + last-column heuristic in `tools.data_profiler.analyze_target` catches most cases.
3. If the target is case-sensitive or non-trivial, pass `--target <exact name>` (the column-name cleaning is matched on read).
4. Add an entry to the quick-reference table and a short section below following the structure above.
5. Confirm a run produces all 7 artefacts in `outputs/<run_id>/` (`report.md`, `plan.json`, `metrics.json`, `reflection.json`, `eda_summary.json`, `confusion_matrix.png`, `run.log`).
