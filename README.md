# CE888 Agentic Data Scientist

An **offline, LLM-free** agentic system for end-to-end classification. The agent profiles a CSV, plans a workflow based on data characteristics, trains and evaluates multiple models, reflects on results with statistical tests, and optionally re-plans — all using rule-based heuristics and a persistent JSON memory.

---

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate        # Windows (Git Bash)
# source venv/bin/activate          # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the agent
python run_agent.py --data data/example_dataset.csv --target auto

# 4. Check outputs
ls outputs/          # timestamped run directory with 7 artefacts
```

---

## System Architecture

```
run_agent.py (CLI)
  └─> AgenticDataScientist.run()          agentic_data_scientist.py
       │
       ├─ Stage 1: EDA Profiling           tools/data_profiler.py
       │    ├─ Column cleaning + mixed-dtype auto-repair
       │    ├─ Missing / skew / outlier / cardinality analysis
       │    └─ Target inference (priority-list + heuristic fallback)
       │
       ├─ Stage 2: Memory Lookup           agents/memory.py
       │    ├─ Exact fingerprint match
       │    └─ Similarity-based retrieval (normalised Manhattan, 5 dims)
       │
       ├─ Stage 3: Conditional Planning    agents/planner.py
       │    └─ 10 branches → {steps, rationale, hints} dict
       │
       └─ REPLAN LOOP (≤ max_replans):
            ├─ Stage 4: Preprocessing + Training   tools/modelling.py
            │    ├─ ColumnTransformer (median/scale + mode/one-hot)
            │    ├─ 5 candidate models (Dummy, LogReg, RF, GB, SVC)
            │    └─ Per-fold CV scores + train metrics
            ├─ Stage 5: Evaluation                 tools/evaluation.py
            │    └─ Rank by balanced_accuracy, then f1_macro
            ├─ Stage 6: Reflection                 agents/reflector.py
            │    ├─ Overfit detection (train-test F1 gap)
            │    ├─ Per-class F1 analysis
            │    ├─ Wilcoxon signed-rank test between models
            │    └─ Severity-tagged issues → replan trigger
            └─ if should_replan → mutate hints → continue loop
```

Each run produces **7 artefacts** in `outputs/<timestamp>/`:
`report.md`, `eda_summary.json`, `plan.json`, `metrics.json`, `reflection.json`, `confusion_matrix.png`, `run.log`

---

## How to Run

### Basic usage

```bash
python run_agent.py --data data/example_dataset.csv --target auto
```

### All options

```bash
python run_agent.py \
    --data data/<file>.csv \
    --target <column_name_or_auto> \
    --output_root outputs \
    --seed 42 \
    --test_size 0.2 \
    --max_replans 2 \
    --quiet
```

| Argument | Default | Description |
|---|---|---|
| `--data` | *(required)* | Path to CSV dataset |
| `--target` | *(required)* | Target column name or `auto` for automatic detection |
| `--output_root` | `outputs` | Root directory for run outputs |
| `--seed` | `42` | Random seed for reproducibility |
| `--test_size` | `0.2` | Test set fraction |
| `--max_replans` | `1` | Maximum re-planning iterations |
| `--quiet` | `False` | Reduce console output |

---

## Planning

The conditional planner (`agents/planner.py`) evaluates **10 branches** over the EDA profile:

| Branch | Trigger | Action |
|---|---|---|
| `drop_ids` | High-cardinality ID columns | Add to `drop_columns` hint |
| `log_transform` | Skewed numerics (skew > 2.0) | Flag for log-transform |
| `target_encode` | High-cardinality categoricals | Flag for target-encoding |
| `cast_fix` | Mixed-dtype columns (80-99% int-like) | Round + cast to Int64 |
| `imbalance_resample` | Imbalance ratio >= 3 | Set `class_weight` or `smote` |
| `missing_severe` | Any column >40% missing | Update impute strategy |
| `missing_moderate` | Any column 5-40% missing | Update impute strategy |
| `small_dataset` | Rows < 1000 | 10-fold CV, LogReg/RF priority |
| `high_dim` | Columns > 100 | RF/GB priority |
| `memory_priority` | Prior best model in memory | Prepend to candidate list |

The planner outputs a structured `{steps, rationale, hints}` dict that drives all downstream behaviour.

---

## Reflection and Re-planning

The reflector (`agents/reflector.py`) analyses results through four lenses:

1. **Overfit detection** — train-vs-test F1 gap > 0.15 triggers high-severity flag
2. **Per-class performance** — worst-class F1 < 0.3 triggers class_collapse flag
3. **Statistical comparison** — Wilcoxon signed-rank test over per-fold CV scores
4. **Baseline comparison** — best model vs. DummyMostFrequent gap

Re-plan strategies:
- **Escalate models** — prepend GradientBoosting/RF when barely beating dummy
- **Regularise** — switch to LogisticRegression when overfit detected
- **Class weight / SMOTE** — escalate resampling when classes collapse

---

## Memory and Learning

The memory system (`agents/memory.py`) persists run results in `agent_memory.json`:

- **Exact retrieval**: SHA-256 fingerprint match for repeated datasets
- **Similarity retrieval**: Normalised Manhattan distance over 5 dimensions (rows, cols, numeric ratio, categorical ratio, imbalance ratio). Threshold: similarity > 0.7
- **Stored per record**: feature summary, applied hints, top-3 models, reflection issues

---

## Testing

```bash
# Run all tests with coverage
pytest tests/

# Quick sanity check
python tests/sanity_check.py
```

**Coverage:** 82.92% across `agents/` and `tools/` (58 tests, floor locked at 60% via `pyproject.toml`).

---

## Datasets

Seven classification datasets are documented in `data/README.md`:

| Dataset | Rows | Target | Key Challenge |
|---|---|---|---|
| `example_dataset.csv` | 20 | `label` | Sanity check (separable) |
| `WineQuality.csv` | 1,699 | `quality` | Mixed-dtype target, 60:1 imbalance |
| `cancer_risk_factors.csv` | 2,000 | `Risk_Level` | 15:1 imbalance, ID column |
| `mobile_price_classification.csv` | 2,000 | `price_range` | Balanced 4-class (control) |
| `water_potability.csv` | 3,276 | `Potability` | 49% missing on `ph`, overfit-prone |

---

## Advanced Features

1. **Wilcoxon signed-rank statistical testing** — Non-parametric paired test over per-fold CV scores to compare model candidates. Reports p-values and significance in `reflection.json`.

2. **Similarity-based memory retrieval** — When no exact dataset match exists, finds the nearest prior dataset using normalised Manhattan distance over 5 feature dimensions. Transfers the prior best model recommendation when similarity exceeds 0.7.

3. **Mixed-dtype auto-repair** — Pattern-1 detection flags float columns that are 80-99% integer-like (e.g., WineQuality's `quality`). Automatically rounds and casts to nullable `Int64`, preventing misclassification as regression.

4. **Multi-strategy re-planning** — Four distinct replan strategies (escalate models, regularise, class_weight, SMOTE flip) compose across iterations, each triggered by specific reflection findings.

5. **10-branch conditional planner** — Data-driven workflow planning with tagged steps and machine-consumable hints, adapting the pipeline to dataset characteristics.

---

## Project Structure

```
ce888-agentic-data-scientist/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Pytest + coverage configuration
├── agentic_data_scientist.py       # Core orchestrator (executor)
├── run_agent.py                    # CLI entry point
├── agents/
│   ├── planner.py                  # Conditional planner (10 branches)
│   ├── reflector.py                # Statistical reflection + replan
│   └── memory.py                   # JSON memory with similarity search
├── tools/
│   ├── data_profiler.py            # EDA profiling + mixed-dtype repair
│   ├── modelling.py                # Preprocessing + model training
│   └── evaluation.py               # Ranking + report generation
├── data/
│   ├── README.md                   # Dataset catalogue (7 datasets)
│   └── *.csv                       # Classification datasets
├── report/
│   ├── README.md                   # Report structure guide
│   └── REPORT.md                   # Technical report (3834 words)
├── tests/
│   ├── test_planner.py             # 14 planner unit tests
│   ├── test_reflector.py           # 16 reflector unit tests
│   ├── test_executor.py            # 8 executor unit tests
│   ├── test_memory.py              # 8 memory unit tests
│   ├── test_data_profiler.py        # 11 EDA/profiler unit tests
│   ├── test_smoke_run.py           # End-to-end smoke test
│   └── sanity_check.py             # Quick sanity check
└── outputs/                        # Run outputs (gitignored)
```

