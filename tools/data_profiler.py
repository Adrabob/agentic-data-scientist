
"""eda

**Arda Kaya**

**Student Number: 2501529**

Pipeline-compatible EDA module for the CE888 Agentic Data Scientist.

Originally an interactive Jupyter notebook (Assignment 1), this module has
been refactored to be import-safe and headless. The public entry point is
`run_eda(csv_path, user_target, output_dir)` which returns a cleaned
DataFrame and a summary dict consumed by the planner.
"""

# These are our imports.
import os
import re
import json
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# Helper class to avoid json errors.
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super(NpEncoder, self).default(obj)


def _clean_single_column_name(name):
    """
    Normalise a single column name using the same rules as ``clean_column_names``.

    Used by ``analyze_target`` so that a user-supplied ``--target`` flag (raw
    CSV header spelling) is matched against the cleaned column names that live
    on the DataFrame. Without this, passing ``--target Risk_Level`` would fail
    silently because the cleaned DataFrame column is ``risk_level``.
    """
    if name is None:
        return None
    col = str(name).strip().lower()
    col = re.sub(r'[^a-z0-9_]', '_', col)
    col = re.sub(r'_+', '_', col)
    col = col.strip('_')
    if col and col[0].isdigit():
        col = f"col_{col}"
    return col


def clean_column_names(df):
    """
    This function just for the column names.
    """
    new_columns = []
    seen_columns = {}
    # Rule 1: Clean the unnamed columns.
    # If column is empty and unnecessarly it will be dropped down.
    unnamed_cols = [c for c in df.columns if "Unnamed" in str(c)]
    if unnamed_cols:
        print(f"-> !!! Warning: {len(unnamed_cols)} unnamed column is dropping down...")
        df = df.drop(columns=unnamed_cols)

    # Convert column names to strings.
    df.columns = df.columns.astype(str)

    for col in df.columns:
        original_col = col

        # Rule 2: Delete the spaces and convert to lowercase.
        col = col.strip().lower()

        # Rule 3: Clear special characters.
        col = re.sub(r'[^a-z0-9_]', '_', col)

        # Reduce consecutive underscores to a single one.
        col = re.sub(r'_+', '_', col)

        # Remove the underscores.
        col = col.strip('_')

        # Rule 4: If it starts with a number, add 'col_' to the beginning.
        if col[0].isdigit():
            col = f"col_{col}"

        # Rule 5: Duplicate name column check.
        if col in seen_columns:
            seen_columns[col] += 1
            col = f"{col}_{seen_columns[col]}" # If there is any same name column, it will be colmn_1 and column_2
        else:
            seen_columns[col] = 1

        new_columns.append(col)

    if list(df.columns) != new_columns:
        print("!!!! Column names have been normalized (lowercase, no spaces) !!!!")

    df.columns = new_columns

    print("\n-> Data uploaded successfully!")
    print(f"-> Size: {df.shape[0]} Rows, {df.shape[1]} Columns")
    print(f"-> First 5 raw:")
    print(df.head().to_string())
    return df


def header_calculater(file_path):
    """
    This function looking for header in csv.
    If csv has not a header, It checks for numerical fatigue in the first row and, since a column name cannot be a number,
    if it's more than 50%, it concludes that the file may not have a header and performs its operations accordingly and
    warn the user.
    """
    try:
        # default reading, first line is considered header
        # low_memory=False disables chunked dtype inference so mixed int/float columns
        # are inferred consistently across the whole file (prevents a subtle bug where
        # early chunks infer int and later chunks promote to float silently).
        df = pd.read_csv(file_path, low_memory=False)
        if df.shape[1] == 1:
            try:
                df_semi = pd.read_csv(file_path, sep=';', low_memory=False)
                # If the column number increases when read with a semicolon, then that's the correct separator.
                if df_semi.shape[1] > 1:
                    df = df_semi
                    print("-> Semicolon (;) separator detected and applied.")
            except:
                pass

        if df.shape[1] == 0:
            print("!!!Error: The file is empty!!!")
            return None

        # Header controle
        numeric_header_count = 0
        total_columns = len(df.columns)

        for col in df.columns:
            try:
                float(str(col)) # It try to convert column names to number
                numeric_header_count += 1
            except ValueError:
                continue

        # If more than 50% of the headlines are numbers, be suspicious.
        ratio = numeric_header_count / total_columns

        if ratio > 0.5:
            print(f"\nATTENTION: The first row of the dataset does not look like a header!")
            print(f" -> Detection: {int(ratio*100)}% of the column names consist of numerical values.")
            print(f" -> Risk: Your first row will be treated as 'Data' to avoid data loss.")
            print(f" -> Action: Automatic column names (col_0, col_1...) have been assigned.")
            print(f" -> SUGGESTION: Please make sure your CSV file has a header row!\n")

            # Read the file again, it says there's no header.
            df = pd.read_csv(file_path, header=None, low_memory=False)

            # Give the columns temporary names, but don't lose the original data.
            df.columns = [f"col_{i}" for i in range(len(df.columns))]

    except Exception as e:
        print(f"Critical Error: File can't read. Reason: {e}")
        return None

    print(f"Data Uploaded. Size: {df.shape}")
    return df


def detect_mixed_dtypes(df):
    """
    Detect columns where the stored dtype hints at a CSV loading inconsistency.

    Two patterns are flagged:

    1. **Float columns that are mostly whole numbers.** pandas promotes an
       otherwise integer column to float64 as soon as a single non-integer
       value appears. If 80–99% of the column's values are integer-like but
       a minority are real floats, the column was probably meant to be int
       and got contaminated mid-file (e.g. first 19k rows int, last 1k float).

    2. **Object columns containing mixed numeric strings** (some values parse
       as int, some as float, no non-numeric values). These should probably
       be a single numeric dtype.

    Returns a list of dicts, one per flagged column.
    """
    mixed_cols = []

    # Pattern 1: float columns that are mostly integer-like
    for col in df.select_dtypes(include=['float']).columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        is_int_like = (series % 1 == 0)
        int_ratio = float(is_int_like.mean())
        # Flag the "mostly int, a few floats" pattern
        if 0.80 < int_ratio < 0.99:
            non_int_mask = ~is_int_like
            first_float_idx = int(series[non_int_mask].index[0]) if non_int_mask.any() else None
            mixed_cols.append({
                "column": col,
                "dtype": "float (mostly int-like)",
                "int_like_ratio": round(int_ratio, 3),
                "n_int_like": int(is_int_like.sum()),
                "n_true_float": int(non_int_mask.sum()),
                "first_float_row": first_float_idx,
                "note": (
                    f"Column '{col}' is float64 but {int_ratio*100:.1f}% of values are whole numbers. "
                    f"First non-integer value at row {first_float_idx}. "
                    f"Possible CSV loading bug — inspect source data for mixed int/float."
                )
            })

    # Pattern 2: object columns that contain mixed numeric strings
    for col in df.select_dtypes(include=['object']).columns:
        series = df[col].dropna().astype(str)
        if len(series) == 0:
            continue
        n_int = 0
        n_float = 0
        n_non_numeric = 0
        first_float_idx = None
        for idx, val in series.items():
            val_stripped = val.strip()
            if val_stripped == "":
                continue
            # Pure integer string (allow leading sign)
            if val_stripped.lstrip('-+').isdigit():
                n_int += 1
                continue
            # Otherwise try float
            try:
                float(val_stripped)
                n_float += 1
                if first_float_idx is None:
                    first_float_idx = int(idx)
            except ValueError:
                n_non_numeric += 1

        # Flag only if values are purely numeric but mixed int/float representations
        if n_int > 0 and n_float > 0 and n_non_numeric == 0:
            mixed_cols.append({
                "column": col,
                "dtype": "object (mixed numeric)",
                "n_int": n_int,
                "n_float": n_float,
                "first_float_row": first_float_idx,
                "note": (
                    f"Column '{col}' is stored as object but contains {n_int} integer-like and "
                    f"{n_float} float-like strings (first float at row {first_float_idx}). "
                    f"Consider converting to a single numeric dtype."
                )
            })

    return mixed_cols


def apply_mixed_dtype_cast_fix(df):
    """
    Repair columns flagged by ``detect_mixed_dtypes``.

    Pattern 1 (float columns that are mostly integer-like, e.g. WineQuality's
    ``quality`` target where the first ~16k rows are whole numbers and the last
    ~1k are fractional) is repaired by rounding every value to the nearest
    integer and casting the column to nullable ``Int64``. This recovers the
    original intent of the data and lets downstream ``analyze_target`` see an
    integer classification target instead of a continuous float.

    Pattern 2 (object columns with mixed int/float strings) is left alone —
    downstream ``to_numeric`` coercions handle it without data loss.

    Returns ``(df_out, applied)`` where ``applied`` is the list of dicts from
    ``detect_mixed_dtypes`` that were actually cast. The input DataFrame is
    not mutated.
    """
    detected = detect_mixed_dtypes(df)
    if not detected:
        return df, []

    df_out = df.copy()
    applied = []
    for entry in detected:
        col = entry.get("column")
        if col not in df_out.columns:
            continue
        # Only Pattern 1 (float → int) is auto-repaired.
        if str(entry.get("dtype", "")).startswith("float"):
            df_out[col] = df_out[col].round().astype("Int64")
            applied.append(entry)
            print(
                f"[cast_fix] '{col}' rounded and cast to Int64 "
                f"(int_like_ratio={entry.get('int_like_ratio')})."
            )

    return df_out, applied


# MODULE 2: Data quality and feature analysis.
def inspect_data_quality(df):
    """
    This function looks at data quality and column attributes.
    It does not perform target analysis.
    It cleans duplicates and reports missing and outlier/skewness.
    """
    report = {} # This is our report for json
    notes = []

    print("=== DATA QUALITY CONTROL ===")

    # Rule 1: Duplicate controle
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        report['duplicates_handled'] = True
        notes.append(f"{duplicates} duplicate lines were deleted.")
    else:
        report['duplicates_handled'] = False


    # Rule 2: Missing value controle.
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]

    report['has_missing_values'] = len(missing_cols) > 0
    # We give only missing value columns. Planner will decide to process.
    report['cols_with_missing'] = missing_cols.index.tolist()

    # Missing ratio for each missing column
    # Planner will determine the Drop value if it's more than 50%, and the Impute value if it's less than 50% based on this data.
    missing_ratios = (df.isnull().mean()).to_dict()
    report['missing_ratios'] = {k: round(v, 4) for k, v in missing_ratios.items() if v > 0}

    if len(missing_cols) > 0:
        notes.append(f"Missing data: The {len(missing_cols)} column contains empty values.")

    # Rule 3: Numerical analysis (skewness, outliers)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    high_skew_cols = []
    high_outlier_cols = []

    # Detailed data for plotting the graph.
    stats_for_plotting = {}

    for col in numeric_cols:
        # Skewness calculation.
        skew_val = df[col].skew()
        if abs(skew_val) > 1.0:
            high_skew_cols.append(col)

        # Outliers (IQR) calculation.
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()

        if outlier_count > 0:
            # Add to list if more than 3% of rows are outliers.
            if outlier_count / len(df) > 0.03:
                high_outlier_cols.append(col)

        # Keep the data for plot.
        stats_for_plotting[col] = {"skew": skew_val, "outliers": outlier_count}

    # Rule 3b: Mixed dtype detection (catches CSV loading bugs where a column
    # is mostly int but gets promoted to float by a few trailing float rows).
    mixed_dtype_cols = detect_mixed_dtypes(df)

    report['numeric_analysis'] = {
        "skewed_cols": high_skew_cols,
        "outlier_cols": high_outlier_cols,
        "mixed_dtype_cols": mixed_dtype_cols
    }

    if high_skew_cols: notes.append(f"Skewness was detected: {len(high_skew_cols)} column.")
    if high_outlier_cols: notes.append(f"High number of outliers detected: {len(high_outlier_cols)} column.")
    if mixed_dtype_cols:
        for entry in mixed_dtype_cols:
            notes.append(entry["note"])

    # Rule 4: Categorical analysis (Cardinality)
    cat_cols_all = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Find Date Candidates (Containing 'date', 'time', 'year')
    date_candidates = [c for c in cat_cols_all if any(x in c.lower() for x in ['date', 'time', 'year', 'month'])]

    # Pure Categorical (Non-historical)
    cat_cols = [c for c in cat_cols_all if c not in date_candidates]

    high_card_cols = []
    useless_cols = []
    potential_ids = []

    for col in cat_cols:
        nunique = df[col].nunique()
        ratio = nunique / len(df)

        if nunique == 1:
            useless_cols.append(col)

        if ratio > 0.9:
            potential_ids.append(col)

        if nunique > 50:
            high_card_cols.append({
                "column": col,
                "nunique": nunique,
                "ratio": round(ratio, 3)
            })

    report['categorical_analysis'] = {
        "high_cardinality_cols": high_card_cols, # Planner will decide to make this a TargetEncoder.
        "potential_ids": potential_ids,          # Planner won't take these.
        "useless_cols": useless_cols             # Planner won't take these.
    }

    report['feature_types'] = {
        "numeric": numeric_cols,
        "categorical": cat_cols,
        "date": date_candidates if 'date_candidates' in locals() else []
    }

    if high_card_cols: notes.append(f"There is high cardinality on {[d['column'] for d in high_card_cols]} columns. Consider alternatives instead of OneHotEncoding.")
    if potential_ids: notes.append(f"Columns that can be ID: {potential_ids}")
    if useless_cols: notes.append(f"Useless columns: {useless_cols}, planner should not take this columns. Because the unique ratio is 1, meaning each row contains a different value, it could be a potential ID. ")
    if numeric_cols: notes.append(f"Numeric columns size: {len(numeric_cols)}")
    if cat_cols: notes.append(f"Categorical columns size: {len(cat_cols)}")
    if date_candidates: notes.append(f"Date columns: {date_candidates}, planner decide to break it down and insert this column into the model.")

    # Rule 5: Finish the report for data quality and feature analysis part.
    report['notes'] = notes

    # ATTENTION: We are not including plotting_data in the main report.
    # We are returning it as a separate variable.
    plotting_data = {
        "skew_cols": high_skew_cols,
        "outlier_cols": high_outlier_cols
    }

    print("===!!! Module 2 Analysis Completed !!!===\n")

    # The function now returns 3 things:
    # 1. Cleaned Data (df)
    # 2. Clean Report for Planner (report)
    # 3. Data for your Graphs (plotting_data)
    return df, report, plotting_data


def plot_data_quality_visual(df, final_summary, plotting_data, output_dir):
    """
    Save EDA plots (skewness histograms + outlier boxplots) to `output_dir`.

    The notebook version of this function used plt.show(); in the headless
    pipeline we always write PNGs so runs can be inspected later from the
    timestamped outputs directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get lists from the dictionary
    high_skew_cols = plotting_data.get("skew_cols", [])
    high_outlier_cols = plotting_data.get("outlier_cols", [])
    sns.set_style("whitegrid")

    # This function builds one figure containing all passed columns as subplots,
    # then saves it to disk.
    def plot_graph(col_list, plot_type, color, filename):
        n = len(col_list)
        if n == 0:
            return

        cols_per_row = 3
        rows = math.ceil(n / cols_per_row)

        fig, axes = plt.subplots(rows, cols_per_row, figsize=(18, 4 * rows))

        # Normalize axes to a flat ndarray regardless of shape
        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        else:
            axes_flat = np.array([axes]).flatten()

        last_i = -1
        for i, col in enumerate(col_list):
            ax = axes_flat[i]
            last_i = i

            if plot_type == "hist":
                skew_val = df[col].skew()
                sns.histplot(data=df, x=col, kde=True, ax=ax, color=color, edgecolor='black', alpha=0.6)
                ax.set_title(f"{col}\n(Skew: {skew_val:.2f})", fontweight='bold', fontsize=10)

            elif plot_type == "box":
                sns.boxplot(x=df[col], ax=ax, color=color, width=0.5)
                ax.set_title(f"{col}\n(Outliers Detected)", fontweight='bold', fontsize=10)

            ax.set_xlabel("")

        # Clean up empty subplots
        for j in range(last_i + 1, len(axes_flat)):
            axes_flat[j].axis('off')

        plt.tight_layout()
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"-> Saved plot: {save_path}")

    numeric_cols_all = final_summary['data_quality_and_features']['feature_types'].get("numeric", [])

    if not high_skew_cols and not high_outlier_cols and numeric_cols_all:
        print("\n===== GOOD DATA QUALITY =====")
        print("No significant skewness or outliers detected.")
        print("Saving the distribution of the first 5 numerical columns as a sample.\n")

        # Choose the first 5 or all if there are fewer than 5
        sample_cols = numeric_cols_all[:5]

        if sample_cols:
            plot_graph(sample_cols, "hist", "mediumseagreen", "eda_sample_distributions.png")
        return
    elif not high_skew_cols and not high_outlier_cols:
        print("There is no numeric values in this dataset.")

    # Start drawings
    if high_skew_cols:
        print(f"\n ===== SKEWNESS DISTRIBUTION ANALYSIS {len(high_skew_cols)} Columns =====")
        plot_graph(high_skew_cols, "hist", "darkgreen", "eda_skewness.png")

    if high_outlier_cols:
        print(f"\n===== OUTLIER DETECTION ANALYSIS {len(high_outlier_cols)} Columns =====")
        plot_graph(high_outlier_cols, "box", "steelblue", "eda_outliers.png")


def analyze_target(df, user_target=None):
    """
    MODULE 3: Target Analysis
    1. Always start your search from the LAST COLUMN.
    2. Don't rely on keywords that might be features, such as 'Total' and 'Amount'.
    3. Focus on strong keywords like 'Quality', 'Revenue', and 'Class'.
    """
    report = {}
    notes = []

    print("TARGET ANALYSIS\n")

    target_col = None

    # LIST 1: High Priority objectives.
    priority_keywords = [
        'class', 'target', 'fraud', 'churn', 'label', 'outcome',
        'is_fraud', 'default', 'survived', 'diagnosis', 'result',
        'y'
    ]

    # LIST 2: Medium Priority(Strong Candidates)
    medium_priority_keywords = [
        'revenue', 'sales', 'price', 'salary', 'score',
        'rating', 'quality', 'vote', 'popularity', 'profit',
        'status'
    ]

    # A. If specified by the user
    # Apply the same normalisation as clean_column_names so that a raw header
    # spelling (e.g. "Risk_Level") still resolves against the cleaned column
    # list (which stores it as "risk_level"). If the user's target cannot be
    # resolved either raw or normalised, we warn and fall through to the
    # auto-detection branch rather than silently picking the last column.
    if user_target:
        normalized_target = _clean_single_column_name(user_target)
        if user_target in df.columns:
            target_col = user_target
            print(f"Target: '{target_col}'")
        elif normalized_target in df.columns:
            target_col = normalized_target
            print(f"Target: '{user_target}' -> normalised to '{target_col}'")
        else:
            notes.append(
                f"user_target '{user_target}' (normalised '{normalized_target}') "
                f"not found in columns; falling back to auto-detection."
            )
            print(
                f"!!! user_target '{user_target}' not in columns — "
                f"falling back to auto-detection."
            )

    # B. Automatic Detection (runs when no user_target or user_target missed)
    if target_col is None:
        # Arrange columns from RIGHT TO LEFT.
        # Because in datasets, the target is almost always at the end (90%).
        cols_reversed = list(reversed(df.columns))

        # Experiment 1: Exact name match(High Priority)
        for col in cols_reversed:
            col_lower = col.lower()
            if col_lower in priority_keywords:
                target_col = col
                print(f"Target Found with priority-list : '{target_col}'")
                break

        # Experiment 2: Medium Priority(Strong Candidates)
        if target_col is None:
            for col in cols_reversed:
                col_lower = col.lower()

                # Is the column an ID or a Date?
                if 'id' in col_lower and df[col].nunique() == len(df): continue
                if 'date' in col_lower or 'time' in col_lower: continue

                # Is there a word match? (quality, revenue, price...)
                if any(keyword in col_lower for keyword in medium_priority_keywords):
                    target_col = col
                    print(f"Target Found with Strong Candidate: '{target_col}'")
                    break

        # Experiment 3: If no matches get last valid column.
        if target_col is None:
            print("No name clue found, searching the rightmost valid column...")
            for col in cols_reversed:
                # Rule 1: ID Check. If the number of uniques is equal to the number of rows.
                if 'id' in col.lower() and df[col].nunique() == len(df):
                    continue

                # Rule 2: Constant Value Control. If they are all the same, they cannot be the target.
                if df[col].nunique() <= 1:
                    continue

                # Rule 3: Unique Text Check (Description, URL, etc.)
                if df[col].dtype == 'object' and df[col].nunique() > len(df) * 0.8:
                    continue

                # Rule 4: Excessively Unbalanced Text Check
                # Only checked on object/categorical data.
                try:
                    if df[col].dtype == 'object':
                        dominance = df[col].value_counts(normalize=True).iloc[0]
                        if dominance > 0.95: # If 95% are the same, it's not a target.
                            continue
                except:
                    pass

                # The first column coming here is the target.
                target_col = col
                print(f"Target Found: '{target_col}'")
                break

    # This is for safety.
    if target_col is None:
        target_col = df.columns[-1] # If we are desperate, we choose the last one.
        report['warning'] = "Target was chosen under duress."

    report['target_col'] = target_col

    # Problem type and analysis
    y = df[target_col]
    unique_count = y.nunique()
    dtype = y.dtype
    problem_type = "unknown"

    is_explicit_class = target_col.lower() in priority_keywords

    if (
        pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    ):
        problem_type = "classification"
    elif pd.api.types.is_numeric_dtype(dtype):
        if is_explicit_class:
            problem_type = "classification"
        # If it's a float and the number of unique elements is very small, then classification might be possible.
        # But generally, float = regression.
        elif pd.api.types.is_float_dtype(dtype) and unique_count < 10:
             problem_type = "classification"
        # We set the threshold for integers to 20 (to avoid the AGE problem).
        elif pd.api.types.is_integer_dtype(dtype) and unique_count < 20:
            problem_type = "classification"
        else:
            problem_type = "regression"

    report['problem_type'] = problem_type
    report['num_classes'] = int(unique_count) if problem_type == "classification" else None

    print(f"Problem Type: {problem_type.upper()} Unique Score:{unique_count}\n")

    # Imbalance analysis.
    if problem_type == "classification":
        vc = y.value_counts()
        if len(vc) > 1:
            min_c = vc.min(); max_c = vc.max()
            imb_ratio = round(max_c / min_c, 2)
            report['imbalance_ratio'] = imb_ratio

            if min_c < 10:
                report['imbalance_status'] = "severe"
                notes.append(f"Critical Imbalance: Minimum class {min_c} units.")
            elif imb_ratio > 3.0:
                report['imbalance_status'] = "imbalanced"
                notes.append(f"There is an imbalance Ratio: {imb_ratio}.")
            else:
                report['imbalance_status'] = "balanced"
        else:
            report['imbalance_ratio'] = 0
            report['imbalance_status'] = "single_class"
            notes.append("!!!! ERROR: There is only 1 class in the target column !!!!")
    else:
        report['imbalance_ratio'] = None
        report['imbalance_status'] = None


    # Last cleaning
    missing_target = y.isnull().sum()
    if missing_target > 0:
        notes.append(f"There is a missing value {missing_target} in the target column.")
        report['rows_to_drop_indices'] = y[y.isnull()].index.tolist()
    else:
        report['rows_to_drop_indices'] = []

    report['notes'] = notes

    print("===!!! Module 3 Analysis Completed !!!===")
    return report


def generate_eda_summary(df, user_target=None):
    """
    MODULE 4: Creating the final JSON report.

    Calls Module 2 (Data quality and feature analysis) then Module 3 (Target
    Analysis) and merges both reports. In addition to the nested structure
    consumed by the human-readable report, this function attaches a set of
    **flat** top-level keys (shape, target, feature_types, imbalance_ratio,
    ...) that the downstream pipeline (modelling.py, planner.py) expects.
    Keeping both views in the same dict lets the rich EDA drive planning
    without breaking the existing pipeline consumers.
    """
    print("->  EDA AGENT IS BEING LAUNCHED...\n")

    # Part 1: Module 2 (Data quality and feature analysis)
    # Since Module 2 deletes duplicate rows, we need to get the current df.
    df_clean, quality_report, plotting_data = inspect_data_quality(df)
    print("=" * 30)

    # Part 2: Module 3 (Target Analysis)
    # We're using a cleaned-up df file so that the row counts match.
    target_report = analyze_target(df_clean, user_target)
    print("=" * 30)

    target_col = target_report['target_col']
    problem_type = target_report.get('problem_type')

    # Flat feature_types (excludes the target, matches data_profiler.profile_dataset shape)
    numeric_all = quality_report['feature_types'].get('numeric', [])
    cat_all = quality_report['feature_types'].get('categorical', [])
    numeric_features = [c for c in numeric_all if c != target_col]
    categorical_features = [c for c in cat_all if c != target_col]

    # class_counts for the target (classification only)
    class_counts = None
    if problem_type == 'classification':
        vc = df_clean[target_col].value_counts(dropna=False)
        class_counts = {str(k): int(v) for k, v in vc.items()}

    # missing_pct as percentage (the existing missing_ratios stores fractions)
    missing_pct = {
        col: round(float(ratio) * 100, 2)
        for col, ratio in quality_report.get('missing_ratios', {}).items()
    }

    combined_notes = quality_report['notes'] + target_report['notes']

    final_summary = {
        "dataset_info": {
            "rows": df_clean.shape[0],
            "cols": df_clean.shape[1]
        },

        # Everything related to the target is here.
        "target_analysis": target_report,

        # Everything about features and quality is here.
        "data_quality_and_features": {
            "duplicates_handled": quality_report['duplicates_handled'],
            "missing_values": {
                "has_missing": quality_report['has_missing_values'],
                "cols": quality_report['cols_with_missing'],
                "ratios": quality_report.get('missing_ratios', {})
            },
            "feature_types": quality_report['feature_types'],
            "numeric_stats": quality_report['numeric_analysis'],
            "categorical_stats": quality_report['categorical_analysis']
        },

        # We're combining all the warning notes.
        "combined_notes": combined_notes,

        # --- Flat keys for pipeline consumers (modelling.py, planner.py) ---
        "shape": {"rows": int(df_clean.shape[0]), "cols": int(df_clean.shape[1])},
        "columns": df_clean.columns.astype(str).tolist(),
        "target": str(target_col),
        "target_dtype": str(df_clean[target_col].dtype),
        "is_classification": problem_type == 'classification',
        # Top-level feature_types excludes the target and drops the 'date' bucket;
        # this is the shape modelling.py / planner.py read. The richer nested
        # view (with date + target included) stays under data_quality_and_features.
        "feature_types": {"numeric": numeric_features, "categorical": categorical_features},
        "n_unique_by_col": {str(c): int(df_clean[c].nunique(dropna=True)) for c in df_clean.columns},
        "missing_pct": missing_pct,
        "imbalance_ratio": target_report.get('imbalance_ratio'),
        "class_counts": class_counts,
        "notes": combined_notes,
    }

    print("\n-> The EDA process is complete. The report is ready.\n")
    return df_clean, final_summary, plotting_data


def dataset_fingerprint(df, target):
    """
    Stable fingerprint for a (dataset, target) pair used as a key into the
    JSON memory store. Same shape/columns/target → same fingerprint, so the
    agent can recognise datasets it has seen before.
    """
    cols = ",".join(df.columns.astype(str).tolist())
    shape = f"{df.shape[0]}x{df.shape[1]}"
    base = f"{shape}|{target}|{cols}"
    h = abs(hash(base)) % (10**12)
    return f"fp_{h}"


def run_eda(csv_path, user_target=None, output_dir=None):
    """
    Pipeline entry point. Chains:
        header_calculater -> clean_column_names -> generate_eda_summary
        -> (optional) plot_data_quality_visual

    Parameters
    ----------
    csv_path : str
        Path to the CSV dataset.
    user_target : str or None
        If provided and present in the columns, used as the target column;
        otherwise target is auto-inferred.
    output_dir : str or None
        If provided, EDA plots are written there as PNG files.

    Returns
    -------
    (df_clean, eda_summary) : (pd.DataFrame, dict)
    """
    df = header_calculater(csv_path)
    if df is None:
        raise RuntimeError(f"Could not load CSV at: {csv_path}")

    df = clean_column_names(df)

    df, cast_fixes_applied = apply_mixed_dtype_cast_fix(df)

    df_clean, eda_summary, plotting_data = generate_eda_summary(df, user_target=user_target)

    if cast_fixes_applied:
        eda_summary.setdefault("data_quality_and_features", {})[
            "mixed_dtype_cols_fixed"
        ] = cast_fixes_applied
        fixed_names = [f["column"] for f in cast_fixes_applied]
        eda_summary.setdefault("notes", []).append(
            f"Cast-fix applied to {len(fixed_names)} col(s): {fixed_names}."
        )

    if output_dir is not None:
        plot_data_quality_visual(df_clean, eda_summary, plotting_data, output_dir)

    return df_clean, eda_summary
