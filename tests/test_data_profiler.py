"""Unit tests for tools.data_profiler: column cleaning, dtype detection, target analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.data_profiler import (
    _clean_single_column_name,
    apply_mixed_dtype_cast_fix,
    analyze_target,
    clean_column_names,
    dataset_fingerprint,
    detect_mixed_dtypes,
)


# --------------------------------------------------------------------------- #
# _clean_single_column_name                                                    #
# --------------------------------------------------------------------------- #


def test_clean_single_column_name_lowercases_and_strips_special_chars():
    assert _clean_single_column_name("Risk_Level") == "risk_level"
    assert _clean_single_column_name("  Some Col!  ") == "some_col"
    assert _clean_single_column_name("A-B.C") == "a_b_c"


def test_clean_single_column_name_prefixes_numeric_start():
    # Columns starting with a digit must be prefixed so they are valid Python-ish
    # identifiers downstream.
    assert _clean_single_column_name("123col") == "col_123col"


def test_clean_single_column_name_handles_none_and_empty():
    assert _clean_single_column_name(None) is None
    # Entirely non-alphanumeric input collapses to an empty string (edge case —
    # upstream clean_column_names catches duplicates afterwards).
    assert _clean_single_column_name("!!!") == ""


# --------------------------------------------------------------------------- #
# clean_column_names                                                           #
# --------------------------------------------------------------------------- #


def test_clean_column_names_dedupes_and_drops_unnamed(capsys):
    df = pd.DataFrame(
        {
            "Name": [1, 2],
            "name": [3, 4],           # collision after lowercasing
            "Unnamed: 0": [5, 6],     # dropped by Rule 1
            "Weird Col!": [7, 8],
        }
    )
    out = clean_column_names(df)
    # Unnamed dropped, collisions disambiguated, special chars normalised.
    assert "unnamed_0" not in out.columns
    assert "name" in out.columns
    assert "name_2" in out.columns
    assert "weird_col" in out.columns


# --------------------------------------------------------------------------- #
# detect_mixed_dtypes (Pattern 1: float-mostly-int)                            #
# --------------------------------------------------------------------------- #


def test_detect_mixed_dtypes_flags_wine_quality_pattern():
    # 95 int-like values + 5 fractional values → 95% int-ratio → flagged.
    values = [float(i % 6 + 3) for i in range(95)] + [3.5, 4.5, 5.5, 6.5, 7.5]
    df = pd.DataFrame({"quality": values})
    flagged = detect_mixed_dtypes(df)
    assert any(c["column"] == "quality" for c in flagged)
    entry = next(c for c in flagged if c["column"] == "quality")
    assert 0.90 <= entry["int_like_ratio"] <= 0.96


def test_detect_mixed_dtypes_ignores_pure_float_and_pure_int():
    df = pd.DataFrame(
        {
            "pure_float": np.random.RandomState(0).uniform(0, 1, size=100),
            "pure_int": np.arange(100, dtype="int64"),
        }
    )
    flagged = detect_mixed_dtypes(df)
    assert flagged == []


# --------------------------------------------------------------------------- #
# apply_mixed_dtype_cast_fix                                                   #
# --------------------------------------------------------------------------- #


def test_apply_mixed_dtype_cast_fix_rounds_and_casts_to_int64():
    values = [float(i % 5 + 1) for i in range(95)] + [1.4, 2.6, 3.1, 4.8, 5.2]
    df = pd.DataFrame({"quality": values})
    out, applied = apply_mixed_dtype_cast_fix(df)
    assert len(applied) == 1
    assert applied[0]["column"] == "quality"
    # Nullable Int64 after repair, no fractional leftovers.
    assert str(out["quality"].dtype) == "Int64"
    # Integer values recovered via rounding.
    assert out["quality"].iloc[-5:].tolist() == [1, 3, 3, 5, 5]


# --------------------------------------------------------------------------- #
# analyze_target                                                               #
# --------------------------------------------------------------------------- #


def test_analyze_target_respects_explicit_user_target_after_normalisation():
    # Simulates the cancer_risk_factors bug: user passes "Risk_Level" but
    # clean_column_names has already lowercased it to "risk_level".
    df = pd.DataFrame(
        {
            "age": [30, 40, 50, 60],
            "risk_level": ["low", "high", "low", "high"],
        }
    )
    report = analyze_target(df, user_target="Risk_Level")
    assert report["target_col"] == "risk_level"


def test_analyze_target_auto_picks_priority_keyword_when_no_user_target():
    # `label` is in the high-priority keyword list; the auto path should pick
    # it even though it isn't the rightmost column.
    df = pd.DataFrame(
        {
            "label": [0, 1, 0, 1],
            "feature_a": [1.0, 2.0, 3.0, 4.0],
        }
    )
    report = analyze_target(df, user_target=None)
    assert report["target_col"] == "label"


# --------------------------------------------------------------------------- #
# dataset_fingerprint                                                          #
# --------------------------------------------------------------------------- #


def test_dataset_fingerprint_stable_for_identical_input():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "label": [0, 1]})
    fp1 = dataset_fingerprint(df, "label")
    fp2 = dataset_fingerprint(df.copy(), "label")
    assert fp1 == fp2
    assert fp1.startswith("fp_")


def test_dataset_fingerprint_changes_when_columns_change():
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4], "label": [0, 1]})
    df2 = pd.DataFrame({"a": [1, 2], "c": [3, 4], "label": [0, 1]})
    assert dataset_fingerprint(df1, "label") != dataset_fingerprint(df2, "label")
