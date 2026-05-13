from typing import Any, Dict, List, Optional, Tuple

import logging
import os
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def build_preprocessor(
    profile: Dict[str, Any],
    drop_columns: Optional[List[str]] = None,
) -> ColumnTransformer:
    num_cols = list(profile["feature_types"]["numeric"])
    cat_cols = list(profile["feature_types"]["categorical"])

    if drop_columns:
        drop_set = set(drop_columns)
        num_cols = [c for c in num_cols if c not in drop_set]
        cat_cols = [c for c in cat_cols if c not in drop_set]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True)),
    ])

    # scikit-learn renamed `sparse` -> `sparse_output` (v1.2+). Support both.
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )


def select_models(
    profile: Dict[str, Any],
    seed: int = 42,
    priority_models: Optional[List[str]] = None,
) -> List[Tuple[str, Any]]:
    rows = profile["shape"]["rows"]
    cols = profile["shape"]["cols"]
    imb = float(profile.get("imbalance_ratio") or 1.0)
    class_weight = "balanced" if imb >= 3.0 else None

    candidates: List[Tuple[str, Any]] = [
        ("DummyMostFrequent", DummyClassifier(strategy="most_frequent")),
        ("LogisticRegression", LogisticRegression(max_iter=2000, class_weight=class_weight)),
        ("RandomForest", RandomForestClassifier(
            n_estimators=300, random_state=seed, n_jobs=-1, class_weight=class_weight
        )),
    ]

    if rows <= 50000:
        candidates.append(("GradientBoosting", GradientBoostingClassifier(random_state=seed)))

    # SVC can be expensive after one-hot; keep for smaller problems
    if rows <= 20000 and cols <= 200:
        candidates.append(("SVC_RBF", SVC(kernel="rbf", probability=True, class_weight=class_weight)))

    # Reorder: priority models first (if present in candidates), others after.
    # Names not in candidates are silently skipped — keeps this safe when a
    # memory hint references a model that the size guards have dropped.
    if priority_models:
        name_to_cand = {n: c for n, c in candidates}
        ordered_names: List[str] = []
        for name in priority_models:
            if name in name_to_cand and name not in ordered_names:
                ordered_names.append(name)
        for name, _ in candidates:
            if name not in ordered_names:
                ordered_names.append(name)
        candidates = [(n, name_to_cand[n]) for n in ordered_names]

    return candidates


def _safe_cv_scores(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int,
    seed: int,
) -> List[float]:
    """Run StratifiedKFold cross_val_score safely on small/imbalanced data.

    Folds are clamped to [2, min_class_count] so we never request more folds
    than the rarest class can support. Returns [] if CV is infeasible (single
    class or rarest class has <2 samples) — callers must treat empty list as
    "stat tests not possible," not as a zero score.
    """
    counts = y_train.value_counts(dropna=True)
    if len(counts) < 2 or int(counts.min()) < 2:
        return []
    n_splits = max(2, min(int(cv_folds), int(counts.min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    try:
        scores = cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=1
        )
        return [float(s) for s in scores]
    except Exception:
        return []


def train_models(
    df: pd.DataFrame,
    target: str,
    preprocessor: ColumnTransformer,
    candidates: List[Tuple[str, Any]],
    seed: int,
    test_size: float,
    output_dir: str,
    verbose: bool = True,
    cv_folds: int = 5,
) -> Dict[str, Any]:
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found.")

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # Drop missing target rows
    mask = ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask]

    # Stratify if possible
    stratify = y if (y.nunique(dropna=True) > 1 and y.value_counts().min() >= 2) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    results: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    for name, model in candidates:
        logger.info("Training candidate: %s", name)

        try:
            pipe = Pipeline(steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ])
            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            y_train_pred = pipe.predict(X_train)

            metrics = {
                "model": name,
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
                "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
                "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
            }

            train_metrics = {
                "accuracy": float(accuracy_score(y_train, y_train_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_train, y_train_pred)),
                "f1_macro": float(f1_score(y_train, y_train_pred, average="macro", zero_division=0)),
            }

            cv_scores = _safe_cv_scores(pipe, X_train, y_train, cv_folds, seed)

            results.append({
                "name": name,
                "pipeline": pipe,
                "metrics": metrics,
                "train_metrics": train_metrics,
                "cv_scores": cv_scores,
                "X_test": X_test,
                "y_test": y_test,
                "y_pred": y_pred,
            })
        except Exception as exc:
            # Per-model isolation: a single bad candidate must not kill the
            # whole run. Record the failure, log the traceback at DEBUG, and
            # continue with the remaining candidates. If every candidate
            # fails, the fallback below raises explicitly instead of
            # silently producing an empty ranking.
            logger.warning("Candidate %s failed: %s", name, exc)
            logger.debug("Traceback for %s:", name, exc_info=True)
            failed.append({
                "model": name,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })

    if not results:
        raise RuntimeError(
            f"All {len(candidates)} candidate models failed during training. "
            f"Failures: {failed}"
        )

    # Sort by balanced accuracy then macro F1
    results.sort(key=lambda r: (r["metrics"]["balanced_accuracy"], r["metrics"]["f1_macro"]), reverse=True)

    # all_metrics carries train_metrics + cv_scores so the reflector can run
    # paired stat tests and overfit checks without re-touching the pipelines.
    all_metrics = [
        {
            **r["metrics"],
            "train_metrics": r["train_metrics"],
            "cv_scores": r["cv_scores"],
        }
        for r in results
    ]

    # Expose failed candidates too so the orchestrator/report can surface them.
    all_metrics.extend(failed)

    return {
        "results": results,
        "best": results[0],
        "all_metrics": all_metrics,
        "failed": failed,
    }
