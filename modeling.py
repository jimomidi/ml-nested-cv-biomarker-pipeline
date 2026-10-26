"""
Strict nested leave-one-out cross-validation for binary
classification of expression data, and a model-selection-aware
permutation test for assessing statistical significance.

Design notes
------------
- The outer loop is leave-one-out (LOO-CV). For every outer fold,
  the entire pipeline (expression prefilter -> scaling -> feature
  selection -> classifier) is refit with an inner grid search
  performed only on the training fold. The held-out sample never
  influences preprocessing, feature selection, or hyperparameter
  choice.
- The permutation test repeats this ENTIRE procedure, including
  model selection, on label-shuffled data. Using the best AUC across
  candidate models for each permutation (rather than a single
  pre-specified model) makes the resulting p-value account for the
  fact that the reported model was chosen by comparing several
  candidates.
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from config import (
    K_GRID,
    MIN_DETECTED_FRACTION,
    MIN_MEAN_EXPRESSION,
    N_JOBS_INNER,
    N_PERMUTATIONS,
    RANDOM_SEED,
)


class ExpressionFilterLog2(BaseEstimator, TransformerMixin):
    """
    Fold-safe expression prefilter and log2(x + 1) transform.

    Fitted only on the training fold to avoid leakage from held-out
    samples. Features that fail the detection/expression thresholds
    are zeroed out rather than dropped, so the transformed matrix
    keeps a fixed number of columns across folds (required for the
    downstream feature-selection mask to align with the raw
    columns).
    """

    def __init__(self, min_detected_fraction: float = 0.30, min_mean_expression: float = 1.0):
        self.min_detected_fraction = min_detected_fraction
        self.min_mean_expression = min_mean_expression

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        detected_fraction = (X > 0).mean(axis=0)
        mean_expression = X.mean(axis=0)

        self.keep_mask_ = (detected_fraction >= self.min_detected_fraction) & (
            mean_expression >= self.min_mean_expression
        )

        if self.keep_mask_.sum() == 0:
            self.keep_mask_[:] = True

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X_log = np.log2(X + 1.0)
        X_log[:, ~self.keep_mask_] = 0.0
        return X_log


def safe_inner_cv(y_train: np.ndarray, random_seed: int = RANDOM_SEED) -> Optional[StratifiedKFold]:
    """
    Return a StratifiedKFold splitter sized to the smallest class in
    `y_train`, or None if inner cross-validation is not feasible
    (fewer than 2 samples in the minority class of this training
    fold).
    """
    counts = pd.Series(y_train).value_counts()
    min_class_count = counts.min()

    if min_class_count < 2:
        return None

    n_splits = min(3, min_class_count)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)


def get_models() -> dict:
    """
    Candidate classifiers and their hyperparameter grids for the
    inner grid search. XGBoost is included only if installed.
    """
    models = {
        "LogisticRegression": (
            LogisticRegression(max_iter=5000, class_weight="balanced"),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "SVM": (
            SVC(probability=True, class_weight="balanced", random_state=RANDOM_SEED),
            {"clf__C": [0.1, 1, 10], "clf__kernel": ["linear", "rbf"]},
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=RANDOM_SEED, class_weight="balanced"),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 3, 5]},
        ),
    }

    try:
        import xgboost as xgb

        models["XGBoost"] = (
            xgb.XGBClassifier(eval_metric="logloss", random_state=RANDOM_SEED),
            {"clf__n_estimators": [100, 200], "clf__max_depth": [2, 3, 4]},
        )
    except ImportError:
        print("XGBoost not installed -> skipped.")

    return models


def run_nested_cv(
    X_raw: pd.DataFrame,
    y: np.ndarray,
    estimator: BaseEstimator,
    param_grid: dict,
    label_encoder: LabelEncoder,
    positive_label: str,
    k_grid: Sequence[int] = K_GRID,
    min_detected_fraction: float = MIN_DETECTED_FRACTION,
    min_mean_expression: float = MIN_MEAN_EXPRESSION,
    n_jobs_inner: int = N_JOBS_INNER,
    random_seed: int = RANDOM_SEED,
    collect_audit: bool = False,
) -> dict:
    """
    Strict leave-one-out nested cross-validation for a single model.

    Returns a dict with keys "auc", "y_true", "y_pred", "y_proba",
    and, if `collect_audit` is True, "selected_feature_lists" (the
    concatenation of features selected in every outer fold) and
    "audit_df" (one row per outer fold).
    """
    outer_cv = LeaveOneOut()

    oof_true, oof_pred, oof_proba = [], [], []
    selected_feature_lists = []
    audit_rows = []

    positive_code = label_encoder.transform([positive_label])[0]

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_raw, y), start=1):
        X_train, X_test = X_raw.iloc[train_idx], X_raw.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline = Pipeline(
            [
                (
                    "prefilter",
                    ExpressionFilterLog2(
                        min_detected_fraction=min_detected_fraction,
                        min_mean_expression=min_mean_expression,
                    ),
                ),
                ("scale", StandardScaler()),
                ("select", SelectKBest(score_func=f_classif)),
                ("clf", clone(estimator)),
            ]
        )

        k_grid_fold = [k for k in k_grid if k <= X_train.shape[1]]
        if not k_grid_fold:
            k_grid_fold = [X_train.shape[1]]

        full_grid = dict(param_grid)
        full_grid["select__k"] = k_grid_fold

        inner_cv = safe_inner_cv(y_train, random_seed=random_seed)

        if inner_cv is not None:
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=full_grid,
                cv=inner_cv,
                scoring="accuracy",
                n_jobs=n_jobs_inner,
                error_score=0,
                refit=True,
            )
            search.fit(X_train, y_train)
            best_pipeline = search.best_estimator_
            best_params = search.best_params_
        else:
            # Too few minority-class samples in this training fold for
            # inner CV: fall back to the middle of the k grid.
            middle_k = k_grid_fold[len(k_grid_fold) // 2]
            pipeline.set_params(select__k=middle_k)
            pipeline.fit(X_train, y_train)
            best_pipeline = pipeline
            best_params = {"select__k": middle_k}

        pred = best_pipeline.predict(X_test)[0]
        proba = best_pipeline.predict_proba(X_test)[0]

        positive_index = list(best_pipeline.named_steps["clf"].classes_).index(positive_code)
        positive_proba = proba[positive_index]

        oof_true.append(y_test[0])
        oof_pred.append(pred)
        oof_proba.append(positive_proba)

        if collect_audit:
            prefilter = best_pipeline.named_steps["prefilter"]
            select_mask = best_pipeline.named_steps["select"].get_support()
            # The prefilter never drops columns (only zeroes them out),
            # so the selector mask aligns directly with the raw columns.
            selected_features = X_raw.columns[select_mask].tolist()
            selected_feature_lists.extend(selected_features)

            true_label = label_encoder.inverse_transform([y_test[0]])[0]
            predicted_label = label_encoder.inverse_transform([pred])[0]

            audit_rows.append(
                {
                    "fold": fold,
                    "test_sample": X_raw.index[test_idx[0]],
                    "true_class": true_label,
                    "predicted_class": predicted_label,
                    "P_positive_class": positive_proba,
                    "correct": true_label == predicted_label,
                    "features_passed_prefilter": int(prefilter.keep_mask_.sum()),
                    "selected_k": best_params["select__k"],
                    "best_params": str(best_params),
                }
            )

    oof_true = np.asarray(oof_true)
    oof_pred = np.asarray(oof_pred)
    oof_proba = np.asarray(oof_proba)

    result = {
        "auc": roc_auc_score(oof_true, oof_proba),
        "y_true": oof_true,
        "y_pred": oof_pred,
        "y_proba": oof_proba,
    }

    if collect_audit:
        result["selected_feature_lists"] = selected_feature_lists
        result["audit_df"] = pd.DataFrame(audit_rows)

    return result


def run_permutation_test(
    X: pd.DataFrame,
    y: np.ndarray,
    models: dict,
    label_encoder: LabelEncoder,
    positive_label: str,
    n_permutations: int = N_PERMUTATIONS,
    random_seed: int = RANDOM_SEED,
) -> dict:
    """
    Model-selection-aware permutation test.

    For every permutation, the label vector is shuffled and the
    entire strict nested-CV procedure is repeated for every
    candidate model; the best AUC across models forms one draw of
    the null distribution.

    Returns a dict with "null_best_aucs" (one value per permutation,
    the best AUC across models), "null_model_aucs" (per-model AUC
    arrays), and "winning_models" (the model with the best AUC in
    each permutation).
    """
    rng = np.random.RandomState(random_seed)

    null_best_aucs = []
    null_model_aucs = {name: [] for name in models}
    winning_models = []

    for permutation in range(n_permutations):
        y_perm = rng.permutation(y)
        perm_aucs = {}

        for model_name, (estimator, param_grid) in models.items():
            result = run_nested_cv(
                X_raw=X,
                y=y_perm,
                estimator=estimator,
                param_grid=param_grid,
                label_encoder=label_encoder,
                positive_label=positive_label,
                collect_audit=False,
            )
            perm_aucs[model_name] = result["auc"]
            null_model_aucs[model_name].append(result["auc"])

        best_model = max(perm_aucs, key=perm_aucs.get)
        null_best_aucs.append(perm_aucs[best_model])
        winning_models.append(best_model)

        if (permutation + 1) == 1 or (permutation + 1) % 10 == 0:
            print(f"...{permutation + 1}/{n_permutations} permutations completed")

    return {
        "null_best_aucs": np.asarray(null_best_aucs),
        "null_model_aucs": {name: np.asarray(values) for name, values in null_model_aucs.items()},
        "winning_models": winning_models,
    }
