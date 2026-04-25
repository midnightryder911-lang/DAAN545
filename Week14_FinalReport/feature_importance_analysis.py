#!/usr/bin/env python3
"""
Author: Mark Nelson
Course: DAAN 545 – Data Mining
Project: Team Final Report

Section: Feature Importance Analysis

Description:
This script performs supervised learning analysis on cybersecurity data to identify
key factors influencing:

1. Financial Loss (Regression)
2. Incident Resolution Time (Regression)
3. Incident Severity (Classification)

Methods Used:
- Random Forest Models (Regressor & Classifier)
- k-Fold Cross-Validation (k=5)
- Feature Importance Ranking
- SHAP (Explainable AI)

Key Purpose:
To support Section 3 of the team project by identifying the most influential
features and translating them into real-world cybersecurity insights.

Outputs:
- Feature importance rankings (Top 10)
- Visualization plots (saved to /outputs/)
- SHAP summary plot for interpretability
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for saving plots

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path("./data/Global_Cybersecurity_Threats_2015-2024_cleaned_weighted(in).csv")
OUTPUT_DIR = Path("./outputs")

# Dataset column names observed in the provided CSV
TARGET_FINANCIAL_LOSS = "Financial Loss (in Million $)"
TARGET_RESOLUTION_TIME = "Incident Resolution Time (in Hours)"
TARGET_SEVERITY = "Incident Severity"  # may or may not exist in the dataset

SAMPLE_WEIGHT_COL = "sample_weight"  # optional; used if present

N_ESTIMATORS = 200  # keep runtime reasonable while still stable
MAX_DEPTH: Optional[int] = 14  # cap depth to speed up CV and SHAP


@dataclass(frozen=True)
class ModelRunResult:
    name: str
    task_type: str  # "regression" | "classification"
    pipeline: Pipeline
    feature_names: np.ndarray
    importances: np.ndarray
    cv_scores: np.ndarray
    test_score: float


def _ensure_outputs_dir() -> None:
    # Centralizes output directory creation so all downstream plotting and
    # file-saving steps are deterministic and do not fail due to missing folders.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    # Explicit path validation provides a clear, early failure mode if the dataset
    # has been moved/renamed, which is preferable to silent downstream errors.
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path.resolve()}")
    # Pandas handles typed parsing and preserves column names as provided, which
    # supports reliable feature/target selection later in the pipeline.
    return pd.read_csv(path)


def create_incident_severity_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    If 'Incident Severity' isn't present, create a simple 3-class label from
    financial loss quantiles so the classification requirement can run end-to-end.
    """
    if TARGET_SEVERITY in df.columns:
        return df

    if TARGET_FINANCIAL_LOSS not in df.columns:
        raise ValueError(
            f"Cannot create '{TARGET_SEVERITY}' because '{TARGET_FINANCIAL_LOSS}' is missing."
        )

    # Quantile-based bins provide a defensible, distribution-aware way to create
    # an ordinal severity proxy when a ground-truth severity label is unavailable.
    q1, q2 = df[TARGET_FINANCIAL_LOSS].quantile([0.33, 0.66]).to_numpy()

    def label(loss: float) -> str:
        if loss <= q1:
            return "Low"
        if loss <= q2:
            return "Medium"
        return "High"

    df = df.copy()
    df[TARGET_SEVERITY] = df[TARGET_FINANCIAL_LOSS].apply(label)

    print(
        f"[info] Column '{TARGET_SEVERITY}' not found; created it from "
        f"'{TARGET_FINANCIAL_LOSS}' quantiles (Low/Medium/High)."
    )
    return df


def split_features_targets(
    df: pd.DataFrame, target_col: str
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")

    # Preserve optional per-row weights (when provided) to reflect prior
    # preprocessing decisions (e.g., down-weighting outliers) during training.
    sample_weight = df[SAMPLE_WEIGHT_COL] if SAMPLE_WEIGHT_COL in df.columns else None

    # Split data into features (X) and target (y)
    # Prevents data leakage between models
    #
    # In multi-target scripts, it is critical to remove the *other* target columns
    # from X. Otherwise, a model may implicitly “cheat” by learning from a related
    # outcome variable that would not be available at prediction time.
    drop_cols = {TARGET_FINANCIAL_LOSS, TARGET_RESOLUTION_TIME, TARGET_SEVERITY}
    if SAMPLE_WEIGHT_COL in df.columns:
        # Row weights are metadata for fitting, not predictive inputs.
        drop_cols.add(SAMPLE_WEIGHT_COL)

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target_col]
    return X, y, sample_weight


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    # Programmatic dtype-based detection allows the same script to generalize to
    # future versions of the dataset without hard-coding feature lists.
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            # Median imputation is robust to skew and outliers, and avoids dropping
            # records which would reduce statistical power.
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            # Most-frequent imputation yields a deterministic placeholder that
            # preserves dataset size while maintaining valid category values.
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # One-hot encoding converts nominal categories into a model-compatible
            # numeric representation without imposing an artificial ordering.
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        # Dropping unspecified columns ensures only intended predictors are used,
        # which improves reproducibility and guards against accidental leakage.
        remainder="drop",
        # Cleaner feature names improve interpretability of the importance ranking.
        verbose_feature_names_out=False,
    )
    return preprocessor, numeric_features, categorical_features


def get_feature_names(preprocessor: ColumnTransformer) -> np.ndarray:
    # Feature names must reflect the *post-encoding* design matrix so that
    # importance scores can be mapped back to human-readable predictors.
    return preprocessor.get_feature_names_out()


def rank_importances(feature_names: np.ndarray, importances: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({"feature": feature_names, "importance": importances})
    out = out.sort_values("importance", ascending=False).reset_index(drop=True)
    return out


def plot_top_importances(
    ranked: pd.DataFrame, title: str, output_path: Path, top_n: int = 20
) -> None:
    # Visualizing the top features supports rapid communication of results and is
    # typically more actionable than inspecting full tables of coefficients/scores.
    top = ranked.head(top_n).iloc[::-1]  # reverse for horizontal bar chart
    plt.figure(figsize=(10, max(4, 0.35 * len(top))))
    sns.barplot(data=top, x="importance", y="feature", orient="h", color="#4C72B0")
    plt.title(title)
    plt.xlabel("Feature importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def train_and_evaluate_regression(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[pd.Series],
    model_name: str,
    random_state: int = 42,
) -> ModelRunResult:
    # Preprocessing is fitted *inside* the pipeline so that cross-validation and
    # train/test evaluation do not “peek” at held-out data distributions.
    preprocessor, _, _ = build_preprocessor(X)
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=random_state,
        n_jobs=-1,
        max_depth=MAX_DEPTH,
    )

    # A single pipeline guarantees identical preprocessing at training and
    # evaluation time, which is essential for fair comparison and deployment.
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
        X,
        y,
        sample_weight,
        test_size=0.2,
        random_state=random_state,
    )

    fit_kwargs = {}
    if sw_train is not None:
        # Sample weights allow the model to emphasize or de-emphasize observations
        # according to prior domain/quality assessments.
        fit_kwargs["model__sample_weight"] = sw_train
    # Random Forests provide strong non-linear baselines and yield intrinsic
    # impurity-based feature importances for interpretability.
    pipeline.fit(X_train, y_train, **fit_kwargs)

    preds = pipeline.predict(X_test)
    test_r2 = r2_score(y_test, preds, sample_weight=sw_test)

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    # cross_val_score doesn't consistently support sample_weight across all sklearn versions
    # Cross-validation estimates generalization performance more reliably than a
    # single split by averaging over multiple train/validation partitions.
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2", n_jobs=-1)

    fitted_preprocessor: ColumnTransformer = pipeline.named_steps["preprocess"]
    feature_names = get_feature_names(fitted_preprocessor)
    # Importances are extracted from the fitted forest and aligned to the encoded
    # feature space (including one-hot expanded variables).
    importances = pipeline.named_steps["model"].feature_importances_

    return ModelRunResult(
        name=model_name,
        task_type="regression",
        pipeline=pipeline,
        feature_names=feature_names,
        importances=importances,
        cv_scores=cv_scores,
        test_score=float(test_r2),
    )


def train_and_evaluate_classification(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[pd.Series],
    model_name: str,
    random_state: int = 42,
) -> ModelRunResult:
    # The same preprocessing strategy is applied for classification so that
    # categorical levels and imputations are learned only from training folds.
    preprocessor, _, _ = build_preprocessor(X)
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
        max_depth=MAX_DEPTH,
    )

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
        X,
        y,
        sample_weight,
        test_size=0.2,
        random_state=random_state,
        # Stratification helps preserve class proportions, producing a more
        # informative test estimate when classes are imbalanced.
        stratify=y if y.nunique() > 1 else None,
    )

    fit_kwargs = {}
    if sw_train is not None:
        fit_kwargs["model__sample_weight"] = sw_train
    pipeline.fit(X_train, y_train, **fit_kwargs)

    preds = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, preds, sample_weight=sw_test)

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    # CV accuracy provides a stability check against variance from a single split.
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

    fitted_preprocessor: ColumnTransformer = pipeline.named_steps["preprocess"]
    feature_names = get_feature_names(fitted_preprocessor)
    importances = pipeline.named_steps["model"].feature_importances_

    return ModelRunResult(
        name=model_name,
        task_type="classification",
        pipeline=pipeline,
        feature_names=feature_names,
        importances=importances,
        cv_scores=cv_scores,
        test_score=float(test_acc),
    )


def run_shap_summary(
    fitted_pipeline: Pipeline,
    X_background: pd.DataFrame,
    output_path: Path,
    max_background: int = 500,
    max_explain: int = 500,
) -> None:
    """
    SHAP analysis provides local (per-observation) attributions that complement
    global feature importance by showing directionality and interaction patterns.

    The script explains a subset of rows to keep runtime and memory usage
    manageable while still producing a representative summary plot.
    """
    preprocessor: ColumnTransformer = fitted_pipeline.named_steps["preprocess"]
    model = fitted_pipeline.named_steps["model"]

    Xb = X_background.sample(n=min(max_background, len(X_background)), random_state=42)
    Xb_t = preprocessor.transform(Xb)

    Xe = X_background.sample(n=min(max_explain, len(X_background)), random_state=7)
    Xe_t = preprocessor.transform(Xe)

    feature_names = preprocessor.get_feature_names_out()

    # TreeExplainer is efficient and well-suited for tree ensembles such as Random Forests.
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xe_t)

    # For classification, shap_values may be a list (one array per class).
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[0]
    else:
        shap_values_to_plot = shap_values

    plt.figure()
    shap.summary_plot(
        shap_values_to_plot,
        features=Xe_t,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def print_top10(result: ModelRunResult) -> None:
    ranked = rank_importances(result.feature_names, result.importances)
    print("\n" + "=" * 80)
    print(f"Model: {result.name} ({result.task_type})")
    if result.task_type == "regression":
        print(f"Test R^2: {result.test_score:.4f}")
        print(
            f"5-fold CV R^2: mean={result.cv_scores.mean():.4f}, std={result.cv_scores.std():.4f}"
        )
    else:
        print(f"Test Accuracy: {result.test_score:.4f}")
        print(
            f"5-fold CV Accuracy: mean={result.cv_scores.mean():.4f}, std={result.cv_scores.std():.4f}"
        )
    print("\nTop 10 features:")
    print(ranked.head(10).to_string(index=False))


def main() -> None:
    # Ensure file outputs are reproducible across environments and that plots are
    # consistently styled for inclusion in reports.
    _ensure_outputs_dir()
    sns.set_theme(style="whitegrid")

    # Load the cleaned dataset used throughout the analysis.
    df = load_dataset(DATA_PATH)
    # If severity is not present in the source data, create a transparent proxy
    # label so that classification analysis can still be demonstrated end-to-end.
    df = create_incident_severity_if_missing(df)

    # --- Regression: Financial Loss ---
    print("[info] Training model for Financial Loss (regression)...", flush=True)
    X_loss, y_loss, sw_loss = split_features_targets(df, TARGET_FINANCIAL_LOSS)
    res_loss = train_and_evaluate_regression(
        X_loss, y_loss, sw_loss, model_name="RandomForestRegressor - Financial Loss"
    )
    print_top10(res_loss)
    ranked_loss = rank_importances(res_loss.feature_names, res_loss.importances)
    # Save top features to CSV for reproducibility and reporting purposes
    ranked_loss.head(10).to_csv(OUTPUT_DIR / "top10_financial_loss.csv", index=False)
    plot_top_importances(
        ranked_loss,
        title="Feature Importance - Financial Loss (Random Forest Regressor)",
        output_path=OUTPUT_DIR / "feature_importance_financial_loss.png",
        top_n=20,
    )

    # --- Regression: Resolution Time ---
    print("[info] Training model for Resolution Time (regression)...", flush=True)
    X_rt, y_rt, sw_rt = split_features_targets(df, TARGET_RESOLUTION_TIME)
    res_rt = train_and_evaluate_regression(
        X_rt,
        y_rt,
        sw_rt,
        model_name="RandomForestRegressor - Resolution Time",
    )
    print_top10(res_rt)
    ranked_rt = rank_importances(res_rt.feature_names, res_rt.importances)
    ranked_rt.head(10).to_csv(OUTPUT_DIR / "top10_resolution_time.csv", index=False)
    plot_top_importances(
        ranked_rt,
        title="Feature Importance - Resolution Time (Random Forest Regressor)",
        output_path=OUTPUT_DIR / "feature_importance_resolution_time.png",
        top_n=20,
    )

    # --- Classification: Incident Severity ---
    print("[info] Training model for Incident Severity (classification)...", flush=True)
    X_sev, y_sev, sw_sev = split_features_targets(df, TARGET_SEVERITY)
    res_sev = train_and_evaluate_classification(
        X_sev,
        y_sev,
        sw_sev,
        model_name="RandomForestClassifier - Incident Severity",
    )
    print_top10(res_sev)
    ranked_sev = rank_importances(res_sev.feature_names, res_sev.importances)
    ranked_sev.head(10).to_csv(OUTPUT_DIR / "top10_severity.csv", index=False)
    plot_top_importances(
        ranked_sev,
        title="Feature Importance - Incident Severity (Random Forest Classifier)",
        output_path=OUTPUT_DIR / "feature_importance_incident_severity.png",
        top_n=20,
    )

    # --- SHAP analysis for at least one model (use Financial Loss by default) ---
    try:
        print("[info] Running SHAP analysis (Financial Loss model)...", flush=True)
        run_shap_summary(
            fitted_pipeline=res_loss.pipeline,
            X_background=X_loss,
            output_path=OUTPUT_DIR / "shap_summary_financial_loss.png",
        )
        print("\n[info] Saved SHAP summary plot for Financial Loss model.")
    except Exception as e:
        # Keep end-to-end execution robust; still surface the issue clearly.
        print(f"\n[warn] SHAP analysis failed: {e}")

    print("\n[done] Saved outputs to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
