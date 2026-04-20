"""
Nelson_Mark_Lesson13_Code.py

Lesson 13 Coding Exercise
Course: DAAN 545 Data Mining

Purpose:
This script implements a logistic regression model for a binary
classification problem using the Bank Customer Churn Prediction dataset.

Tasks completed:
1. Load and inspect the dataset
2. Preprocess the data
3. Separate features and target
4. Encode categorical variables
5. Split data into training and testing sets
6. Scale numeric features
7. Train a logistic regression model
8. Evaluate performance using accuracy, confusion matrix, ROC curve, and AUC

Notes:
- Logistic regression is used here because it is the required baseline model
  for this lesson.
- Model performance suggests the dataset may contain nonlinear relationships
  that are not fully captured by logistic regression.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Paths — CSV lives next to this script
# -----------------------------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parent / "Bank Customer Churn Prediction.csv"
CONFUSION_MATRIX_PNG = Path(__file__).resolve().parent / "confusion_matrix.png"
ROC_CURVE_PNG = Path(__file__).resolve().parent / "roc_curve.png"
MODEL_METRICS_TXT = Path(__file__).resolve().parent / "model_metrics.txt"

# Fixed random seed so train/test split and model results are reproducible
RANDOM_STATE = 42


def show_plot_or_close() -> None:
    """
    Show the current figure in interactive environments (e.g. Spyder, Jupyter
    with a GUI backend). On non-interactive backends (common in terminals /
    servers), close the figure to avoid warnings and free memory.
    """
    backend = matplotlib.get_backend().lower()
    if "agg" in backend or backend in ("pdf", "ps", "svg", "template"):
        plt.close()
    else:
        plt.show()


def main() -> None:
    # -------------------------------------------------------------------------
    # 1. Load the dataset
    # -------------------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)

    # -------------------------------------------------------------------------
    # 2. Basic dataset info (shape, dtypes, quick stats, missing values)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape (rows, columns): {df.shape}")
    print()
    print("Column dtypes:")
    print(df.dtypes)
    print()
    print("First few rows:")
    print(df.head())
    print()
    print("Numeric summary:")
    print(df.describe())
    print()
    print("Missing values per column:")
    print(df.isna().sum())
    print()

    # -------------------------------------------------------------------------
    # 3. Separate features and target (churn)
    #    Drop customer_id — it identifies rows but does not predict churn.
    # -------------------------------------------------------------------------
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    target_col = "churn"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # -------------------------------------------------------------------------
    # 4. One-hot encode categorical columns
    #    pandas detects object columns; country & gender are categorical here.
    #    drop_first=True avoids redundant “dummy” columns (multicollinearity).
    # -------------------------------------------------------------------------
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    print("Categorical columns one-hot encoded:", categorical_cols)
    print(f"Feature matrix shape after encoding: {X_encoded.shape}")
    print()

    # -------------------------------------------------------------------------
    # 5. Train / test split (70% train, 30% test)
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,  # keeps churn rate similar in train and test
    )

    # -------------------------------------------------------------------------
    # 6. Scale features (mean 0, variance 1) — important for logistic regression
    #    Fit scaler on training data only, then transform train and test.
    # -------------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------------------------------------------------------------
    # 7. Train logistic regression
    #    max_iter increased so optimization converges on this dataset.
    # -------------------------------------------------------------------------
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    # -------------------------------------------------------------------------
    # 8. Predict on the test set
    # -------------------------------------------------------------------------
    y_pred = model.predict(X_test_scaled)
    # Probabilities of class 1 (churn) — needed for ROC / AUC
    y_proba_churn = model.predict_proba(X_test_scaled)[:, 1]

    # -------------------------------------------------------------------------
    # 9. Accuracy
    #    Accuracy can be misleading on imbalanced datasets (one class much more
    #    common than the other), so also use the confusion matrix and AUC.
    # -------------------------------------------------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    print("=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)
    print(f"Test set accuracy: {accuracy:.4f}")
    print()

    # -------------------------------------------------------------------------
    # 10. Confusion matrix (seaborn heatmap)
    # -------------------------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
    )
    plt.title("Confusion Matrix — Bank Churn (Test Set)")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PNG, dpi=150)
    show_plot_or_close()

    print(f"Confusion matrix plot saved to: {CONFUSION_MATRIX_PNG}")
    print()

    # -------------------------------------------------------------------------
    # 11. ROC curve and AUC
    #     AUC reflects class separation across thresholds (overall ranking of
    #     predicted probabilities), not performance at a single decision cutoff.
    # -------------------------------------------------------------------------
    fpr, tpr, thresholds = roc_curve(y_test, y_proba_churn)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Bank Churn")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROC_CURVE_PNG, dpi=150)
    show_plot_or_close()

    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"ROC curve plot saved to: {ROC_CURVE_PNG}")

    print()
    print("Interpretation:")
    print("- The model performs reasonably well overall, but accuracy alone is not enough.")
    print("- The confusion matrix shows that many actual churn cases were missed.")
    print("- The AUC score indicates moderate class separation.")
    print("- This suggests logistic regression works as a baseline,")
    print("  but more advanced models may be needed if nonlinear patterns exist.")

    # Optional: persist headline metrics for reports or submission
    MODEL_METRICS_TXT.write_text(
        f"Accuracy: {accuracy:.4f}\n"
        f"AUC: {roc_auc:.4f}\n",
        encoding="utf-8",
    )
    print()
    print(f"Metrics saved to: {MODEL_METRICS_TXT}")


if __name__ == "__main__":
    main()
