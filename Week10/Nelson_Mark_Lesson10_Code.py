"""
Mall Customer Segmentation — preprocessing, K-Means, DBSCAN tuning, and saved figures.

Loads the Mall_Customers dataset, preprocesses features, runs K-Means diagnostics
and a k=5 fit, sweeps DBSCAN eps values, selects a best eps, and saves the tuned
DBSCAN plot next to this script.
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
DATA_PATH = "/home/mnelson/daan545_lesson10/Mall_Customers.csv"
FINAL_K = 5  # chosen cluster count for the final K-Means model

# PNG outputs are written beside this file (stable regardless of cwd).
OUTPUT_DIR = Path(__file__).resolve().parent

# DBSCAN hyperparameters (density-based clustering on scaled features)
DBSCAN_MIN_SAMPLES = 5
# Candidate neighborhood radii (scaled feature space); tuned run picks one below.
DBSCAN_EPS_CANDIDATES = [0.7, 0.9, 1.1, 1.3]


def dbscan_cluster_and_noise_counts(labels: np.ndarray) -> tuple[int, int]:
    """Return (number of clusters, number of noise points); label -1 is noise."""
    unique = set(labels.tolist())
    n_noise = int(np.sum(labels == -1))
    n_clusters = sum(1 for lab in unique if lab != -1)
    return n_clusters, n_noise


def plot_dbscan_income_spending(
    ax,
    df_plot: pd.DataFrame,
    labels: np.ndarray,
    income_col: str,
    spend_col: str,
    title: str,
) -> None:
    """Scatter: income vs spending, colored by DBSCAN labels; noise shown as gray x."""
    unique = sorted(set(labels.tolist()))
    non_noise = [lab for lab in unique if lab != -1]
    for label in non_noise:
        mask = labels == label
        ax.scatter(
            df_plot.loc[mask, income_col],
            df_plot.loc[mask, spend_col],
            label=f"Cluster {label}",
            alpha=0.75,
            edgecolors="k",
            linewidths=0.2,
        )
    if -1 in unique:
        mask_noise = labels == -1
        ax.scatter(
            df_plot.loc[mask_noise, income_col],
            df_plot.loc[mask_noise, spend_col],
            c="dimgray",
            marker="x",
            s=55,
            linewidths=1.25,
            label="Noise",
            zorder=5,
        )
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.set_title(title)
    ax.legend(title="DBSCAN cluster")
    ax.grid(True, alpha=0.3)


def select_best_eps_dbscan(
    rows: list[tuple[float, int, int]],
) -> float:
    """
    Pick eps with a balance of fewer noise points and multi-cluster structure.

    Preference order:
    1. At least two clusters (otherwise segmentation is trivial).
    2. Minimize noise count (denser neighborhoods).
    3. Tie-break: cluster count closest to FINAL_K (aligns with K-Means structure).
    """
    valid = [r for r in rows if r[1] >= 2]
    pool = valid if valid else rows
    best_eps, _, _ = min(
        pool,
        key=lambda r: (r[2], abs(r[1] - FINAL_K)),
    )
    return best_eps


def main() -> None:
    # -----------------------------------------------------------------------
    # 2. Load data
    # -----------------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)

    # -----------------------------------------------------------------------
    # 3. Initial exploration: first rows and schema
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("First 5 rows (raw)")
    print("=" * 60)
    print(df.head(), "\n")

    print("=" * 60)
    print("Dataset info (dtypes, non-null counts)")
    print("=" * 60)
    df.info()
    print()

    # -----------------------------------------------------------------------
    # 4. Drop identifier column (not a modeling feature)
    # -----------------------------------------------------------------------
    df = df.drop(columns=["CustomerID"])

    # -----------------------------------------------------------------------
    # 5. Encode Gender as integers (e.g., Female/Male -> 0/1)
    # -----------------------------------------------------------------------
    gender_encoder = LabelEncoder()
    df["Gender"] = gender_encoder.fit_transform(df["Gender"])

    # -----------------------------------------------------------------------
    # 6. Standardize all features to zero mean and unit variance
    # -----------------------------------------------------------------------
    feature_columns = list(df.columns)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    df_preprocessed = pd.DataFrame(X_scaled, columns=feature_columns)

    # -----------------------------------------------------------------------
    # 7. Preprocessed preview
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("First 5 rows (preprocessed — scaled features)")
    print("=" * 60)
    print(df_preprocessed.head())

    # -----------------------------------------------------------------------
    # 8. K-Means diagnostics (scaled features; sweep k before choosing FINAL_K)
    # -----------------------------------------------------------------------
    # Use a NumPy array so clustering APIs receive a dense numeric matrix.
    X = df_preprocessed.to_numpy()
    rng = 42  # reproducible cluster initializations
    n_init = "auto"  # sklearn default batch count

    # Inertia (within-cluster sum of squares) for k = 1, ..., 10
    k_inertia_range = list(range(1, 11))
    inertias: list[float] = []
    for k in k_inertia_range:
        kmeans = KMeans(n_clusters=k, random_state=rng, n_init=n_init)
        kmeans.fit(X)
        inertias.append(float(kmeans.inertia_))

    # Silhouette score requires at least 2 clusters; evaluate k = 2, ..., 10
    k_silhouette_range = list(range(2, 11))
    silhouette_scores: list[float] = []
    for k in k_silhouette_range:
        kmeans = KMeans(n_clusters=k, random_state=rng, n_init=n_init)
        labels = kmeans.fit_predict(X)
        silhouette_scores.append(float(silhouette_score(X, labels)))

    print()
    print("=" * 60)
    print("K-Means diagnostics (not a final fitted model)")
    print("=" * 60)
    print(f"k for inertia:        {k_inertia_range}")
    print(f"Inertia list:         {inertias}")
    print()
    print(f"k for silhouette:     {k_silhouette_range}")
    print(f"Silhouette score list: {silhouette_scores}")

    # Elbow plot: inertia vs k
    fig_elbow, ax_elbow = plt.subplots(figsize=(8, 5))
    ax_elbow.plot(k_inertia_range, inertias, marker="o")
    ax_elbow.set_xlabel("Number of clusters (k)")
    ax_elbow.set_ylabel("Inertia (within-cluster sum of squares)")
    ax_elbow.set_title("Elbow Method for K-Means")
    ax_elbow.set_xticks(k_inertia_range)
    ax_elbow.grid(True, alpha=0.3)
    fig_elbow.tight_layout()
    fig_elbow.savefig(OUTPUT_DIR / "elbow_method.png", dpi=150)

    # Silhouette plot: score vs k
    fig_sil, ax_sil = plt.subplots(figsize=(8, 5))
    ax_sil.plot(k_silhouette_range, silhouette_scores, marker="o", color="darkgreen")
    ax_sil.set_xlabel("Number of clusters (k)")
    ax_sil.set_ylabel("Silhouette score")
    ax_sil.set_title("Silhouette Score vs. Number of Clusters")
    ax_sil.set_xticks(k_silhouette_range)
    ax_sil.grid(True, alpha=0.3)
    fig_sil.tight_layout()
    fig_sil.savefig(OUTPUT_DIR / "silhouette_scores.png", dpi=150)

    # -----------------------------------------------------------------------
    # 9. Final K-Means model (k = 5) and cluster visualization
    # -----------------------------------------------------------------------
    # Fit on the same scaled matrix X used for diagnostics.
    kmeans_final = KMeans(
        n_clusters=FINAL_K, random_state=rng, n_init=n_init
    )
    cluster_labels = kmeans_final.fit_predict(X)

    # Attach labels to the original-scale feature table for interpretation.
    df["Cluster"] = cluster_labels

    income_col = "Annual Income (k$)"
    spend_col = "Spending Score (1-100)"

    fig_clusters, ax_clusters = plt.subplots(figsize=(9, 6))
    for cluster_id in range(FINAL_K):
        mask = df["Cluster"] == cluster_id
        ax_clusters.scatter(
            df.loc[mask, income_col],
            df.loc[mask, spend_col],
            label=f"Cluster {cluster_id}",
            alpha=0.75,
            edgecolors="k",
            linewidths=0.2,
        )
    ax_clusters.set_xlabel("Annual Income (k$)")
    ax_clusters.set_ylabel("Spending Score (1-100)")
    ax_clusters.set_title(f"Mall Customers — K-Means Segments (k = {FINAL_K})")
    ax_clusters.legend(title="Cluster")
    ax_clusters.grid(True, alpha=0.3)
    fig_clusters.tight_layout()
    fig_clusters.savefig(OUTPUT_DIR / "kmeans_clusters.png", dpi=150)

    # -----------------------------------------------------------------------
    # 10. DBSCAN: eps sweep, choose best eps, final fit and tuned plot
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("DBSCAN eps sweep (min_samples fixed)")
    print("=" * 60)
    sweep_rows: list[tuple[float, int, int]] = []
    for eps in DBSCAN_EPS_CANDIDATES:
        dbscan_try = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES)
        labels_try = dbscan_try.fit_predict(X)
        n_cl, n_nz = dbscan_cluster_and_noise_counts(labels_try)
        sweep_rows.append((eps, n_cl, n_nz))
        print(f"  eps = {eps}")
        print(f"    Number of clusters: {n_cl}")
        print(f"    Number of noise points: {n_nz}")
        print()

    best_eps = select_best_eps_dbscan(sweep_rows)
    best_row = next(r for r in sweep_rows if r[0] == best_eps)
    print("-" * 60)
    print(
        "Selected best eps (fewer noise when possible, at least 2 clusters; "
        "ties broken toward cluster count near K-Means k):"
    )
    print(f"  best eps = {best_eps}")
    print(f"  -> clusters = {best_row[1]}, noise = {best_row[2]}")
    print("-" * 60)

    dbscan_final = DBSCAN(eps=best_eps, min_samples=DBSCAN_MIN_SAMPLES)
    dbscan_labels = dbscan_final.fit_predict(X)
    df["DBSCAN_Cluster"] = dbscan_labels

    n_final_clusters, n_final_noise = dbscan_cluster_and_noise_counts(dbscan_labels)
    print()
    print("=" * 60)
    print("DBSCAN final model (tuned eps)")
    print("=" * 60)
    print(f"eps = {best_eps}, min_samples = {DBSCAN_MIN_SAMPLES}")
    print(f"Number of clusters found: {n_final_clusters}")
    print(f"Number of noise points:   {n_final_noise}")

    fig_dbscan, ax_dbscan = plt.subplots(figsize=(9, 6))
    plot_dbscan_income_spending(
        ax_dbscan,
        df,
        dbscan_labels,
        income_col,
        spend_col,
        title=(
            "Mall Customers — DBSCAN (tuned) "
            f"(eps={best_eps}, min_samples={DBSCAN_MIN_SAMPLES})"
        ),
    )
    fig_dbscan.tight_layout()
    fig_dbscan.savefig(OUTPUT_DIR / "dbscan_clusters_tuned.png", dpi=150)

    plt.show()


if __name__ == "__main__":
    main()
