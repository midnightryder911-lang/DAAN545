"""
Customer segmentation with K-Means and hierarchical clustering (Online Retail).

Author
    Mark Nelson

Course
    DAAN 545 — Data Mining (Penn State)

Assignment
    Lesson 8 Coding Exercise

Dataset
    ``data/Online Retail.xlsx`` — transactional retail records with columns including
    ``InvoiceNo``, ``StockCode``, ``Description``, ``Quantity``, ``InvoiceDate``,
    ``UnitPrice``, ``CustomerID``, and ``Country``.

Purpose
    Build a customer-level feature table, scale features for distance-based clustering,
    choose a cluster count using the elbow method for K-Means, compare K-Means to
    Ward hierarchical clustering, visualize results, and report silhouette scores.
    Outputs include plots under ``outputs/`` and a CSV of per-customer cluster labels.

Workflow (high level)
    1. Load the Excel file and inspect structure.
    2. Remove rows without a ``CustomerID`` so every remaining row maps to a customer.
    3. Compute line revenue ``TotalPrice = Quantity * UnitPrice``.
    4. Aggregate to one row per ``CustomerID`` (total spend and invoice frequency).
    5. Cluster on ``TotalSpent`` and ``PurchaseFrequency``.
    6. Standardize features; fit K-Means for several *k* (elbow), then a final K-Means
       at ``N_CLUSTERS``; fit hierarchical clustering for comparison.
    7. Save figures (elbow, cluster scatters, dendrogram) and export labeled customers.

Scaling vs. plotting
    K-Means and hierarchical clustering use *scaled* features so neither variable
    dominates distances simply because its units are larger. Scatter plots use the
    *original* ``TotalSpent`` and ``PurchaseFrequency`` so axes read in dollars and
    invoice counts, which is easier to interpret for a business audience.

Choosing *k*
    After reviewing ``outputs/elbow_curve.png``, update ``N_CLUSTERS`` so the final
    K-Means model, hierarchical model (same *k*), silhouettes, and plots align with
    your chosen elbow.

Dependencies
    pandas, openpyxl, scikit-learn, matplotlib, scipy

    Install if needed::

        pip install pandas openpyxl scikit-learn matplotlib scipy
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend: write PNGs without a display (servers/CI).
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Paths (script lives in Week8/; data and outputs sit beside it)
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Online Retail.xlsx"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Number of clusters for the final K-Means fit, hierarchical clustering, silhouettes,
# and scatter plots. Adjust after inspecting outputs/elbow_curve.png.
N_CLUSTERS = 4

# Elbow search range: k >= 2 so silhouette scores later remain well-defined.
K_MIN, K_MAX = 2, 10

# A dendrogram with one leaf per customer would be unreadable and expensive to render;
# we linkage-plot a random subset at fixed seed for a clear teaching visualization.
DENDROGRAM_SAMPLE_SIZE = 200
RNG_SEED = 42


def main() -> None:
    """
    Execute the full Week 8 workflow: load data, engineer customer features, scale,
    run elbow K-Means and final K-Means, hierarchical clustering, save plots and CSV,
    and print diagnostics including silhouette scores and K-Means centers in original units.
    """
    print("=" * 72)
    print("DAAN 545 Week 8 — Clustering: Online Retail customers")
    print("=" * 72)

    # --- [1] Load transaction data --------------------------------------------
    print("\n[1] Reading Excel:", DATA_PATH)
    df = pd.read_excel(DATA_PATH, engine="openpyxl")
    print(f"    Loaded shape: {df.shape}")
    print(f"    Columns: {list(df.columns)}")

    # --- [2] Require CustomerID -------------------------------------------------
    # Clustering is per customer; rows without ID cannot be assigned to anyone and
    # would break aggregation. Dropping them follows standard practice for this dataset.
    print("\n[2] Dropping rows with missing CustomerID …")
    before = len(df)
    df = df.dropna(subset=["CustomerID"])
    after = len(df)
    print(f"    Rows before: {before:,}  after: {after:,}  removed: {before - after:,}")

    # --- [3] Line-level revenue -------------------------------------------------
    # TotalPrice is the economic value of each invoice line; summing it later yields
    # each customer's lifetime spend in this sample window.
    print("\n[3] Creating TotalPrice = Quantity * UnitPrice …")
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # --- [4] One row per customer -----------------------------------------------
    # CustomerID is the clustering unit. TotalSpent captures monetary volume;
    # PurchaseFrequency (distinct invoices) captures engagement breadth—both are
    # common RFM-style inputs for segmentation.
    print("\n[4] Aggregating by CustomerID …")
    customer_df = df.groupby("CustomerID", as_index=False).agg(
        TotalSpent=("TotalPrice", "sum"),
        PurchaseFrequency=("InvoiceNo", "nunique"),
    )
    print(f"    Unique customers: {len(customer_df):,}")
    print("\n    First few rows (customer-level):")
    print(customer_df.head().to_string(index=False))

    # --- [5] Features used in the cluster model ---------------------------------
    # We deliberately use only these two columns so the exercise matches the assignment
    # and stays easy to visualize in 2D.
    feature_cols = ["TotalSpent", "PurchaseFrequency"]
    print("\n[5] Clustering features:", ", ".join(feature_cols))
    X_raw = customer_df[feature_cols].to_numpy(dtype=float)

    # --- [6] Standardize --------------------------------------------------------
    # K-Means (squared Euclidean distance) and Ward linkage are scale-sensitive.
    # StandardScaler puts both features on comparable variance so neither dominates.
    print("\n[6] Scaling features with StandardScaler …")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    print(
        "    Scaled feature means (should be ~0):",
        np.round(X_scaled.mean(axis=0), 4).tolist(),
    )
    print(
        "    Scaled feature stds (should be ~1):",
        np.round(X_scaled.std(axis=0, ddof=0), 4).tolist(),
    )

    # --- [7] Elbow method ---------------------------------------------------------
    # Inertia = within-cluster sum of squared distances to centroids (on scaled data).
    # It decreases as k increases; the "elbow" is where marginal gains flatten— a
    # heuristic for choosing a parsimonious k.
    print(f"\n[7] Elbow method: K-Means inertia for k = {K_MIN} … {K_MAX} …")
    k_values = list(range(K_MIN, K_MAX + 1))
    inertias = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RNG_SEED, n_init="auto")
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        print(f"    k={k:2d}  inertia={km.inertia_:,.2f}")

    fig_elbow, ax_elbow = plt.subplots(figsize=(8, 5))
    ax_elbow.plot(k_values, inertias, marker="o", color="steelblue")
    ax_elbow.set_xlabel("Number of clusters (k)")
    ax_elbow.set_ylabel("Inertia (within-cluster sum of squares)")
    ax_elbow.set_title("Elbow Method for K-Means (scaled features)")
    ax_elbow.set_xticks(k_values)
    ax_elbow.grid(True, alpha=0.3)
    elbow_path = OUTPUT_DIR / "elbow_curve.png"
    fig_elbow.tight_layout()
    fig_elbow.savefig(elbow_path, dpi=150)
    plt.close(fig_elbow)
    print(f"    Saved elbow plot: {elbow_path}")

    # --- [8] Final K-Means, centroids in business units, scatter ----------------
    print(f"\n[8] Final K-Means (k={N_CLUSTERS}) on scaled features …")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RNG_SEED, n_init="auto")
    kmeans_labels = kmeans.fit_predict(X_scaled)
    customer_df = customer_df.copy()
    customer_df["KMeansCluster"] = kmeans_labels

    # Cluster centers live in scaled space; inverse_transform maps them back to
    # TotalSpent (currency) and PurchaseFrequency (count) for interpretation.
    centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
    print("    K-Means cluster centers (original feature scale):")
    for c in range(N_CLUSTERS):
        ts, pf = centers_original[c]
        print(
            f"      Cluster {c}:  TotalSpent={ts:,.2f}  "
            f"PurchaseFrequency={pf:.2f}"
        )

    # Plot in original units so stakeholders see dollars and invoice counts.
    fig_km, ax_km = plt.subplots(figsize=(8, 6))
    scatter = ax_km.scatter(
        customer_df["TotalSpent"],
        customer_df["PurchaseFrequency"],
        c=kmeans_labels,
        cmap="tab10",
        alpha=0.5,
        edgecolors="none",
    )
    ax_km.set_xlabel("TotalSpent")
    ax_km.set_ylabel("PurchaseFrequency")
    ax_km.set_title(f"K-Means clusters (k={N_CLUSTERS}) — original feature scale")
    plt.colorbar(scatter, ax=ax_km, label="Cluster")
    ax_km.grid(True, alpha=0.3)
    kmeans_plot_path = OUTPUT_DIR / "kmeans_clusters.png"
    fig_km.tight_layout()
    fig_km.savefig(kmeans_plot_path, dpi=150)
    plt.close(fig_km)
    print(f"    Saved K-Means scatter: {kmeans_plot_path}")

    # --- [9] Hierarchical clustering + scatter ----------------------------------
    # Same k as K-Means for a direct comparison; Ward linkage minimizes variance
    # within merged clusters and pairs naturally with Euclidean distance on scaled data.
    print(f"\n[9] Hierarchical clustering (Agglomerative, Ward, k={N_CLUSTERS}) …")
    hc = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="ward")
    hc_labels = hc.fit_predict(X_scaled)
    customer_df["HierarchicalCluster"] = hc_labels

    fig_hc, ax_hc = plt.subplots(figsize=(8, 6))
    scatter_h = ax_hc.scatter(
        customer_df["TotalSpent"],
        customer_df["PurchaseFrequency"],
        c=hc_labels,
        cmap="tab10",
        alpha=0.5,
        edgecolors="none",
    )
    ax_hc.set_xlabel("TotalSpent")
    ax_hc.set_ylabel("PurchaseFrequency")
    ax_hc.set_title(
        f"Hierarchical (Ward) clusters (k={N_CLUSTERS}) — original feature scale"
    )
    plt.colorbar(scatter_h, ax=ax_hc, label="Cluster")
    ax_hc.grid(True, alpha=0.3)
    hc_plot_path = OUTPUT_DIR / "hierarchical_clusters.png"
    fig_hc.tight_layout()
    fig_hc.savefig(hc_plot_path, dpi=150)
    plt.close(fig_hc)
    print(f"    Saved hierarchical scatter: {hc_plot_path}")

    # --- [10] Dendrogram (sampled) ----------------------------------------------
    # scipy dendrogram expects a linkage matrix; plotting all ~4k leaves would be
    # illegible, so we show structure on a reproducible random sample.
    print(
        f"\n[10] Dendrogram (Ward linkage on a random sample of "
        f"{min(DENDROGRAM_SAMPLE_SIZE, len(customer_df))} customers) …"
    )
    rng = np.random.default_rng(RNG_SEED)
    n = X_scaled.shape[0]
    sample_idx = rng.choice(n, size=min(DENDROGRAM_SAMPLE_SIZE, n), replace=False)
    Z = linkage(X_scaled[sample_idx], method="ward")

    fig_den, ax_den = plt.subplots(figsize=(12, 5))
    dendrogram(Z, ax=ax_den, truncate_mode="level", p=5, color_threshold=0)
    ax_den.set_title(
        f"Hierarchical clustering dendrogram (Ward; sample n={len(sample_idx)})"
    )
    ax_den.set_xlabel("Index (truncated leaves)")
    ax_den.set_ylabel("Distance")
    dend_path = OUTPUT_DIR / "hierarchical_dendrogram.png"
    fig_den.tight_layout()
    fig_den.savefig(dend_path, dpi=150)
    plt.close(fig_den)
    print(f"    Saved dendrogram: {dend_path}")

    # --- [11] Silhouette scores --------------------------------------------------
    # Silhouette measures how well-separated clusters are (−1 to 1, higher is better).
    # Computed on scaled X so it matches the space where both algorithms operated.
    print(f"\n[11] Silhouette scores (scaled features, k={N_CLUSTERS}) …")
    sil_kmeans = silhouette_score(X_scaled, kmeans_labels)
    sil_hc = silhouette_score(X_scaled, hc_labels)
    print(f"    K-Means silhouette:        {sil_kmeans:.4f}")
    print(f"    Hierarchical silhouette:   {sil_hc:.4f}")

    # --- [12] Export labeled customer table --------------------------------------
    csv_path = OUTPUT_DIR / "clustered_customers.csv"
    customer_df.to_csv(csv_path, index=False)
    print(f"\n[12] Saved customer-level results: {csv_path}")

    # --- Summary -----------------------------------------------------------------
    print("\n" + "=" * 72)
    print("Cluster sizes (K-Means):")
    print(customer_df["KMeansCluster"].value_counts().sort_index().to_string())
    print("\nCluster sizes (Hierarchical):")
    print(customer_df["HierarchicalCluster"].value_counts().sort_index().to_string())
    print("=" * 72)
    print("Done. Plots and CSV saved under:", OUTPUT_DIR)
    print("=" * 72)


if __name__ == "__main__":
    main()
