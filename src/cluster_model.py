"""K-Means customer segmentation with an automatically selected cluster count."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "dataset" / "Mall_Customers.csv"
FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]
RANDOM_STATE = 42
MAX_CLUSTERS = 10


def load_and_clean_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the source data and discard records without usable clustering features."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    customers = pd.read_csv(path)
    missing_columns = [column for column in FEATURES if column not in customers.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required feature columns: {missing_columns}")

    features = customers[FEATURES].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    valid_rows = features.notna().all(axis=1)
    dropped_count = int((~valid_rows).sum())
    cleaned_customers = customers.loc[valid_rows].copy().reset_index(drop=True)
    cleaned_features = features.loc[valid_rows].reset_index(drop=True)

    if len(cleaned_features) < 3:
        raise ValueError(
            "At least three rows with valid income and spending-score values are required."
        )
    if dropped_count:
        print(f"Dropped {dropped_count} row(s) with missing or invalid feature values.")

    return cleaned_customers, cleaned_features


def select_cluster_count(
    scaled_features: np.ndarray, max_clusters: int = MAX_CLUSTERS
) -> tuple[int, list[int], list[float]]:
    """Compute WCSS and choose the elbow as the point farthest from the end-to-end line."""
    upper_bound = min(max_clusters, len(scaled_features) - 1)
    cluster_counts = list(range(1, upper_bound + 1))
    if len(cluster_counts) < 3:
        return 1, cluster_counts, [
            float(
                KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
                .fit(scaled_features)
                .inertia_
            )
            for k in cluster_counts
        ]

    wcss = [
        float(
            KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            .fit(scaled_features)
            .inertia_
        )
        for k in cluster_counts
    ]
    if max(wcss) <= np.finfo(float).eps:
        return 1, cluster_counts, wcss

    points = np.column_stack((cluster_counts, wcss)).astype(float)
    start, end = points[0], points[-1]
    line = end - start
    line_length = np.linalg.norm(line)
    if line_length <= np.finfo(float).eps:
        return 1, cluster_counts, wcss

    distances = np.abs(
        line[0] * (start[1] - points[:, 1])
        - (start[0] - points[:, 0]) * line[1]
    ) / line_length
    elbow_index = int(np.argmax(distances[1:-1])) + 1
    return cluster_counts[elbow_index], cluster_counts, wcss


def save_elbow_plot(cluster_counts: list[int], wcss: list[float], output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(x=cluster_counts, y=wcss, marker="o", ax=ax)
    ax.set(title="Elbow Method for K-Means", xlabel="Number of clusters (k)", ylabel="WCSS")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_segment_plot(
    result: pd.DataFrame, centers: np.ndarray, output_path: Path
) -> None:
    plot_data = result.copy()
    plot_data["Segment"] = plot_data["Segment"].astype(str)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=plot_data,
        x=FEATURES[0],
        y=FEATURES[1],
        hue="Segment",
        palette="tab10",
        s=75,
        ax=ax,
    )
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="X",
        s=220,
        c="black",
        edgecolor="white",
        linewidth=1,
        label="Centroids",
    )
    ax.set_title("Customer Segments")
    ax.legend(title="Segment", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    customers, features = load_and_clean_data()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    optimal_k, cluster_counts, wcss = select_cluster_count(scaled_features)

    model = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
    zero_based_segments = model.fit_predict(scaled_features)
    result = customers.copy()
    result["Segment"] = zero_based_segments + 1

    dataset_dir = PROJECT_ROOT / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_csv = dataset_dir / "Customer_Segments.csv"
    result.to_csv(output_csv, index=False)

    save_elbow_plot(cluster_counts, wcss, PROJECT_ROOT / "elbow_curve.png")
    original_scale_centers = scaler.inverse_transform(model.cluster_centers_)
    save_segment_plot(result, original_scale_centers, PROJECT_ROOT / "customer_segments.png")

    print(f"Selected {optimal_k} cluster(s) using the WCSS elbow method.")
    print("WCSS: " + ", ".join(f"k={k}: {value:.3f}" for k, value in zip(cluster_counts, wcss)))
    print(f"Segmented data saved to: {output_csv}")
    print(f"Plots saved to: {PROJECT_ROOT / 'elbow_curve.png'} and "
          f"{PROJECT_ROOT / 'customer_segments.png'}")


if __name__ == "__main__":
    main()
