"""Train the Gaia DR3 clustering models from the engineered Parquet dataset.

This script mirrors the modelling settings in the latest executed notebook:

* StandardScaler()
* KMeans(n_clusters=4, random_state=42, n_init=10)
* UMAP(n_neighbors=50, min_dist=0.1, n_components=2, random_state=42)
* HDBSCAN(min_cluster_size=500, min_samples=500,
          cluster_selection_method='leaf')

It also reproduces the notebook's evaluation sampling procedures:
* Silhouette: 10,000 rows selected with np.random.default_rng(42).
* DBCV: up to 20,000 non-noise rows selected with np.random.seed(42)'s
  legacy RandomState sampling, evaluated in the 2D UMAP embedding.

Run from the project root:
    python SCRIPTS/train.py

Optional paths:
    python SCRIPTS/train.py --input DATA/processed/gaia_engineered.parquet
    python SCRIPTS/train.py --model-dir MODEL --output-dir OUTPUTS
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import umap
import hdbscan
from hdbscan.validity import validity_index
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger("gaia_model_training")
SCRIPT_VERSION = "1.0.0"
RANDOM_SEED = 42

FEATURES = [
    "x",
    "y",
    "z",
    "tangential_velocity",
    "abs_g_mag",
    "bp_rp",
]

# These parameters are deliberately kept identical to the latest notebook.
SCALER_CONFIG: dict[str, Any] = {}
KMEANS_CONFIG: dict[str, Any] = {
    "n_clusters": 4,
    "random_state": RANDOM_SEED,
    "n_init": 10,
}
UMAP_CONFIG: dict[str, Any] = {
    "n_neighbors": 50,
    "min_dist": 0.1,
    "n_components": 2,
    "random_state": RANDOM_SEED,
}
HDBSCAN_CONFIG: dict[str, Any] = {
    "min_cluster_size": 500,
    "min_samples": 500,
    "cluster_selection_method": "leaf",
}


def sha256_file(path: Path) -> str:
    """Calculate a file's SHA256 without reading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(distribution: str) -> str:
    """Return installed package version for experiment metadata."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def validate_engineered_data(data: pd.DataFrame) -> None:
    """Check required columns, row count, and finite feature values."""
    if data.empty:
        raise ValueError("The engineered Gaia dataset is empty.")

    required = ["source_id", *FEATURES]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(
            "The engineered Parquet file is missing required columns: "
            + ", ".join(missing)
        )

    if data["source_id"].isna().any():
        raise ValueError("source_id contains missing values.")

    feature_values = data[FEATURES].to_numpy(dtype=np.float64)
    if not np.isfinite(feature_values).all():
        raise ValueError(
            "One or more model features contain NaN or infinite values. "
            "Fix the data pipeline before fitting models."
        )

    if len(data) <= 10_000:
        raise ValueError(
            "The current notebook evaluates silhouette on 10,000 observations. "
            "The dataset must contain more than 10,000 rows to reproduce that procedure."
        )


def calculate_silhouette(
    scaled_features: np.ndarray,
    kmeans_labels: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    """Reproduce the notebook's fixed-seed 10,000-row silhouette sample."""
    rng = np.random.default_rng(seed=RANDOM_SEED)
    sample_indices = rng.choice(
        scaled_features.shape[0], size=10_000, replace=False
    )

    sampled_features = scaled_features[sample_indices]
    sampled_labels = kmeans_labels[sample_indices]
    unique_sample_labels = np.unique(sampled_labels)

    if not 2 <= len(unique_sample_labels) < len(sampled_labels):
        raise ValueError(
            "Silhouette score is undefined for the sampled K-Means labels."
        )

    score = float(silhouette_score(sampled_features, sampled_labels))
    details = {
        "metric": "silhouette_score",
        "value": score,
        "sample_size": int(len(sample_indices)),
        "sampling": "numpy.default_rng(seed=42), without replacement",
        "feature_space": "standardized six-dimensional feature space",
    }
    return score, details


def calculate_dbcv(
    embedding: np.ndarray,
    hdbscan_labels: np.ndarray,
) -> tuple[float | None, dict[str, Any]]:
    """Reproduce the notebook's DBCV calculation on non-noise UMAP points.

    The notebook excludes label -1 before sampling, then calculates DBCV
    on up to 20,000 observations in the 2D UMAP embedding. This value must
    therefore be described as a sampled, non-noise, UMAP-space DBCV score.
    """
    non_noise_indices = np.where(hdbscan_labels != -1)[0]
    n_clusters = len(set(hdbscan_labels.tolist()) - {-1})
    sample_size = min(20_000, len(non_noise_indices))

    details: dict[str, Any] = {
        "metric": "hdbscan.validity.validity_index",
        "value": None,
        "sample_size": int(sample_size),
        "eligible_non_noise_rows": int(len(non_noise_indices)),
        "sampling": "numpy.random.seed(42) / legacy RandomState, without replacement",
        "noise_label_excluded_before_sampling": True,
        "feature_space": "two-dimensional UMAP embedding",
        "interpretation_note": (
            "This is not a full-catalogue DBCV score; it is calculated on a "
            "sample of non-noise observations in the UMAP embedding, matching "
            "the evaluation procedure in the current notebook."
        ),
    }

    if n_clusters < 2 or sample_size < 2:
        details["status"] = "skipped: fewer than two non-noise clusters or rows"
        return None, details

    # np.random.seed(42) followed by np.random.choice in the notebook uses
    # the legacy RandomState sampling sequence. A local RandomState reproduces
    # that sequence without changing global NumPy random state.
    rng = np.random.RandomState(RANDOM_SEED)
    sample_indices = rng.choice(
        non_noise_indices, size=sample_size, replace=False
    )

    sampled_embedding = embedding[sample_indices].astype(np.float64)
    sampled_labels = hdbscan_labels[sample_indices]

    if len(np.unique(sampled_labels)) < 2:
        details["status"] = "skipped: sampled rows contain fewer than two clusters"
        return None, details

    try:
        score = float(validity_index(sampled_embedding, sampled_labels))
        details["value"] = score
        details["status"] = "calculated"
        return score, details
    except Exception as exc:  # DBCV should not prevent model artifacts being saved.
        details["status"] = "failed"
        details["error"] = f"{type(exc).__name__}: {exc}"
        LOGGER.exception("DBCV calculation failed; continuing with model export.")
        return None, details


def build_hdbscan_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Build the same physical cluster summary used in the notebook."""
    summary = (
        data.groupby("hdbscan_label", as_index=False)
        .agg(
            Star_Count=("hdbscan_label", "size"),
            Mean_X_pc=("x", "mean"),
            Mean_Y_pc=("y", "mean"),
            Mean_Z_pc=("z", "mean"),
            Median_Velocity=("tangential_velocity", "median"),
            Mean_BP_RP=("bp_rp", "mean"),
            Mean_Abs_G=("abs_g_mag", "mean"),
        )
    )

    summary["Mean_Distance_pc"] = np.sqrt(
        summary["Mean_X_pc"] ** 2
        + summary["Mean_Y_pc"] ** 2
        + summary["Mean_Z_pc"] ** 2
    )

    # Keeping the distance column beside the three Cartesian coordinates,
    # consistent with the notebook's summary table.
    ordered_columns = [
        "hdbscan_label",
        "Star_Count",
        "Mean_X_pc",
        "Mean_Y_pc",
        "Mean_Z_pc",
        "Mean_Distance_pc",
        "Median_Velocity",
        "Mean_BP_RP",
        "Mean_Abs_G",
    ]
    return summary[ordered_columns].round(2).sort_values(
        "Star_Count", ascending=False
    )


def save_json(payload: dict[str, Any], destination: Path) -> None:
    """Write JSON metadata in a readable, portable format."""
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def run_training(
    input_path: Path,
    model_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Fit models, evaluate them, and serialize production artifacts."""
    input_path = input_path.expanduser().resolve()
    model_dir = model_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Engineered Parquet file not found: {input_path}\n"
            "Run `python SCRIPTS/data_pipeline.py` first."
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Reading engineered Gaia data: %s", input_path)
    data = pd.read_parquet(input_path)
    validate_engineered_data(data)

    row_count = int(len(data))
    source_sha256 = sha256_file(input_path)
    raw_features = data[FEATURES].to_numpy(dtype=np.float64)

    LOGGER.info("Rows loaded: %s", row_count)
    LOGGER.info("Feature order: %s", FEATURES)

    # 1. StandardScaler — fit on the complete engineered training catalogue.
    LOGGER.info("Fitting StandardScaler...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(raw_features)

    # 2. K-Means baseline — exact notebook parameters.
    LOGGER.info("Fitting K-Means baseline...")
    kmeans = KMeans(**KMEANS_CONFIG)
    kmeans_labels = kmeans.fit_predict(scaled_features)

    silhouette_value, silhouette_details = calculate_silhouette(
        scaled_features, kmeans_labels
    )
    LOGGER.info("K-Means clusters: %s", len(np.unique(kmeans_labels)))
    LOGGER.info("K-Means silhouette (10,000-row sample): %.4f", silhouette_value)

    # 3. UMAP — exact notebook parameters and six-dimensional scaled input.
    LOGGER.info("Fitting UMAP. This can take some time on the full catalogue...")
    reducer = umap.UMAP(**UMAP_CONFIG)
    embedding = reducer.fit_transform(scaled_features)

    if embedding.shape != (row_count, 2):
        raise RuntimeError(
            f"UMAP returned unexpected shape {embedding.shape}; expected {(row_count, 2)}."
        )
    if not np.isfinite(embedding).all():
        raise RuntimeError("UMAP produced NaN or infinite embedding coordinates.")

    # 4. HDBSCAN — exact final notebook parameters (Leaf selection).
    LOGGER.info("Fitting final HDBSCAN Leaf model...")
    clusterer = hdbscan.HDBSCAN(**HDBSCAN_CONFIG)
    hdbscan_labels = clusterer.fit_predict(embedding)

    # Generating prediction data after fitting so a future API can use
    # hdbscan.approximate_predict without changing the fitted labels.
    clusterer.generate_prediction_data()

    if not np.array_equal(hdbscan_labels, np.asarray(clusterer.labels_)):
        raise RuntimeError("HDBSCAN fit_predict labels differ from model.labels_.")

    n_hdbscan_clusters = int(len(set(hdbscan_labels.tolist()) - {-1}))
    n_noise = int(np.sum(hdbscan_labels == -1))
    noise_percentage = float(100.0 * n_noise / row_count)
    clustered_count = int(row_count - n_noise)

    dbcv_value, dbcv_details = calculate_dbcv(embedding, hdbscan_labels)

    LOGGER.info("HDBSCAN clusters (excluding noise): %s", n_hdbscan_clusters)
    LOGGER.info("HDBSCAN noise observations: %s", n_noise)
    LOGGER.info("HDBSCAN noise percentage: %.2f%%", noise_percentage)
    if dbcv_value is not None:
        LOGGER.info("DBCV on sampled non-noise UMAP points: %.4f", dbcv_value)
    else:
        LOGGER.warning("DBCV was not calculated successfully: %s", dbcv_details.get("status"))

    # 5. Combine original engineered data, model labels, and UMAP coordinates.
    clustered_data = data.copy()
    clustered_data["kmeans_label"] = kmeans_labels.astype(np.int32)
    clustered_data["hdbscan_label"] = hdbscan_labels.astype(np.int32)
    clustered_data["umap_1"] = embedding[:, 0]
    clustered_data["umap_2"] = embedding[:, 1]

    # Invariants: every input observation retains exactly one label from each model.
    if len(clustered_data) != row_count:
        raise RuntimeError("Unexpected row-count change during model output assembly.")
    if int(clustered_data["kmeans_label"].isna().sum()) != 0:
        raise RuntimeError("K-Means labels contain missing values.")
    if int(clustered_data["hdbscan_label"].isna().sum()) != 0:
        raise RuntimeError("HDBSCAN labels contain missing values.")

    # 6. Cluster summaries.
    hdbscan_summary = build_hdbscan_summary(clustered_data)
    kmeans_summary = (
        clustered_data.groupby("kmeans_label", as_index=False)
        .agg(
            Star_Count=("kmeans_label", "size"),
            Mean_X_pc=("x", "mean"),
            Mean_Y_pc=("y", "mean"),
            Mean_Z_pc=("z", "mean"),
            Median_Velocity=("tangential_velocity", "median"),
            Mean_BP_RP=("bp_rp", "mean"),
            Mean_Abs_G=("abs_g_mag", "mean"),
        )
        .sort_values("kmeans_label")
        .round(2)
    )

    if int(hdbscan_summary["Star_Count"].sum()) != row_count:
        raise RuntimeError("HDBSCAN cluster summary does not reconcile to input rows.")
    if int(kmeans_summary["Star_Count"].sum()) != row_count:
        raise RuntimeError("K-Means cluster summary does not reconcile to input rows.")

    # 7. Persist the fitted estimators.
    LOGGER.info("Serializing fitted model artifacts...")
    artifacts = {
        "standard_scaler.joblib": scaler,
        "kmeans_model.joblib": kmeans,
        "umap_reducer.joblib": reducer,
        "hdbscan_model.joblib": clusterer,
    }
    for filename, estimator in artifacts.items():
        joblib.dump(estimator, model_dir / filename, compress=3)

    # 8. Persist model inputs/outputs and diagnostic tables.
    clustered_path = output_dir / "clustered_stars.parquet"
    clustered_data.to_parquet(clustered_path, index=False)

    hdbscan_summary_path = output_dir / "hdbscan_cluster_summary.csv"
    kmeans_summary_path = output_dir / "kmeans_cluster_summary.csv"
    hdbscan_summary.to_csv(hdbscan_summary_path, index=False)
    kmeans_summary.to_csv(kmeans_summary_path, index=False)

    embedding_path = output_dir / "umap_embeddings.npy"
    scaled_path = output_dir / "scaled_features.npy"
    np.save(embedding_path, embedding)
    np.save(scaled_path, scaled_features)

    membership = (
        pd.Series(hdbscan_labels, name="star_count")
        .value_counts()
        .sort_index()
        .rename_axis("hdbscan_label")
        .reset_index()
    )
    membership_path = output_dir / "hdbscan_cluster_membership.csv"
    membership.to_csv(membership_path, index=False)

    # Label names are deliberately generic. They are algorithmic IDs, not
    # astrophysical class labels or independently verified physical identities.
    label_mapping = {
        str(int(label)): (
            "Noise / unassigned" if int(label) == -1 else f"Cluster {int(label)}"
        )
        for label in sorted(np.unique(hdbscan_labels).tolist())
    }
    mapping_path = model_dir / "hdbscan_label_mapping.json"
    save_json(label_mapping, mapping_path)

    # K-Means cluster centers converted back to the original physical units.
    centers_physical = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_), columns=FEATURES
    )
    centers_physical.insert(
        0, "kmeans_label", np.arange(len(centers_physical), dtype=int)
    )
    centers_path = output_dir / "kmeans_centers_physical.csv"
    centers_physical.to_csv(centers_path, index=False)

    # 9. Metrics and configuration are saved separately for API and reporting.
    metrics: dict[str, Any] = {
        "dataset_rows": row_count,
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "kmeans": {
            "cluster_count": int(len(np.unique(kmeans_labels))),
            "silhouette": silhouette_details,
            "parameters": KMEANS_CONFIG,
        },
        "umap": {
            "embedding_dimensions": int(embedding.shape[1]),
            "parameters": UMAP_CONFIG,
        },
        "hdbscan": {
            "cluster_count_excluding_noise": n_hdbscan_clusters,
            "noise_count": n_noise,
            "clustered_count": clustered_count,
            "noise_percentage": noise_percentage,
            "dbcv": dbcv_details,
            "parameters": HDBSCAN_CONFIG,
        },
        "evaluation_notes": [
            "Silhouette is sampled on 10,000 rows in the standardized 6D feature space.",
            "DBCV is sampled on up to 20,000 non-noise rows in the 2D UMAP embedding.",
            "The two metrics use different spaces and sampling procedures; they are not directly comparable.",
            "HDBSCAN label -1 denotes noise/unassigned observations, not a verified astrophysical class.",
        ],
    }
    metrics_path = output_dir / "evaluation_metrics.json"
    save_json(metrics, metrics_path)

    model_metadata: dict[str, Any] = {
        "project": "Gaia DR3 Unsupervised Clustering",
        "training_script_version": SCRIPT_VERSION,
        "trained_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "random_seed": RANDOM_SEED,
        "input": {
            "path": str(input_path),
            "sha256": source_sha256,
            "rows": row_count,
            "feature_columns_in_order": FEATURES,
        },
        "models": {
            "standard_scaler": SCALER_CONFIG,
            "kmeans": KMEANS_CONFIG,
            "umap": UMAP_CONFIG,
            "hdbscan": HDBSCAN_CONFIG,
        },
        "package_versions": {
            package: package_version(package)
            for package in [
                "numpy", "pandas", "scikit-learn", "umap-learn",
                "hdbscan", "joblib", "numba", "pynndescent", "pyarrow",
            ]
        },
        "prediction_support": {
            "hdbscan_prediction_data_generated": True,
            "method": "hdbscan.approximate_predict",
            "note": "Approximate assignment to existing fitted clusters; does not discover new clusters.",
        },
        "outputs": {
            "clustered_catalogue": str(clustered_path),
            "hdbscan_cluster_summary": str(hdbscan_summary_path),
            "kmeans_cluster_summary": str(kmeans_summary_path),
            "umap_embeddings": str(embedding_path),
            "scaled_features": str(scaled_path),
            "evaluation_metrics": str(metrics_path),
        },
    }
    metadata_path = model_dir / "model_metadata.json"
    save_json(model_metadata, metadata_path)

    LOGGER.info("Training and artifact export completed successfully.")
    LOGGER.info("Models saved to: %s", model_dir)
    LOGGER.info("Outputs saved to: %s", output_dir)

    print("\n" + "=" * 64)
    print("GAIA DR3 MODEL TRAINING SUMMARY")
    print("=" * 64)
    print(f"Observations:             {row_count:,}")
    print(f"K-Means clusters:         {len(np.unique(kmeans_labels))}")
    print(f"K-Means silhouette:       {silhouette_value:.4f}")
    print(f"UMAP embedding:           {embedding.shape}")
    print(f"HDBSCAN clusters:         {n_hdbscan_clusters}")
    print(f"HDBSCAN noise points:     {n_noise:,}")
    print(f"HDBSCAN noise percentage: {noise_percentage:.2f}%")
    print(
        "DBCV (sampled non-noise UMAP): "
        + (f"{dbcv_value:.4f}" if dbcv_value is not None else "not available")
    )
    print(f"\nModel directory: {model_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 64)

    return {
        "metrics": metrics,
        "model_metadata": model_metadata,
        "model_dir": model_dir,
        "output_dir": output_dir,
    }


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Train the project's StandardScaler, K-Means, UMAP and HDBSCAN "
            "models using the exact settings from the latest notebook."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "DATA" / "processed" / "gaia_engineered.parquet",
        help="Engineered Gaia Parquet input.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=project_root / "MODEL",
        help="Directory for serialized estimators and model metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "OUTPUTS",
        help="Directory for predictions, embeddings, metrics and summaries.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run_training(args.input, args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
