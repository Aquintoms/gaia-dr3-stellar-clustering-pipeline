"""Verify the saved Gaia clustering artifacts without retraining any model.

Run from the project root with the project's .venv active:
    python SCRIPTS/verify_artifacts.py

This script only loads existing artifacts and performs inference on a small
sample from OUTPUTS/clustered_stars.parquet. It does not modify project files.
"""
from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import hdbscan
import umap  # noqa: F401 - importing confirms umap-learn is available


FEATURES = [
    "x",
    "y",
    "z",
    "tangential_velocity",
    "abs_g_mag",
    "bp_rp",
]
SAMPLE_SIZE = 10


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    model_dir = project_root / "MODEL"
    output_dir = project_root / "OUTPUTS"

    artifact_paths = {
        "metadata": model_dir / "model_metadata.json",
        "label mapping": model_dir / "hdbscan_label_mapping.json",
        "scaler": model_dir / "standard_scaler.joblib",
        "K-Means model": model_dir / "kmeans_model.joblib",
        "UMAP reducer": model_dir / "umap_reducer.joblib",
        "HDBSCAN model": model_dir / "hdbscan_model.joblib",
        "clustered catalogue": output_dir / "clustered_stars.parquet",
        "saved scaled features": output_dir / "scaled_features.npy",
    }

    missing = [f"{name}: {path}" for name, path in artifact_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required project artifacts are missing:\n  - " + "\n  - ".join(missing)
        )

    print("GAIA MODEL ARTIFACT VERIFICATION")
    print("=" * 64)
    print(f"Project root: {project_root}")
    print(f"Python: {platform.python_version()} ({sys.executable})")

    metadata = json.loads(artifact_paths["metadata"].read_text(encoding="utf-8"))
    recorded_features = metadata.get("input", {}).get("feature_columns_in_order")
    if recorded_features != FEATURES:
        raise RuntimeError(
            "Feature order in model_metadata.json does not match the verification script: "
            f"{recorded_features!r}"
        )
    expected_versions = metadata.get("package_versions", {})
    distributions = ["numpy", "pandas", "scikit-learn", "joblib", "umap-learn", "hdbscan"]
    mismatches = []
    print("\nPackage versions (training metadata vs current environment):")
    for dist in distributions:
        expected = expected_versions.get(dist, "not-recorded")
        actual = package_version(dist)
        print(f"  {dist}: {expected} vs {actual}")
        if expected not in ("not-recorded", actual):
            mismatches.append(f"{dist}: trained with {expected}, currently {actual}")

    if mismatches:
        print("\nWARNING: package-version differences detected:")
        for mismatch in mismatches:
            print(f"  - {mismatch}")
        print("Continue interpreting results cautiously; do not retrain as part of this check.")

    print("\nLoading serialized estimators...")
    scaler = joblib.load(artifact_paths["scaler"])
    kmeans = joblib.load(artifact_paths["K-Means model"])
    reducer = joblib.load(artifact_paths["UMAP reducer"])
    hdbscan_model = joblib.load(artifact_paths["HDBSCAN model"])
    label_mapping = json.loads(artifact_paths["label mapping"].read_text(encoding="utf-8"))

    if getattr(scaler, "n_features_in_", None) != len(FEATURES):
        raise RuntimeError(
            f"Scaler expects {getattr(scaler, 'n_features_in_', None)} features; "
            f"expected {len(FEATURES)}."
        )
    if getattr(kmeans, "n_features_in_", None) != len(FEATURES):
        raise RuntimeError(
            f"K-Means expects {getattr(kmeans, 'n_features_in_', None)} features; "
            f"expected {len(FEATURES)}."
        )
    if getattr(kmeans, "n_clusters", None) != 4:
        raise RuntimeError(f"Expected four K-Means clusters, found {getattr(kmeans, 'n_clusters', None)}.")

    prediction_data = getattr(hdbscan_model, "prediction_data_", None)
    if prediction_data is None:
        raise RuntimeError(
            "HDBSCAN prediction_data_ is missing. approximate_predict cannot be used "
            "until prediction data is generated and the model artifact is re-exported."
        )

    print("  StandardScaler: loaded; six input features")
    print("  K-Means: loaded; four clusters; six input features")
    print("  UMAP: loaded")
    print("  HDBSCAN: loaded; prediction data is present")

    catalogue = pd.read_parquet(artifact_paths["clustered catalogue"])
    required_columns = ["source_id", *FEATURES, "kmeans_label"]
    absent_columns = [col for col in required_columns if col not in catalogue.columns]
    if absent_columns:
        raise RuntimeError("Clustered catalogue is missing columns: " + ", ".join(absent_columns))
    if len(catalogue) == 0:
        raise RuntimeError("Clustered catalogue is empty.")
    expected_rows = metadata.get("input", {}).get("rows")
    if expected_rows is not None and len(catalogue) != expected_rows:
        raise RuntimeError(
            f"Clustered catalogue has {len(catalogue)} rows; metadata records {expected_rows}."
        )

    sample = catalogue.loc[:, FEATURES].head(SAMPLE_SIZE)
    raw_features = sample.to_numpy(dtype=np.float64)
    if not np.isfinite(raw_features).all():
        raise RuntimeError("The selected sample contains NaN or infinite feature values.")

    scaled_features = scaler.transform(raw_features)
    if scaled_features.shape != (len(sample), len(FEATURES)):
        raise RuntimeError(f"Unexpected scaled shape: {scaled_features.shape}")

    # Confirm that the saved scaler reproduces the training-time scaled values.
    saved_scaled = np.load(artifact_paths["saved scaled features"], mmap_mode="r")
    if saved_scaled.ndim != 2 or saved_scaled.shape[1] != len(FEATURES):
        raise RuntimeError(f"Unexpected saved scaled-features shape: {saved_scaled.shape}")
    if saved_scaled.shape[0] < len(sample):
        raise RuntimeError("Saved scaled-features array is shorter than the verification sample.")
    max_scaled_difference = float(np.max(np.abs(scaled_features - saved_scaled[: len(sample)])))
    if not np.allclose(scaled_features, saved_scaled[: len(sample)], rtol=1e-7, atol=1e-9):
        raise RuntimeError(
            "Scaler output differs from the saved training array; "
            f"maximum absolute difference: {max_scaled_difference:.3g}"
        )

    kmeans_predictions = kmeans.predict(scaled_features)
    saved_kmeans_labels = catalogue["kmeans_label"].head(len(sample)).to_numpy(dtype=int)
    if not np.array_equal(kmeans_predictions, saved_kmeans_labels):
        raise RuntimeError(
            "K-Means predictions for the training sample do not match the labels "
            "stored in clustered_stars.parquet."
        )

    umap_embedding = reducer.transform(scaled_features)
    if umap_embedding.shape != (len(sample), 2):
        raise RuntimeError(f"Unexpected UMAP transform shape: {umap_embedding.shape}")
    if not np.isfinite(umap_embedding).all():
        raise RuntimeError("UMAP transform produced non-finite coordinates.")

    hdbscan_predictions, hdbscan_strengths = hdbscan.approximate_predict(
        hdbscan_model, umap_embedding
    )
    if len(hdbscan_predictions) != len(sample) or len(hdbscan_strengths) != len(sample):
        raise RuntimeError("HDBSCAN approximate prediction returned an unexpected result length.")
    if not np.isfinite(hdbscan_strengths).all():
        raise RuntimeError("HDBSCAN prediction strengths contain non-finite values.")
    if np.any(hdbscan_strengths < 0) or np.any(hdbscan_strengths > 1):
        raise RuntimeError("HDBSCAN prediction strengths fall outside [0, 1].")

    unmapped = sorted({str(int(label)) for label in hdbscan_predictions} - set(label_mapping))
    if unmapped:
        raise RuntimeError("HDBSCAN predicted labels absent from label mapping: " + ", ".join(unmapped))

    print("\nInference checks:")
    print(f"  Sample rows: {len(sample)}")
    print(f"  Maximum scaler difference vs saved training array: {max_scaled_difference:.3g}")
    print("  K-Means predictions match saved training labels: YES")
    print(f"  UMAP transform shape: {umap_embedding.shape}")
    print("  HDBSCAN approximate prediction: SUCCESS")
    print("\nSample predictions:")
    for i, source_id in enumerate(catalogue["source_id"].head(len(sample)).tolist()):
        label = int(hdbscan_predictions[i])
        description = label_mapping[str(label)]
        print(
            f"  source_id={source_id} | K-Means={int(kmeans_predictions[i])} | "
            f"HDBSCAN={label} ({description}) | strength={float(hdbscan_strengths[i]):.4f}"
        )

    print("\n" + "=" * 64)
    print("PASS: Saved artifacts loaded and the sample inference path completed.")
    print("No models were fitted or retrained; no project artifacts were modified.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
