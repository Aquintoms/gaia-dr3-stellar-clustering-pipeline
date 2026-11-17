"""Gaia DR3 ingestion, quality filtering, and feature engineering.

This script reproduces the data-preparation steps currently present in
Gaia_clustering_project.ipynb. It deliberately does NOT fit StandardScaler,
K-Means, UMAP, or HDBSCAN; those belong to later pipeline stages.

Default usage from the project root:
    python SCRIPTS/data_pipeline.py

Optional:
    python SCRIPTS/data_pipeline.py --input DATA/gaia_subset_raw.csv
    python SCRIPTS/data_pipeline.py --output-dir DATA/processed
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord


LOGGER = logging.getLogger("gaia_data_pipeline")
PIPELINE_VERSION = "1.0.0"

# Preserving this exact feature order throughout scaling and modelling.
FEATURES = [
    "x",
    "y",
    "z",
    "tangential_velocity",
    "abs_g_mag",
    "bp_rp",
]

REQUIRED_INPUT_COLUMNS = [
    "source_id",
    "ra",
    "dec",
    "parallax",
    "parallax_error",
    "pmra",
    "pmdec",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "ruwe",
]

# The BP/RP excess factor is recorded as a diagnostic, but is NOT filtered,
# matching the current notebook's actual processing.
QUALITY_CUTS = {
    "ruwe": "ruwe < 1.4",
    "parallax": "parallax > 0",
    "parallax_error": "parallax_error > 0",
    "parallax_snr": "parallax / parallax_error > 5",
}


def sha256_file(path: Path) -> str:
    """Return a SHA256 digest for a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_package_version(package_name: str) -> str:
    """Return an installed package version, or a clear fallback label."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def validate_input_schema(df: pd.DataFrame) -> None:
    """Fail early if the Gaia input does not have the notebook's columns."""
    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "The input CSV is missing required Gaia columns: " + ", ".join(missing)
        )

    if df.empty:
        raise ValueError("The input Gaia CSV contains no observations.")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the derived quantities used by the current notebook."""
    result = df.copy()

    # Same derived observables created in the notebook's feature diagnostics.
    result["bp_rp"] = (
        result["phot_bp_mean_mag"] - result["phot_rp_mean_mag"]
    )

    # Parallax signal-to-noise ratio. Division by zero is excluded by the
    # quality mask below; suppressing warnings here, then validate retained rows.
    with np.errstate(divide="ignore", invalid="ignore"):
        result["parallax_snr"] = (
            result["parallax"] / result["parallax_error"]
        )

    before_count = int(len(result))

    # Counting individual cut failures for the run report. The final mask is the
    # intersection of these criteria, as in the current notebook.
    cut_checks = {
        "ruwe_lt_1_4": result["ruwe"].notna() & (result["ruwe"] < 1.4),
        "parallax_gt_0": result["parallax"].notna() & (result["parallax"] > 0),
        "parallax_error_gt_0": (
            result["parallax_error"].notna() & (result["parallax_error"] > 0)
        ),
        "parallax_snr_gt_5": (
            result["parallax_snr"].notna() & (result["parallax_snr"] > 5)
        ),
    }

    quality_mask = pd.Series(True, index=result.index)
    for check in cut_checks.values():
        quality_mask &= check

    LOGGER.info("Rows before quality filtering: %s", before_count)
    for check_name, check_values in cut_checks.items():
        LOGGER.info("Rows passing %s: %s", check_name, int(check_values.sum()))

    result = result.loc[quality_mask].copy()
    after_count = int(len(result))

    if result.empty:
        raise ValueError(
            "No observations remain after applying the notebook's quality cuts."
        )

    # Confirming the exact retained-data conditions before feature engineering.
    assert (result["ruwe"] < 1.4).all()
    assert (result["parallax"] > 0).all()
    assert (result["parallax_error"] > 0).all()
    assert (result["parallax_snr"] > 5).all()

    # The current notebook does not apply a BP/RP excess-factor cut. 
    # Recording this diagnostic without silently changing the modelling population.
    excess_col = "phot_bp_rp_excess_factor"
    excess_factor_in_range = None
    if excess_col in result.columns:
        excess_factor_in_range = int(
            (
                (result[excess_col] > 0.8)
                & (result[excess_col] < 1.4)
            ).sum()
        )

    # Every source column required for the derived physical quantities must
    # be finite. Stopping rather than silently dropping data and changing the
    # experiment's sample size.
    feature_source_columns = [
        "ra",
        "dec",
        "parallax",
        "pmra",
        "pmdec",
        "phot_g_mean_mag",
        "phot_bp_mean_mag",
        "phot_rp_mean_mag",
    ]
    numeric_sources = result[feature_source_columns].to_numpy(dtype=float)
    invalid_source_mask = ~np.isfinite(numeric_sources).all(axis=1)
    if invalid_source_mask.any():
        bad_count = int(invalid_source_mask.sum())
        raise ValueError(
            f"{bad_count} retained rows contain missing or non-finite values "
            "needed for feature engineering. No rows were silently removed. "
            "Inspect the source CSV and document any additional cleaning rule "
            "before proceeding."
        )

    # Feature engineering: these formulas and units match the latest notebook.
    # Distance (pc) from parallax in milliarcseconds.
    result["distance_pc"] = 1000.0 / result["parallax"]

    # Absolute G-band magnitude.
    result["abs_g_mag"] = (
        result["phot_g_mean_mag"]
        + 5.0
        - 5.0 * np.log10(result["distance_pc"])
    )

    # Total proper motion (mas/year).
    result["pm_total"] = np.sqrt(
        result["pmra"] ** 2 + result["pmdec"] ** 2
    )

    # Tangential velocity (km/s); proper motion is converted from mas/year
    # to arcseconds/year by dividing by 1000.
    result["tangential_velocity"] = (
        4.74
        * (result["pm_total"] / 1000.0)
        * result["distance_pc"]
    )

    # Convert sky coordinates to heliocentric Cartesian coordinates (pc).
    sky_coordinates = SkyCoord(
        ra=result["ra"].to_numpy(dtype=float) * u.degree,
        dec=result["dec"].to_numpy(dtype=float) * u.degree,
        distance=result["distance_pc"].to_numpy(dtype=float) * u.pc,
        frame="icrs",
    )
    result["x"] = sky_coordinates.cartesian.x.to_value(u.pc)
    result["y"] = sky_coordinates.cartesian.y.to_value(u.pc)
    result["z"] = sky_coordinates.cartesian.z.to_value(u.pc)

    # Check the six model features are all finite before they are handed to
    # the scaler in the next pipeline stage.
    feature_values = result[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(feature_values).all():
        raise ValueError(
            "Non-finite values were generated in the six model features. "
            "Inspect the data and feature calculations before model fitting."
        )

    LOGGER.info("Rows after quality filtering: %s", after_count)
    LOGGER.info("Rows removed by quality filtering: %s", before_count - after_count)
    LOGGER.info("BP/RP excess factor in (0.8, 1.4): %s", excess_factor_in_range)

    # Store run diagnostics in DataFrame attrs for the manifest builder.
    result.attrs["pipeline_diagnostics"] = {
        "rows_before_filtering": before_count,
        "rows_after_filtering": after_count,
        "rows_removed": before_count - after_count,
        "individual_cut_pass_counts": {
            key: int(value.sum()) for key, value in cut_checks.items()
        },
        "phot_bp_rp_excess_factor_0_8_1_4_count_not_filtered": excess_factor_in_range,
    }

    return result


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write Parquet with an actionable error if no engine is installed."""
    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:
        raise RuntimeError(
            "Parquet output requires pyarrow or fastparquet. Install the project "
            "dependencies in the active .venv, for example: "
            "python -m pip install pyarrow"
        ) from exc


def run_pipeline(input_path: Path, output_dir: Path) -> dict[str, Any]:
    """Run ingestion, quality filtering, and feature engineering."""
    input_path = input_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Gaia input CSV not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Reading Gaia DR3 CSV: %s", input_path)
    raw_df = pd.read_csv(input_path)
    validate_input_schema(raw_df)

    input_rows = int(len(raw_df))
    source_sha256 = sha256_file(input_path)

    engineered_df = engineer_features(raw_df)
    diagnostics = engineered_df.attrs.get("pipeline_diagnostics", {})

    # Use two explicit artifacts so the raw/processed boundary is visible.
    # The quality-filtered file retains original Gaia columns plus bp_rp and
    # parallax_snr. The engineered file additionally contains derived features.
    quality_path = output_dir / "gaia_quality_filtered.parquet"
    engineered_path = output_dir / "gaia_engineered.parquet"

    quality_columns = [
        column for column in engineered_df.columns
        if column not in {
            "distance_pc",
            "abs_g_mag",
            "pm_total",
            "tangential_velocity",
            "x",
            "y",
            "z",
        }
    ]
    quality_df = engineered_df[quality_columns].copy()

    write_parquet(quality_df, quality_path)
    write_parquet(engineered_df, engineered_path)

    manifest: dict[str, Any] = {
        "pipeline_name": "Gaia DR3 data preparation pipeline",
        "pipeline_version": PIPELINE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "input": {
            "path": str(input_path),
            "filename": input_path.name,
            "sha256": source_sha256,
            "rows": input_rows,
            "columns": list(raw_df.columns),
        },
        "quality_filtering": {
            "rules": QUALITY_CUTS,
            **diagnostics,
        },
        "output": {
            "quality_filtered_parquet": str(quality_path),
            "engineered_parquet": str(engineered_path),
            "rows": int(len(engineered_df)),
            "feature_columns_in_order": FEATURES,
        },
        "package_versions": {
            package: get_package_version(package)
            for package in ["numpy", "pandas", "astropy", "pyarrow"]
        },
    }

    manifest_path = output_dir / "data_pipeline_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )

    LOGGER.info("Data preparation completed successfully.")
    LOGGER.info("Engineered rows: %s", len(engineered_df))
    LOGGER.info("Six model features: %s", FEATURES)
    LOGGER.info("Engineered data: %s", engineered_path)
    LOGGER.info("Manifest: %s", manifest_path)

    return manifest


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Gaia DR3 data using the quality filters and feature "
            "engineering currently implemented in the project notebook."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "DATA" / "gaia_subset_raw.csv",
        help="Path to the raw Gaia DR3 CSV (default: DATA/gaia_subset_raw.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "DATA" / "processed",
        help="Directory for processed Parquet files and manifest.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run_pipeline(args.input, args.output_dir)


if __name__ == "__main__":
    main()
