"""FastAPI service for the Gaia DR3 clustering project.

The service has two distinct responsibilities:
1. Query the catalogue whose K-Means and HDBSCAN labels were produced by the
   completed training pipeline.
2. Assign new, already-engineered six-feature vectors to the existing fitted
   K-Means and HDBSCAN clusters. It does not retrain or discover new clusters.

Run from the project root with:
    python -m uvicorn main:app --reload
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field


LOGGER = logging.getLogger("gaia_api")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

FEATURE_ORDER = [
    "x",
    "y",
    "z",
    "tangential_velocity",
    "abs_g_mag",
    "bp_rp",
]

LABEL_COLUMNS = {
    "kmeans": "kmeans_label",
    "hdbscan": "hdbscan_label",
}

# Resolve the actual uppercase directory names used by this repository.
PROJECT_ROOT = Path(
    os.getenv("GAIA_PROJECT_ROOT", str(Path(__file__).resolve().parent))
).expanduser().resolve()
MODEL_DIR = Path(os.getenv("GAIA_MODEL_DIR", "MODEL"))
OUTPUT_DIR = Path(os.getenv("GAIA_OUTPUT_DIR", "OUTPUTS"))
if not MODEL_DIR.is_absolute():
    MODEL_DIR = (PROJECT_ROOT / MODEL_DIR).resolve()
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = (PROJECT_ROOT / OUTPUT_DIR).resolve()

MODEL_FILES = {
    "scaler": "standard_scaler.joblib",
    "kmeans": "kmeans_model.joblib",
    "umap": "umap_reducer.joblib",
    "hdbscan": "hdbscan_model.joblib",
}
METADATA_FILE = "model_metadata.json"
CATALOGUE_FILE = "clustered_stars.parquet"
LABEL_MAPPING_FILE = "hdbscan_label_mapping.json"

NUMERIC_FILTERS = {
    "ra": ("ra_min", "ra_max"),
    "dec": ("dec_min", "dec_max"),
    "bp_rp": ("bp_rp_min", "bp_rp_max"),
    "abs_g_mag": ("abs_g_mag_min", "abs_g_mag_max"),
    "tangential_velocity": (
        "tangential_velocity_min",
        "tangential_velocity_max",
    ),
    "distance_pc": ("distance_pc_min", "distance_pc_max"),
    "x": ("x_min", "x_max"),
    "y": ("y_min", "y_max"),
    "z": ("z_min", "z_max"),
}

DEFAULT_STAR_FIELDS = [
    "source_id",
    "ra",
    "dec",
    "parallax",
    "ruwe",
    "x",
    "y",
    "z",
    "distance_pc",
    "tangential_velocity",
    "abs_g_mag",
    "bp_rp",
    "kmeans_label",
    "hdbscan_label",
    "umap_1",
    "umap_2",
]


class GaiaFeatureVector(BaseModel):
    """One observation expressed in the six features used during training."""

    model_config = ConfigDict(extra="forbid")

    source_id: str | None = Field(
        default=None,
        description="Optional Gaia DR3 source identifier; returned as a string.",
    )
    x: float = Field(allow_inf_nan=False, description="Cartesian X coordinate (pc).")
    y: float = Field(allow_inf_nan=False, description="Cartesian Y coordinate (pc).")
    z: float = Field(allow_inf_nan=False, description="Cartesian Z coordinate (pc).")
    tangential_velocity: float = Field(
        allow_inf_nan=False, description="Tangential velocity (km/s)."
    )
    abs_g_mag: float = Field(
        allow_inf_nan=False, description="Absolute Gaia G-band magnitude."
    )
    bp_rp: float = Field(allow_inf_nan=False, description="Gaia BP-RP colour index.")


class PredictionRequest(BaseModel):
    """A bounded batch of feature vectors for inference."""

    model_config = ConfigDict(extra="forbid")

    stars: list[GaiaFeatureVector] = Field(min_length=1, max_length=2000)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_catalogue(frame: pd.DataFrame) -> None:
    required = {"source_id", *FEATURE_ORDER, "kmeans_label", "hdbscan_label"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "The clustered catalogue is missing required columns: "
            + ", ".join(missing)
        )
    if frame.empty:
        raise ValueError("The clustered catalogue contains no rows.")
    if frame["source_id"].isna().any():
        raise ValueError("The clustered catalogue contains missing source_id values.")
    if frame[["kmeans_label", "hdbscan_label"]].isna().any().any():
        raise ValueError("The clustered catalogue contains missing cluster labels.")


def _load_runtime() -> dict[str, Any]:
    """Load and validate the saved catalogue, metadata and fitted estimators."""
    catalogue_path = OUTPUT_DIR / CATALOGUE_FILE
    metadata_path = MODEL_DIR / METADATA_FILE

    if not catalogue_path.is_file():
        raise FileNotFoundError(f"Clustered catalogue not found: {catalogue_path}")

    catalogue = pd.read_parquet(catalogue_path)
    _validate_catalogue(catalogue)
    metadata = _load_json(metadata_path)

    metadata_features = metadata.get("input", {}).get("feature_columns_in_order")
    if metadata_features != FEATURE_ORDER:
        raise ValueError(
            "Model metadata feature order does not match the API feature order. "
            f"Metadata={metadata_features!r}; API={FEATURE_ORDER!r}"
        )

    missing_models = [
        str(MODEL_DIR / filename)
        for filename in MODEL_FILES.values()
        if not (MODEL_DIR / filename).is_file()
    ]
    if missing_models:
        raise FileNotFoundError(
            "Missing serialized model artifact(s): " + ", ".join(missing_models)
        )

    estimators = {
        key: joblib.load(MODEL_DIR / filename)
        for key, filename in MODEL_FILES.items()
    }

    scaler_features = getattr(estimators["scaler"], "n_features_in_", None)
    kmeans_features = getattr(estimators["kmeans"], "n_features_in_", None)
    if scaler_features != len(FEATURE_ORDER):
        raise ValueError(f"Scaler expects {scaler_features} features, not six.")
    if kmeans_features != len(FEATURE_ORDER):
        raise ValueError(f"K-Means expects {kmeans_features} features, not six.")
    if getattr(estimators["hdbscan"], "prediction_data_", None) is None:
        raise ValueError("HDBSCAN prediction data is absent from the saved model.")

    label_mapping_path = MODEL_DIR / LABEL_MAPPING_FILE
    label_mapping = (
        _load_json(label_mapping_path) if label_mapping_path.is_file() else {}
    )

    return {
        "catalogue": catalogue,
        "metadata": metadata,
        "label_mapping": label_mapping,
        **estimators,
    }


def _json_value(value: Any) -> Any:
    """Convert NumPy/Pandas scalar values to JSON-safe Python values."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        record = {str(key): _json_value(value) for key, value in row.items()}
        if record.get("source_id") is not None:
            # Gaia source IDs are 64-bit integers; strings avoid precision loss
            # in JavaScript/JSON consumers.
            record["source_id"] = str(record["source_id"])
        records.append(record)
    return records


def _build_cluster_summaries(frame: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    summaries: dict[str, list[dict[str, Any]]] = {}
    numeric_summary_columns = [
        column
        for column in [
            "x",
            "y",
            "z",
            "distance_pc",
            "tangential_velocity",
            "abs_g_mag",
            "bp_rp",
        ]
        if column in frame.columns
    ]

    for algorithm, label_column in LABEL_COLUMNS.items():
        items: list[dict[str, Any]] = []
        for cluster_id, group in frame.groupby(label_column, sort=False):
            cluster_id = int(cluster_id)
            item: dict[str, Any] = {
                "cluster_id": cluster_id,
                "label": (
                    "Noise / unassigned"
                    if algorithm == "hdbscan" and cluster_id == -1
                    else f"Cluster {cluster_id}"
                ),
                "is_noise": bool(algorithm == "hdbscan" and cluster_id == -1),
                "star_count": int(len(group)),
                "catalogue_fraction": float(len(group) / len(frame)),
            }
            for column in numeric_summary_columns:
                values = pd.to_numeric(group[column], errors="coerce")
                if values.notna().any():
                    item[f"mean_{column}"] = _json_value(values.mean())
                    if column == "tangential_velocity":
                        item["median_tangential_velocity"] = _json_value(
                            values.median()
                        )
            items.append(item)
        summaries[algorithm] = sorted(
            items, key=lambda item: item["star_count"], reverse=True
        )

    return summaries


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy, read-only resources once when the service starts."""
    LOGGER.info("Loading Gaia API resources from %s", PROJECT_ROOT)
    runtime = _load_runtime()
    runtime["cluster_summaries"] = _build_cluster_summaries(runtime["catalogue"])
    app.state.runtime = runtime
    LOGGER.info(
        "Gaia API ready: %s catalogue rows; %s HDBSCAN labels including noise.",
        len(runtime["catalogue"]),
        len(runtime["cluster_summaries"]["hdbscan"]),
    )
    yield


app = FastAPI(
    title="Gaia DR3 Stellar Clustering API",
    description=(
        "Query the saved, pre-clustered Gaia DR3 catalogue and assign new, "
        "already-engineered six-feature observations to existing fitted clusters. "
        "The API does not retrain models or discover new clusters."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _runtime(request: Request) -> dict[str, Any]:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="The model service is not ready.")
    return runtime


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    return {
        "service": "Gaia DR3 Stellar Clustering API",
        "api_version": app.version,
        "documentation": "/docs",
        "health": "/health",
        "api_base": "/api/v1",
    }


@app.get("/health", tags=["system"])
def health(request: Request) -> dict[str, Any]:
    runtime = _runtime(request)
    return {
        "status": "ok",
        "catalogue_loaded": True,
        "catalogue_rows": int(len(runtime["catalogue"])),
        "models_loaded": ["scaler", "kmeans", "umap", "hdbscan"],
        "hdbscan_prediction_data_available": True,
    }


@app.get("/api/v1/summary", tags=["catalogue"])
def catalogue_summary(request: Request) -> dict[str, Any]:
    runtime = _runtime(request)
    frame: pd.DataFrame = runtime["catalogue"]
    kmeans_counts = frame["kmeans_label"].value_counts().sort_index()
    hdbscan_counts = frame["hdbscan_label"].value_counts().sort_index()
    noise_count = int(hdbscan_counts.get(-1, 0))
    metadata_input = runtime["metadata"].get("input", {})

    return {
        "catalogue": "Gaia DR3",
        "total_observations": int(len(frame)),
        "feature_count": len(FEATURE_ORDER),
        "features_in_order": FEATURE_ORDER,
        "dataset_sha256": metadata_input.get("sha256"),
        "kmeans": {
            "cluster_count": int(frame["kmeans_label"].nunique()),
            "cluster_sizes": {
                str(int(label)): int(count) for label, count in kmeans_counts.items()
            },
        },
        "hdbscan": {
            "cluster_count_excluding_noise": int(
                hdbscan_counts.index[hdbscan_counts.index != -1].nunique()
            ),
            "noise_count": noise_count,
            "noise_percentage": float(100.0 * noise_count / len(frame)),
            "cluster_sizes": {
                str(int(label)): int(count) for label, count in hdbscan_counts.items()
            },
        },
    }


@app.get("/api/v1/model-info", tags=["models"])
def model_info(request: Request) -> dict[str, Any]:
    runtime = _runtime(request)
    metadata = runtime["metadata"]
    # Do not return machine-specific absolute file paths from the training PC.
    return {
        "project": metadata.get("project"),
        "training_script_version": metadata.get("training_script_version"),
        "trained_utc": metadata.get("trained_utc"),
        "python_version": metadata.get("python_version"),
        "random_seed": metadata.get("random_seed"),
        "feature_columns_in_order": FEATURE_ORDER,
        "dataset_rows": metadata.get("input", {}).get("rows"),
        "models": metadata.get("models", {}),
        "package_versions_at_training": metadata.get("package_versions", {}),
        "prediction_support": metadata.get("prediction_support", {}),
        "loaded_models": list(MODEL_FILES.keys()),
    }


@app.get("/api/v1/clusters", tags=["clusters"])
def list_clusters(
    request: Request,
    algorithm: Literal["hdbscan", "kmeans"] = Query(default="hdbscan"),
    include_noise: bool = Query(default=True),
) -> dict[str, Any]:
    runtime = _runtime(request)
    clusters = runtime["cluster_summaries"][algorithm]
    if algorithm == "hdbscan" and not include_noise:
        clusters = [cluster for cluster in clusters if not cluster["is_noise"]]
    return {"algorithm": algorithm, "count": len(clusters), "clusters": clusters}


@app.get("/api/v1/clusters/{cluster_id}", tags=["clusters"])
def get_cluster(
    cluster_id: int,
    request: Request,
    algorithm: Literal["hdbscan", "kmeans"] = Query(default="hdbscan"),
) -> dict[str, Any]:
    runtime = _runtime(request)
    for cluster in runtime["cluster_summaries"][algorithm]:
        if cluster["cluster_id"] == cluster_id:
            return {"algorithm": algorithm, **cluster}
    raise HTTPException(
        status_code=404,
        detail=f"No {algorithm} cluster with ID {cluster_id} exists in this catalogue.",
    )


@app.get("/api/v1/stars", tags=["stars"])
def query_stars(
    request: Request,
    algorithm: Literal["hdbscan", "kmeans"] = Query(default="hdbscan"),
    cluster_id: int | None = Query(default=None),
    include_noise: bool = Query(default=True),
    ra_min: float | None = None,
    ra_max: float | None = None,
    dec_min: float | None = None,
    dec_max: float | None = None,
    bp_rp_min: float | None = None,
    bp_rp_max: float | None = None,
    abs_g_mag_min: float | None = None,
    abs_g_mag_max: float | None = None,
    tangential_velocity_min: float | None = None,
    tangential_velocity_max: float | None = None,
    distance_pc_min: float | None = None,
    distance_pc_max: float | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
    sort_by: str | None = None,
    sort_order: Literal["asc", "desc"] = Query(default="asc"),
    fields: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, ge=1, le=5000),
) -> dict[str, Any]:
    runtime = _runtime(request)
    frame: pd.DataFrame = runtime["catalogue"]
    label_column = LABEL_COLUMNS[algorithm]

    selected_mask = pd.Series(True, index=frame.index)
    if cluster_id is not None:
        selected_mask &= pd.to_numeric(frame[label_column], errors="coerce") == cluster_id
    elif algorithm == "hdbscan" and not include_noise:
        selected_mask &= pd.to_numeric(frame[label_column], errors="coerce") != -1

    provided_filters = {
        "ra": (ra_min, ra_max),
        "dec": (dec_min, dec_max),
        "bp_rp": (bp_rp_min, bp_rp_max),
        "abs_g_mag": (abs_g_mag_min, abs_g_mag_max),
        "tangential_velocity": (tangential_velocity_min, tangential_velocity_max),
        "distance_pc": (distance_pc_min, distance_pc_max),
        "x": (x_min, x_max),
        "y": (y_min, y_max),
        "z": (z_min, z_max),
    }

    applied_filters: dict[str, Any] = {"algorithm": algorithm}
    for column, (minimum, maximum) in provided_filters.items():
        if minimum is None and maximum is None:
            continue
        if column not in frame.columns:
            raise HTTPException(
                status_code=400,
                detail=f"The saved catalogue does not contain filter column '{column}'.",
            )
        if minimum is not None and maximum is not None and minimum > maximum:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid range for {column}: minimum exceeds maximum.",
            )
        values = pd.to_numeric(frame[column], errors="coerce")
        if minimum is not None:
            selected_mask &= values >= minimum
            applied_filters[f"{column}_min"] = minimum
        if maximum is not None:
            selected_mask &= values <= maximum
            applied_filters[f"{column}_max"] = maximum

    if cluster_id is not None:
        applied_filters["cluster_id"] = cluster_id
    elif algorithm == "hdbscan":
        applied_filters["include_noise"] = include_noise

    filtered = frame.loc[selected_mask]
    total = int(len(filtered))

    allowed_sort_fields = (
        set(NUMERIC_FILTERS)
        | set(LABEL_COLUMNS.values())
        | {"source_id", "umap_1", "umap_2"}
    )
    effective_sort = sort_by or "source_id"
    if effective_sort not in allowed_sort_fields or effective_sort not in filtered.columns:
        raise HTTPException(
            status_code=400,
            detail=f"sort_by must be one of the available fields: {sorted(allowed_sort_fields)}",
        )

    filtered = filtered.sort_values(
        by=effective_sort,
        ascending=(sort_order == "asc"),
        kind="mergesort",
        na_position="last",
    )
    start = (page - 1) * page_size
    page_frame = filtered.iloc[start : start + page_size]

    if fields:
        requested_fields = [field.strip() for field in fields.split(",") if field.strip()]
        if not requested_fields:
            raise HTTPException(status_code=400, detail="The fields parameter is empty.")
        unknown_fields = sorted(set(requested_fields).difference(frame.columns))
        if unknown_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown response field(s): {unknown_fields}",
            )
        page_frame = page_frame.loc[:, requested_fields]
    else:
        selected_fields = [
            column for column in DEFAULT_STAR_FIELDS if column in page_frame.columns
        ]
        page_frame = page_frame.loc[:, selected_fields]

    return {
        "algorithm": algorithm,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": int((total + page_size - 1) // page_size),
        "sort_by": effective_sort,
        "sort_order": sort_order,
        "filters": applied_filters,
        "items": _records(page_frame),
    }


@app.get("/api/v1/stars/by-id/{source_id}", tags=["stars"])
def get_star_by_source_id(source_id: str, request: Request) -> dict[str, Any]:
    runtime = _runtime(request)
    frame: pd.DataFrame = runtime["catalogue"]
    matches = frame.loc[frame["source_id"].astype(str) == source_id]
    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Gaia source_id {source_id} was not found in the saved catalogue.",
        )
    return {"star": _records(matches.iloc[[0]])[0]}


@app.post("/api/v1/predict", tags=["prediction"])
def predict_new_stars(
    payload: PredictionRequest,
    request: Request,
) -> dict[str, Any]:
    """Assign new feature vectors to existing K-Means/HDBSCAN clusters.

    Inputs must already be engineered into the six physical features and units
    used for training. HDBSCAN uses approximate_predict; it does not refit or
    discover additional clusters.
    """
    runtime = _runtime(request)
    raw_features = np.asarray(
        [[getattr(star, feature) for feature in FEATURE_ORDER] for star in payload.stars],
        dtype=np.float64,
    )
    if not np.isfinite(raw_features).all():
        raise HTTPException(status_code=422, detail="All six features must be finite numbers.")

    try:
        scaled_features = runtime["scaler"].transform(raw_features)
        kmeans_labels = runtime["kmeans"].predict(scaled_features)
        embedding = runtime["umap"].transform(scaled_features)

        import hdbscan

        hdbscan_labels, strengths = hdbscan.approximate_predict(
            runtime["hdbscan"], embedding
        )
    except Exception as exc:
        LOGGER.exception("Prediction failed for supplied feature vectors.")
        raise HTTPException(
            status_code=422,
            detail="The supplied features could not be processed by the saved models.",
        ) from exc

    items = []
    for index, star in enumerate(payload.stars):
        hdb_label = int(hdbscan_labels[index])
        items.append(
            {
                "source_id": star.source_id,
                "kmeans_cluster": int(kmeans_labels[index]),
                "hdbscan_cluster": hdb_label,
                "hdbscan_label": runtime["label_mapping"].get(
                    str(hdb_label),
                    "Noise / unassigned" if hdb_label == -1 else f"Cluster {hdb_label}",
                ),
                "hdbscan_membership_strength": float(strengths[index]),
                "umap_1": float(embedding[index, 0]),
                "umap_2": float(embedding[index, 1]),
            }
        )

    return {
        "prediction_type": "approximate assignment to existing fitted clusters",
        "new_clusters_discovered": False,
        "input_feature_order": FEATURE_ORDER,
        "count": len(items),
        "items": items,
    }
