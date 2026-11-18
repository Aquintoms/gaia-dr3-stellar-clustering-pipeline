# Docker Deployment Test Results

## Deployment summary

The Gaia DR3 clustering FastAPI application was built and launched using Docker Compose.

- Compose service: `gaia-api`
- Container: `gaia-stellar-clustering-api`
- Image: `gaia-stellar-clustering-api:1.0.0`
- Host binding: `127.0.0.1:8000` to container port `8000`
- Runtime status: `Up (healthy)`
- Compose configuration validation: passed with `docker compose config --quiet`

## Container health verification

The container health check and `GET /health` endpoint returned a successful response:

```json
{
  "status": "ok",
  "catalogue_loaded": true,
  "catalogue_rows": 249295,
  "models_loaded": ["scaler", "kmeans", "umap", "hdbscan"],
  "hdbscan_prediction_data_available": true
}
```

This confirms that the 249,295-row catalogue and all four saved model components were loaded in the container, and HDBSCAN prediction data was available.

## API endpoint tests in Docker

The following endpoints were exercised against the Docker-hosted API. Each returned HTTP `200`:

| Method | Endpoint | Result |
|---|---|---|
| GET | `/health` | HTTP 200 |
| GET | `/api/v1/summary` | HTTP 200 |
| GET | `/api/v1/model-info` | HTTP 200 |
| GET | `/api/v1/clusters` | HTTP 200 |
| GET | `/api/v1/stars/by-id/2964390723303808` | HTTP 200 |
| POST | `/api/v1/predict` | HTTP 200 |

## Prediction verification

A prediction request was submitted for the existing Gaia DR3 catalogue observation with source ID `2964390723303808`, using the six input features in the expected order:

1. `x`
2. `y`
3. `z`
4. `tangential_velocity`
5. `abs_g_mag`
6. `bp_rp`

The Docker-hosted API returned:

```json
{
  "prediction_type": "approximate assignment to existing fitted clusters",
  "new_clusters_discovered": false,
  "input_feature_order": [
    "x",
    "y",
    "z",
    "tangential_velocity",
    "abs_g_mag",
    "bp_rp"
  ],
  "count": 1,
  "items": [
    {
      "source_id": "2964390723303808",
      "kmeans_cluster": 3,
      "hdbscan_cluster": 24,
      "hdbscan_label": "Cluster 24",
      "hdbscan_membership_strength": 0.8401282604319379,
      "umap_1": 10.219209671020508,
      "umap_2": 5.689267635345459
    }
  ]
}
```

The K-Means assignment (`3`), HDBSCAN assignment (`24`), and membership strength (approximately `0.840128`) are consistent with the earlier local API test for this observation. The response explicitly describes the operation as an approximate assignment to existing fitted clusters; it does not discover or fit new clusters. HDBSCAN membership strength is not a calibrated probability.

The UMAP coordinates are produced by transforming the submitted observation through the fitted reducer. They should not be treated as guaranteed bit-for-bit reproductions of coordinates stored from the original UMAP fit.

## Outcome

**Docker deployment verification passed for the checks recorded above.** The container reached healthy status, the catalogue and saved models loaded, and the tested API endpoints returned HTTP 200, including a prediction request using a real catalogue observation.

These tests verify successful container startup, resource loading, and the tested API request paths. They do not, by themselves, establish model generalization performance beyond the separate model evaluation already documented in the project.
