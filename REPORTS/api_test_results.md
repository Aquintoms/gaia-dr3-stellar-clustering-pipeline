# FastAPI Functional Test Record

**Project:** Gaia DR3 Unsupervised Clustering  
**Test date:** 18 November 2026  
**Test method:** Manual functional tests through FastAPI Swagger UI (`/docs`)  
**Service address:** `http://127.0.0.1:8000`

## Summary

Ten endpoint test scenarios completed successfully. Every request returned HTTP 200.

| # | Method and endpoint | Test parameters / purpose | Observed result |
|---:|---|---|---|
| 1 | `GET /health` | Check application, catalogue and estimator loading | `status=ok`; 249,295 catalogue rows; scaler, K-Means, UMAP and HDBSCAN loaded; HDBSCAN prediction data available |
| 2 | `GET /api/v1/summary` | Check catalogue dimensions, features, dataset hash and cluster counts | 249,295 observations; six expected features; K-Means 4 clusters; HDBSCAN 26 clusters plus noise; 210,655 noise observations (84.5003%) |
| 3 | `GET /api/v1/model-info` | Inspect training configuration and package metadata | Feature order, model parameters, Python/package versions and HDBSCAN approximate prediction support returned |
| 4 | `GET /api/v1/clusters?algorithm=hdbscan&include_noise=true` | List HDBSCAN groups | 27 labels returned: 26 non-noise clusters and label -1 (noise) |
| 5 | `GET /api/v1/clusters/24?algorithm=hdbscan` | Retrieve one HDBSCAN cluster summary | Cluster 24 returned; 9,356 observations; catalogue fraction approximately 0.03753 |
| 6 | `GET /api/v1/stars` | Query HDBSCAN cluster 24, page 1, page size 5, selected fields and ascending source ID | 9,356 matches; five records returned; 1,872 pages; all returned labels were 24 |
| 7 | `GET /api/v1/stars/by-id/2964390723303808` | Retrieve one Gaia observation by source ID | One record returned; K-Means label 3 and HDBSCAN label 24 |
| 8 | `POST /api/v1/predict` | Predict the same observation using its six engineered features | K-Means prediction 3; HDBSCAN approximate prediction 24; membership strength 0.8401282604; no new clusters discovered |
| 9 | `GET /api/v1/clusters?algorithm=kmeans&include_noise=true` | List K-Means groups | Four clusters returned with counts: 92,678; 27,144; 26,363; 103,110 |
| 10 | `GET /api/v1/stars` | Query K-Means cluster 0, page 1, page size 5, selected fields and ascending source ID | 92,678 matches; five records returned; 18,536 pages; all returned labels were 0 |

## Reconciliation checks

- K-Means cluster counts sum to 249,295.
- HDBSCAN non-noise observations sum to 38,640; adding 210,655 noise observations gives 249,295.
- The API-reported dataset SHA-256 was `38d5c00cd88c101b8bdb8d329bc2130f1b4ee6edfe683876fb17d86a6944c3ed`, matching the training metadata.
- The sample prediction reproduced the selected observation's stored K-Means and HDBSCAN labels.

## Scope and limitations

These were manual functional tests against the local FastAPI service, saved model artifacts and clustered catalogue. The prediction test used one already-seen catalogue observation as a consistency check. It does not establish generalization performance on unseen data, model quality beyond the separately recorded evaluation, load capacity, security under hostile traffic, or successful container deployment. Docker validation and broader automated tests remain separate tasks.

The HDBSCAN membership strength is an algorithmic membership-strength measure, not a calibrated probability.
