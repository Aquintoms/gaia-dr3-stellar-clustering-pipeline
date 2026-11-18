# Saved Model Artifact Verification

**Project:** Gaia DR3 Unsupervised Clustering  
**Verification date:** 18 November 2026  
**Verification script:** `SCRIPTS/verify_artifacts.py`  
**Command:** `python SCRIPTS\\verify_artifacts.py`

## Purpose

Verify that the serialized preprocessing and clustering estimators can be loaded and used for a small inference pass in the project's Python environment, without fitting or retraining any model.

## Environment

- Python: 3.12.7
- Environment: project `.venv`
- NumPy: 2.5.4
- pandas: 3.1.1
- scikit-learn: 1.9.1
- joblib: 1.6.0
- umap-learn: 0.5.12
- hdbscan: 0.8.44

The package versions reported by the verification run matched the versions recorded in `MODEL/model_metadata.json` for all six packages checked.

## Results

| Check | Result |
|---|---|
| StandardScaler deserialized | PASS |
| K-Means deserialized (4 clusters; 6 input features) | PASS |
| UMAP reducer deserialized | PASS |
| HDBSCAN deserialized | PASS |
| HDBSCAN prediction data available | PASS |
| Sample size | 10 observations |
| Scaler output versus saved training array | Exact match; maximum absolute difference 0 |
| K-Means sample predictions versus stored training labels | Match |
| UMAP transform output shape | `(10, 2)` |
| HDBSCAN approximate prediction | Completed successfully |

The HDBSCAN predictions included both noise/unassigned observations (`-1`) and assignments to existing clusters. This is consistent with approximate prediction: it assigns new observations to the fitted clustering structure and does not discover new clusters.

## Interpretation and limitations

This check establishes that the saved artifacts can be loaded together in the project's recorded Python environment and that the tested sample inference path executes successfully.

The K-Means comparison checks consistency with stored training labels for the first ten catalogue rows; it is not an independent estimate of model generalisation performance. The ten-row HDBSCAN output is a smoke test of approximate prediction, not a new clustering experiment or a scientific validation of cluster identities.

The test did not fit or retrain any model and did not modify the existing project artifacts. The original notebook was not rerun.

## Conclusion

**PASS:** The saved StandardScaler, K-Means, UMAP and HDBSCAN artifacts were loaded successfully, and the sample inference workflow completed in the project's `.venv`.
