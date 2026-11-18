# Gaia DR3 Stellar Clustering: An End-to-End Machine Learning Pipeline

An end-to-end machine-learning engineering project that prepares a Gaia Data Release 3 (Gaia DR3) stellar catalogue subset, engineers physical and kinematic features, applies unsupervised clustering, evaluates the resulting groups, and serves the fitted models through a FastAPI application packaged with Docker.

The project uses astronomy as its application domain. Its main focus is the reproducible machine-learning workflow: data preparation, feature engineering, model training and evaluation, artifact verification, API inference, and containerized deployment.

## Project at a glance

- **Dataset:** 249,295 Gaia DR3 source records in the processed project dataset.
- **Input features:** six engineered spatial, kinematic, and photometric features.
- **Clustering methods:** K-Means and HDBSCAN.
- **Dimensionality reduction:** UMAP, used to create a two-dimensional representation for HDBSCAN clustering and exploration.
- **Serving:** FastAPI, with a Docker Compose deployment.
- **Python:** 3.12 (the validated project environment used Python 3.12.7).

## Repository structure

```text
gaia-dr3-stellar-clustering-pipeline/
|-- DATA/
|   |-- gaia_subset_raw.csv
|   |-- gaia_subset_raw_snapshot.parquet
|   |-- final_article_data.csv
|   `-- processed/                 # Generated pipeline outputs
|-- MODEL/                         # Locally stored fitted model artifacts
|-- NOTEBOOK/
|   `-- Gaia_clustering_project.ipynb
|-- OUTPUTS/                       # Generated predictions, embeddings and reports
|   `-- environment-lock.txt       # Captured package versions
|-- REPORTS/
|   |-- api_test_results.md
|   |-- artifact_verification.md
|   `-- docker_deployment_test_results.md
|-- SCRIPTS/
|   |-- data_pipeline.py
|   |-- train.py
|   `-- verify_artifacts.py
|-- main.py                        # FastAPI application
|-- requirements.txt               # Project / modelling environment
|-- requirements-api.txt           # Pinned API runtime dependencies
|-- Dockerfile
|-- docker-compose.yml
|-- .dockerignore
|-- .gitignore
`-- README.md
```

The `MODEL/` and `OUTPUTS/` directories contain generated files and are required at runtime. Their contents are not all tracked in ordinary Git history; see [Model and data artifacts](#model-and-data-artifacts) before attempting to run the API or Docker deployment from a fresh clone.

## Data and preparation

The project uses a subset of Gaia DR3 catalogue records. The working dataset contains 249,295 observations. The data files in `DATA/` include the raw CSV input, a Parquet snapshot, and a supplementary CSV.

### Quality checks

The data pipeline applies the following astrometric quality conditions:

- `ruwe < 1.4`
- positive parallax
- positive parallax uncertainty
- parallax signal-to-noise ratio greater than 5

The BP/RP excess-factor interval was examined as a diagnostic, but it was **not** used as an additional row filter in the validated pipeline. This distinction is intentional and should be preserved when reproducing the results.

### Feature engineering

The pipeline derives distance, absolute G-band magnitude, total proper motion, tangential velocity, and Cartesian Galactic-position coordinates. The Cartesian coordinates are calculated using Astropy coordinate transformations.

The six model input features, in the required order, are:

1. `x`
2. `y`
3. `z`
4. `tangential_velocity`
5. `abs_g_mag`
6. `bp_rp`

The validated data-pipeline output contains 249,295 rows. A comparison against the corresponding notebook feature values found identical values for all six feature columns in the checked pipeline run.

## Modelling approach

### Preprocessing

A `StandardScaler` is fitted to the six model features. The fitted scaler is saved and reused by the API so that prediction inputs are transformed consistently with the training workflow.

### K-Means

K-Means is fitted using four clusters, `random_state=42`, and `n_init=10`.

The recorded cluster counts are:

| K-Means label | Number of stars |
|---:|---:|
| 0 | 92,678 |
| 1 | 27,144 |
| 2 | 26,363 |
| 3 | 103,110 |
| **Total** | **249,295** |

The silhouette score is **0.3236**. It was calculated on a 10,000-row sample in the standardized six-feature space.

### UMAP and HDBSCAN

UMAP reduces the standardized six-feature representation to two dimensions using:

| Parameter | Value |
|---|---:|
| `n_components` | 2 |
| `n_neighbors` | 50 |
| `min_dist` | 0.1 |
| `random_state` | 42 |

HDBSCAN is fitted to the two-dimensional UMAP representation with:

| Parameter | Value |
|---|---|
| `min_cluster_size` | 500 |
| `min_samples` | 500 |
| `cluster_selection_method` | `leaf` |

The fitted result contains **26 non-noise clusters**. HDBSCAN assigned **210,655 observations (84.50%)** the label `-1`, which means that the observations were treated as noise or were not assigned to a cluster under the fitted density-based model. It does not mean that those stars are invalid or physically unimportant.

The reported DBCV score is **0.2178**. It was calculated on a sample of up to 20,000 non-noise observations in the two-dimensional UMAP space. It is not directly comparable with the K-Means silhouette score because the metrics were calculated on different representations and samples.

Cluster labels are algorithmic assignments, not confirmed astrophysical classes. K-Means and HDBSCAN labels are separate label systems and should not be interpreted as corresponding one-to-one.

## Reproducing the pipeline

The commands below assume that the repository has been cloned, the required input data is present under `DATA/`, and commands are run from the repository root.

### 1. Create and activate a Python environment (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, use the activation method permitted by your machine's execution policy, or invoke the virtual environment's Python directly as `.\.venv\Scripts\python.exe`.

### 2. Run the data pipeline

```powershell
python SCRIPTS/data_pipeline.py
```

The pipeline reads the raw Gaia subset from `DATA/` and writes processed Parquet outputs and a run manifest under `DATA/processed/`.

### 3. Train the models and generate outputs

```powershell
python SCRIPTS/train.py
```

The training script fits the scaler, K-Means, UMAP, and HDBSCAN workflow, saves fitted artifacts under `MODEL/`, and writes the clustered catalogue and evaluation outputs under `OUTPUTS/`.

### 4. Verify the saved artifacts

```powershell
python SCRIPTS/verify_artifacts.py
```

This verification script loads the saved artifacts and performs sample inference checks. It is intended to verify the saved models; it does not need to retrain them.

The notebook in `NOTEBOOK/` contains the exploratory and modelling work. The scripted pipeline and training workflow are the reproducible execution path for the project. Existing validated outputs should not be regenerated unless a reproducibility check or a deliberate model change requires it.

## Model and data artifacts

The API expects the fitted models and the clustered catalogue to be available locally. The key runtime files are:

```text
MODEL/
├── standard_scaler.joblib
├── kmeans_model.joblib
├── umap_reducer.joblib
├── hdbscan_model.joblib
├── hdbscan_label_mapping.json
└── model_metadata.json

OUTPUTS/
└── clustered_stars.parquet
```

Other generated outputs include evaluation metrics, cluster summaries, cluster-membership tables, K-Means centres in physical units, scaled features, UMAP embeddings, and environment information.

**Important for a fresh clone:** the large fitted model files, processed catalogue, and generated arrays are not included in the normal GitHub source checkout. A versioned artifact bundle or an approved institutional storage location must be provided separately and copied into the expected `MODEL/` and `OUTPUTS/` paths. No public artifact-download URL is specified in this README. The UMAP reducer is particularly large and exceeds GitHub's ordinary 100 MB per-file limit, so it should not be added to a normal Git commit.

Keep the artifact bundle associated with the same code and model configuration used to generate it. The API metadata and verification report should be checked when restoring artifacts.

## Running the API locally

The API loads the saved models from `MODEL/` and the clustered catalogue from `OUTPUTS/`. Ensure the required artifacts are present before starting it.

```powershell
python -m pip install -r requirements-api.txt
uvicorn main:app --host 127.0.0.1 --port 8000
```

Open the interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

The local health endpoint is:

```text
http://127.0.0.1:8000/health
```

### API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Application, catalogue, and model readiness |
| `GET` | `/api/v1/summary` | Dataset and clustering summary |
| `GET` | `/api/v1/model-info` | Model configuration and metadata |
| `GET` | `/api/v1/clusters` | List cluster information |
| `GET` | `/api/v1/clusters/{cluster_id}` | Retrieve details for a cluster |
| `GET` | `/api/v1/stars` | Browse catalogue records, with supported filters and pagination |
| `GET` | `/api/v1/stars/by-id/{source_id}` | Retrieve a star using its Gaia source ID |
| `POST` | `/api/v1/predict` | Assign supplied feature records to the fitted K-Means and HDBSCAN models |

The prediction endpoint performs **approximate assignment to existing fitted clusters**. It does not discover or fit new clusters for incoming records. HDBSCAN membership strength is an assignment-strength measure, not a calibrated probability.

### Example prediction request

```json
{
  "stars": [
    {
      "source_id": "2964390723303808",
      "x": 383.24498137385564,
      "y": 458.21768407472973,
      "z": 45.51895113909228,
      "tangential_velocity": 87.49996158911833,
      "abs_g_mag": 5.716966639237349,
      "bp_rp": 1.0959029999999998
    }
  ]
}
```

The validated test observation returned K-Means label `3` and HDBSCAN label `24` in both the local and Docker-hosted API tests.

## Docker deployment

Docker Desktop with Linux containers and Docker Compose is required. The Docker image installs the API runtime and application code. The fitted model files and clustered catalogue are mounted from the host machine at runtime as read-only volumes; they are not copied into the image build context.

Before starting the service, make sure that the required local files exist in `MODEL/` and `OUTPUTS/` as described above.

Build and start the service from the repository root:

```powershell
docker compose up -d --build
```

Check the container status:

```powershell
docker compose ps
```

View recent logs:

```powershell
docker compose logs --tail 100 gaia-api
```

Follow logs continuously:

```powershell
docker compose logs -f gaia-api
```

The service is published on the host at `http://127.0.0.1:8000`. Open `http://127.0.0.1:8000/docs` to use Swagger UI. The Compose configuration includes a health check against `/health` and uses a writable temporary filesystem location for the Numba cache while keeping the container root filesystem read-only.

Stop and remove the container and Compose network with:

```powershell
docker compose down
```

The deployment was tested with the catalogue loaded (249,295 rows), all four model components loaded, and HDBSCAN prediction data available. The tested API requests returned HTTP 200, including the sample prediction shown above. See `REPORTS/docker_deployment_test_results.md` for the recorded deployment test evidence.

## Test and verification evidence

The `REPORTS/` directory contains the following project evidence:

- `api_test_results.md` — local API endpoint test results.
- `artifact_verification.md` — checks that saved model artifacts load and support sample inference.
- `docker_deployment_test_results.md` — Docker deployment and endpoint test results.

These checks establish that the saved artifacts can be loaded and that the tested service routes operate in the validated environment. They do not by themselves establish that the clusters are scientifically validated classes or that predictions generalize to all Gaia sources.

## Limitations and interpretation

- The analysis uses the project subset of Gaia DR3, not the complete Gaia catalogue.
- The clusters are unsupervised groupings produced by the selected features, preprocessing, and algorithms. They should not be treated as labelled astrophysical ground truth.
- HDBSCAN's `-1` label denotes noise/unassigned observations under the fitted density model.
- The high HDBSCAN noise fraction is an important characteristic of this particular feature space and parameterization; it should be reported rather than hidden.
- Silhouette and DBCV were evaluated on different samples and representations, so their numeric values should not be ranked directly against one another.
- Approximate prediction assigns new records to existing fitted structures; it does not refit the clustering models or discover new clusters.

## Gaia data acknowledgement and citations

The Gaia Data Release 3 Archive requests that users of Gaia DR3 data include the following acknowledgement and cite both the Gaia mission paper and the Gaia DR3 release paper. The acknowledgement below is reproduced using the wording supplied in the [official Gaia DR3 credit and citation instructions](https://gea.esac.esa.int/archive/documentation/GDR3/Miscellaneous/sec_credit_and_citation_instructions/):

This work has made use of data from the European Space Agency (ESA) mission Gaia (https://www.cosmos.esa.int/gaia), processed by the Gaia Data Processing and Analysis Consortium (DPAC, https://www.cosmos.esa.int/web/gaia/dpac/consortium). Funding for the DPAC has been provided by national institutions, in particular the institutions participating in the Gaia Multilateral Agreement.

### Required Gaia references

1. Gaia Collaboration, T. Prusti, J. H. J. de Bruijne, et al. (2016). **The Gaia mission.** *Astronomy & Astrophysics*, 595, A1. https://doi.org/10.1051/0004-6361/201629272
2. Gaia Collaboration, A. Vallenari, A. G. A. Brown, et al. (2023). **Gaia Data Release 3: Summary of the content and survey properties.** *Astronomy & Astrophysics*, 674, A1. https://doi.org/10.1051/0004-6361/202243940

The Gaia Archive also lists additional Gaia DR3 content and validation papers that should be cited where relevant to the catalogue products or processing used. Consult the [official credit and citation instructions](https://gea.esac.esa.int/archive/documentation/GDR3/Miscellaneous/sec_credit_and_citation_instructions/) and the [Gaia DR3 papers list](https://www.cosmos.esa.int/web/gaia/dr3-papers) when preparing further academic work.

Additional data access and documentation resources:

- [Gaia DR3 documentation](https://gea.esac.esa.int/archive/documentation/GDR3/)
- [Gaia ESA Archive](https://gea.esac.esa.int/archive/)

## Project repository

[https://github.com/Aquintoms/gaia-dr3-stellar-clustering-pipeline](https://github.com/Aquintoms/gaia-dr3-stellar-clustering-pipeline)
