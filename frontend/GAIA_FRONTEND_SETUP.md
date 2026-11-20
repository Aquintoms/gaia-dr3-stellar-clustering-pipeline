# Gaia Stellar Explorer — frontend patch

This patch replaces the generated Next.js starter page with the Gaia Stellar Explorer interface and connects it to the existing FastAPI service through Next.js rewrites.

## Apply the patch

Copy/replace these files in the existing `frontend/` directory:

- `next.config.ts`
- `src/app/layout.tsx`
- `src/app/page.tsx`
- `src/app/globals.css`
- `src/components/UmapExplorer.tsx`
- `src/lib/api.ts`

Do not replace `package.json`, `package-lock.json`, or `node_modules`.

## Run locally

Keep the existing Gaia FastAPI Docker service running on `http://127.0.0.1:8000`.

From the `frontend/` directory:

```powershell
npm run dev
```

Open `http://localhost:3000`.

The Next.js development server proxies `/api/*` to the FastAPI `/api/*` routes and `/backend/health` to the FastAPI `/health` route. This avoids browser CORS configuration changes.

If FastAPI uses a different address, set `GAIA_API_URL` before starting Next.js, for example:

```powershell
$env:GAIA_API_URL = "http://127.0.0.1:8001"
npm run dev
```

The map initially loads a 5,000-row preview. Use **Load full catalogue** to fetch the remaining pages (up to 249,295 observations) from the existing paginated API. The API caps each page at 5,000 records.

## Scientific notes

- UMAP coordinates are a two-dimensional embedding, not physical sky coordinates.
- K-Means and HDBSCAN labels are separate algorithmic label systems.
- HDBSCAN label `-1` is shown as noise / unassigned, not as an astrophysical class.
- Prediction accepts the six already-engineered input features in the training order and assigns them to existing fitted models; it does not discover new clusters.
