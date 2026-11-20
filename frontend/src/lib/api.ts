export type Algorithm = "kmeans" | "hdbscan";

export interface ApiHealth {
  status: string;
  catalogue_loaded: boolean;
  catalogue_rows: number;
  models_loaded: string[];
  hdbscan_prediction_data_available: boolean;
}

export interface KMeansCounts {
  cluster_count: number;
  cluster_sizes: Record<string, number>;
}

export interface HdbscanCounts {
  cluster_count_excluding_noise: number;
  noise_count: number;
  noise_percentage: number;
  cluster_sizes: Record<string, number>;
}

export interface CatalogueSummary {
  catalogue: string;
  total_observations: number;
  feature_count: number;
  features_in_order: string[];
  dataset_sha256: string | null;
  kmeans: KMeansCounts;
  hdbscan: HdbscanCounts;
}

export interface ClusterSummary {
  algorithm?: Algorithm;
  cluster_id: number;
  label: string;
  is_noise: boolean;
  star_count: number;
  catalogue_fraction: number;
  mean_bp_rp?: number | null;
  mean_abs_g_mag?: number | null;
  mean_distance_pc?: number | null;
  mean_tangential_velocity?: number | null;
  median_tangential_velocity?: number | null;
}

export interface ClusterListResponse {
  algorithm: Algorithm;
  count: number;
  clusters: ClusterSummary[];
}

export interface StarPoint {
  source_id: string;
  umap_1: number;
  umap_2: number;
  kmeans_label: number;
  hdbscan_label: number;
  bp_rp?: number | null;
  abs_g_mag?: number | null;
  distance_pc?: number | null;
  tangential_velocity?: number | null;
  ra?: number | null;
  dec?: number | null;
  x?: number | null;
  y?: number | null;
  z?: number | null;
}

export interface StarsPage {
  algorithm: Algorithm;
  total: number;
  page: number;
  page_size: number;
  pages: number;
  items: StarPoint[];
}

export interface StarRecord extends StarPoint {
  parallax?: number | null;
  ruwe?: number | null;
  pmra?: number | null;
  pmdec?: number | null;
  phot_g_mean_mag?: number | null;
  phot_bp_mean_mag?: number | null;
  phot_rp_mean_mag?: number | null;
  parallax_snr?: number | null;
  pm_total?: number | null;
}

export interface StarLookupResponse {
  star: StarRecord;
}

export interface PredictionInput {
  source_id?: string;
  x: number;
  y: number;
  z: number;
  tangential_velocity: number;
  abs_g_mag: number;
  bp_rp: number;
}

export interface PredictionItem {
  source_id: string | null;
  kmeans_cluster: number;
  hdbscan_cluster: number;
  hdbscan_label: string;
  hdbscan_membership_strength: number;
  umap_1: number;
  umap_2: number;
}

export interface PredictionResponse {
  prediction_type: string;
  new_clusters_discovered: boolean;
  input_feature_order: string[];
  count: number;
  items: PredictionItem[];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      // Keep the HTTP status message when the server does not return JSON.
    }
    throw new Error(detail);
  }

  return (await response.json()) as T;
}

export function getHealth() {
  return request<ApiHealth>("/backend/health");
}

export function getSummary() {
  return request<CatalogueSummary>("/api/v1/summary");
}

export function getClusters(algorithm: Algorithm, includeNoise = true) {
  const params = new URLSearchParams({ algorithm, include_noise: String(includeNoise) });
  return request<ClusterListResponse>(`/api/v1/clusters?${params.toString()}`);
}

export function getStarsPage(options: {
  algorithm: Algorithm;
  page: number;
  pageSize?: number;
  clusterId?: number | null;
}) {
  const params = new URLSearchParams({
    algorithm: options.algorithm,
    page: String(options.page),
    page_size: String(options.pageSize ?? 5000),
    sort_by: "source_id",
    sort_order: "asc",
    fields: "source_id,umap_1,umap_2,x,y,z,kmeans_label,hdbscan_label,bp_rp,abs_g_mag,distance_pc,tangential_velocity,ra,dec",
  });
  if (options.clusterId !== undefined && options.clusterId !== null) {
    params.set("cluster_id", String(options.clusterId));
  }
  return request<StarsPage>(`/api/v1/stars?${params.toString()}`);
}

export function getStar(sourceId: string) {
  return request<StarLookupResponse>(`/api/v1/stars/by-id/${encodeURIComponent(sourceId)}`);
}

export function predictStar(input: PredictionInput) {
  return request<PredictionResponse>("/api/v1/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ stars: [input] }),
  });
}
