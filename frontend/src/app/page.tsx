"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import type { FormEvent, ReactNode } from "react";
import UmapExplorer from "@/components/UmapExplorer";
import SpatialExplorer from "@/components/SpatialExplorer";
import {
  Algorithm,
  ApiHealth,
  CatalogueSummary,
  ClusterSummary,
  PredictionInput,
  PredictionResponse,
  StarRecord,
  StarPoint,
  getClusters,
  getHealth,
  getStar,
  getStarsPage,
  getSummary,
  predictStar,
} from "@/lib/api";

type View = "overview" | "map" | "clusters" | "lookup" | "predict";
type MapMode = "spatial" | "umap";

const VIEW_LABELS: Record<View, string> = {
  overview: "Overview",
  map: "Stellar atlas",
  clusters: "Cluster explorer",
  lookup: "Star lookup",
  predict: "Model inference",
};

const INITIAL_FEATURES = [
  { key: "x", label: "Cartesian X", unit: "pc", placeholder: "383.245" },
  { key: "y", label: "Cartesian Y", unit: "pc", placeholder: "458.218" },
  { key: "z", label: "Cartesian Z", unit: "pc", placeholder: "45.519" },
  { key: "tangential_velocity", label: "Tangential velocity", unit: "km/s", placeholder: "87.500" },
  { key: "abs_g_mag", label: "Absolute G magnitude", unit: "mag", placeholder: "5.717" },
  { key: "bp_rp", label: "BP − RP colour", unit: "mag", placeholder: "1.096" },
] as const;

function number(value: number | null | undefined, digits = 1) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function percent(value: number | undefined) {
  if (value === undefined || !Number.isFinite(value)) return "—";
  return `${value.toFixed(2)}%`;
}

function count(value: number | undefined) {
  return value === undefined ? "—" : value.toLocaleString();
}

function Icon({ name }: { name: string }) {
  const common = { width: 18, height: 18, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.65, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
  const paths: Record<string, ReactNode> = {
    overview: <><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="4" rx="1.5"/><rect x="14" y="10" width="7" height="11" rx="1.5"/><rect x="3" y="13" width="7" height="8" rx="1.5"/></>,
    map: <><circle cx="12" cy="12" r="8.5"/><ellipse cx="12" cy="12" rx="3.5" ry="8.5"/><path d="M3.8 8.5c2.6 1.8 13.8 1.8 16.4 0M3.8 15.5c2.6-1.8 13.8-1.8 16.4 0"/></>,
    clusters: <><circle cx="8" cy="8" r="3"/><circle cx="17" cy="7" r="2"/><circle cx="15" cy="17" r="3"/><path d="m10.5 9.5 3.5 5M10.5 8.3l4.5-.7M8.7 11l4.3 4"/></>,
    search: <><circle cx="10.8" cy="10.8" r="6.5"/><path d="m16 16 4.5 4.5"/></>,
    predict: <><path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1m8.6 8.6 2.1 2.1m0-12.8-2.1 2.1m-8.6 8.6-2.1 2.1"/><circle cx="12" cy="12" r="4.3"/></>,
    orbit: <><circle cx="12" cy="12" r="1.6" fill="currentColor" stroke="none"/><ellipse cx="12" cy="12" rx="9" ry="4.1" transform="rotate(-32 12 12)"/><path d="M19 6.4a2 2 0 1 0 0 .1"/></>,
    refresh: <><path d="M20 7v5h-5M4 17v-5h5"/><path d="M5.7 9A7 7 0 0 1 18 6l2 6M4 12l2 6a7 7 0 0 0 12.3-3"/></>,
    arrow: <><path d="M7 17 17 7M8 7h9v9"/></>,
  };
  return <svg {...common} aria-hidden="true">{paths[name] ?? paths.orbit}</svg>;
}

function Sidebar({ view, onChange, backendOnline }: { view: View; onChange: (view: View) => void; backendOnline: boolean }) {
  const items: { id: View; label: string; icon: string }[] = [
    { id: "overview", label: "Overview", icon: "overview" },
    { id: "map", label: "Stellar atlas", icon: "map" },
    { id: "clusters", label: "Cluster explorer", icon: "clusters" },
    { id: "lookup", label: "Star lookup", icon: "search" },
    { id: "predict", label: "Model inference", icon: "predict" },
  ];
  return (
    <aside className="sidebar">
      <div className="brand-lockup">
        <div className="brand-mark"><Icon name="orbit" /></div>
        <div><div className="brand-name">GAIA <span>ATLAS</span></div><div className="brand-caption">STELLAR EXPLORER</div></div>
      </div>
      <div className="sidebar-section-label">WORKSPACE</div>
      <nav className="side-nav" aria-label="Main navigation">
        {items.map((item) => (
          <button key={item.id} className={`nav-item ${view === item.id ? "selected" : ""}`} onClick={() => onChange(item.id)}>
            <Icon name={item.icon} /><span>{item.label}</span>{view === item.id && <i className="nav-active-dot" />}
          </button>
        ))}
      </nav>
      <div className="sidebar-bottom">
        <div className="side-project-card">
          <div className="side-project-top"><span className="tiny-orbit">✳</span><span>PROJECT DATASET</span></div>
          <strong>Gaia DR3</strong>
          <span>Orion–Taurus sightline</span>
          <div className="side-project-rule" />
          <div className="side-project-meta"><span>249,295 sources</span><span className="dataset-dot" /></div>
        </div>
        <div className="sidebar-status"><span className={backendOnline ? "status-dot online" : "status-dot"} /><span>{backendOnline ? "API connected" : "API connection pending"}</span></div>
        <div className="sidebar-footer">RESEARCH INTERFACE <span>·</span> DR3 / 01</div>
      </div>
    </aside>
  );
}

function MetricCard({ label, value, detail, accent, icon }: { label: string; value: string; detail: string; accent: string; icon: string }) {
  return (
    <article className="metric-card">
      <div className="metric-top"><span>{label}</span><span className={`metric-icon ${accent}`}>{icon}</span></div>
      <div className="metric-value">{value}</div>
      <div className="metric-detail">{detail}</div>
      <div className={`metric-accent-line ${accent}`} />
    </article>
  );
}

function ClusterMiniCard({ cluster, algorithm, onClick }: { cluster: ClusterSummary; algorithm: Algorithm; onClick: () => void }) {
  const noise = algorithm === "hdbscan" && cluster.cluster_id === -1;
  return (
    <button className={`cluster-mini-card ${noise ? "noise-card" : ""}`} onClick={onClick}>
      <div className="cluster-mini-head"><span className="cluster-mini-label">{noise ? "NOISE / UNASSIGNED" : `CLUSTER ${cluster.cluster_id.toString().padStart(2, "0")}`}</span><span className="cluster-mini-arrow">↗</span></div>
      <strong>{count(cluster.star_count)}</strong>
      <div className="cluster-mini-track"><i style={{ width: `${Math.max(2, cluster.catalogue_fraction * 100)}%` }} /></div>
      <span className="cluster-mini-foot">{percent(cluster.catalogue_fraction * 100)} of catalogue</span>
    </button>
  );
}

function StarDetails({ star }: { star: StarRecord }) {
  const details: [string, string][] = [
    ["Right ascension", `${number(star.ra, 5)}°`],
    ["Declination", `${number(star.dec, 5)}°`],
    ["Distance", `${number(star.distance_pc, 1)} pc`],
    ["Parallax", `${number(star.parallax, 4)} mas`],
    ["Tangential velocity", `${number(star.tangential_velocity, 2)} km/s`],
    ["Absolute G magnitude", number(star.abs_g_mag, 3)],
    ["BP − RP colour", number(star.bp_rp, 3)],
    ["RUWE", number(star.ruwe, 3)],
    ["K-Means label", String(star.kmeans_label)],
    ["HDBSCAN label", star.hdbscan_label === -1 ? "Noise / unassigned" : `Cluster ${star.hdbscan_label}`],
  ];
  return (
    <div className="star-result-card">
      <div className="star-result-head"><div><div className="eyebrow">GAIA DR3 SOURCE RECORD</div><h3>{star.source_id}</h3><p>Catalogue observation · Saved clustering assignment</p></div><div className="star-identity-icon">✦</div></div>
      <div className="star-detail-grid">{details.map(([label, value]) => <div className="star-detail-cell" key={label}><span>{label}</span><strong>{value}</strong></div>)}</div>
      <p className="subtle-note">Cluster labels are model assignments, not confirmed astrophysical classifications.</p>
    </div>
  );
}

export default function Home() {
  const [view, setView] = useState<View>("overview");
  const [lightsOut, setLightsOut] = useState(false);
  const [summary, setSummary] = useState<CatalogueSummary | null>(null);
  const [health, setHealth] = useState<ApiHealth | null>(null);
  const [backendError, setBackendError] = useState("");
  const [clustersHdbscan, setClustersHdbscan] = useState<ClusterSummary[]>([]);
  const [clustersKmeans, setClustersKmeans] = useState<ClusterSummary[]>([]);
  const [points, setPoints] = useState<StarPoint[]>([]);
  const [pointTotal, setPointTotal] = useState(249295);
  const [pointPages, setPointPages] = useState(1);
  const [algorithm, setAlgorithm] = useState<Algorithm>("hdbscan");
  const [mapMode, setMapMode] = useState<MapMode>("spatial");
  const [loadingFull, setLoadingFull] = useState(false);
  const [loadProgress, setLoadProgress] = useState(0);
  const [loadedFull, setLoadedFull] = useState(false);
  const [activeCluster, setActiveCluster] = useState<ClusterSummary | null>(null);
  const [starQuery, setStarQuery] = useState("");
  const [selectedStar, setSelectedStar] = useState<StarRecord | null>(null);
  const [lookupError, setLookupError] = useState("");
  const [lookupLoading, setLookupLoading] = useState(false);
  const [predictionValues, setPredictionValues] = useState<Record<string, string>>({});
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null);
  const [predictionError, setPredictionError] = useState("");
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [globalLoading, setGlobalLoading] = useState(true);
  const [toast, setToast] = useState("");

  const refreshData = useCallback(async () => {
    setGlobalLoading(true);
    setBackendError("");
    try {
      const [healthData, summaryData, hdbData, kmData, firstPage] = await Promise.all([
        getHealth(), getSummary(), getClusters("hdbscan", true), getClusters("kmeans"),
        getStarsPage({ algorithm: "hdbscan", page: 1, pageSize: 5000 }),
      ]);
      setHealth(healthData);
      setSummary(summaryData);
      setClustersHdbscan(hdbData.clusters);
      setClustersKmeans(kmData.clusters);
      setPoints(firstPage.items);
      setPointTotal(firstPage.total);
      setPointPages(firstPage.pages);
      setLoadedFull(firstPage.pages <= 1);
    } catch (error) {
      setBackendError(error instanceof Error ? error.message : "Could not connect to the Gaia API.");
      setHealth(null);
    } finally {
      setGlobalLoading(false);
    }
  }, []);

  useEffect(() => { void refreshData(); }, [refreshData]);

  const backendOnline = health?.status === "ok";
  const visibleClusters = algorithm === "hdbscan" ? clustersHdbscan : clustersKmeans;
  const topClusters = useMemo(() => visibleClusters.filter((cluster) => !cluster.is_noise).slice(0, 4), [visibleClusters]);

  const loadFullCatalogue = async () => {
    if (loadingFull || loadedFull || pointPages <= 1) return;
    setLoadingFull(true);
    setLoadProgress(0);
    try {
      const pages: StarPoint[][] = new Array(pointPages);
      pages[0] = points;
      let nextPage = 2;
      let completed = 1;
      const workers = Array.from({ length: Math.min(3, pointPages - 1) }, async () => {
        while (true) {
          const pageNumber = nextPage++;
          if (pageNumber > pointPages) return;
          const response = await getStarsPage({
            algorithm,
            page: pageNumber,
            pageSize: 5000,
            clusterId: activeCluster?.cluster_id ?? null,
          });
          pages[pageNumber - 1] = response.items;
          completed += 1;
          setLoadProgress((completed / pointPages) * 100);
        }
      });
      await Promise.all(workers);
      const flattened = pages.flat();
      setPoints(flattened);
      setPointTotal(flattened.length);
      setLoadedFull(true);
      setToast(`Loaded ${flattened.length.toLocaleString()} observations into the map.`);
      window.setTimeout(() => setToast(""), 3500);
    } catch (error) {
      setBackendError(error instanceof Error ? error.message : "Could not load the complete catalogue.");
    } finally {
      setLoadingFull(false);
    }
  };

  const openCluster = async (cluster: ClusterSummary, selectedAlgorithm: Algorithm) => {
    setAlgorithm(selectedAlgorithm);
    setActiveCluster(cluster);
    setLoadedFull(false);
    setLoadingFull(false);
    setLoadProgress(0);
    setView("map");
    setGlobalLoading(true);
    try {
      const firstPage = await getStarsPage({ algorithm: selectedAlgorithm, page: 1, pageSize: 5000, clusterId: cluster.cluster_id });
      setPoints(firstPage.items);
      setPointTotal(firstPage.total);
      setPointPages(firstPage.pages);
      setLoadedFull(firstPage.pages <= 1);
    } catch (error) {
      setBackendError(error instanceof Error ? error.message : "Could not load this cluster.");
    } finally {
      setGlobalLoading(false);
    }
  };

  const openAllStars = async () => {
    setActiveCluster(null);
    setLoadedFull(false);
    setGlobalLoading(true);
    try {
      const firstPage = await getStarsPage({ algorithm, page: 1, pageSize: 5000 });
      setPoints(firstPage.items);
      setPointTotal(firstPage.total);
      setPointPages(firstPage.pages);
      setLoadedFull(firstPage.pages <= 1);
    } catch (error) {
      setBackendError(error instanceof Error ? error.message : "Could not load the stellar catalogue.");
    } finally {
      setGlobalLoading(false);
    }
  };

  const selectPoint = async (point: StarPoint) => {
    setStarQuery(point.source_id);
    setView("lookup");
    setLookupLoading(true);
    setLookupError("");
    try {
      const response = await getStar(point.source_id);
      setSelectedStar(response.star);
    } catch (error) {
      setLookupError(error instanceof Error ? error.message : "Could not retrieve this source.");
    } finally {
      setLookupLoading(false);
    }
  };

  const submitLookup = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!/^\d+$/.test(starQuery.trim())) {
      setLookupError("Enter a numeric Gaia DR3 source identifier.");
      setSelectedStar(null);
      return;
    }
    setLookupLoading(true);
    setLookupError("");
    setSelectedStar(null);
    try {
      const response = await getStar(starQuery.trim());
      setSelectedStar(response.star);
    } catch (error) {
      setLookupError(error instanceof Error ? error.message : "Source lookup failed.");
    } finally {
      setLookupLoading(false);
    }
  };

  const loadExampleStar = async () => {
    setLookupLoading(true);
    setLookupError("");
    try {
      const response = await getStar("2964390723303808");
      setStarQuery(response.star.source_id);
      setSelectedStar(response.star);
    } catch (error) {
      setLookupError(error instanceof Error ? error.message : "Could not load the example source.");
    } finally {
      setLookupLoading(false);
    }
  };

  const fillExamplePrediction = async () => {
    try {
      const response = await getStar("2964390723303808");
      const star = response.star;
      setPredictionValues({
        x: String(star.x ?? ""), y: String(star.y ?? ""), z: String(star.z ?? ""),
        tangential_velocity: String(star.tangential_velocity ?? ""),
        abs_g_mag: String(star.abs_g_mag ?? ""), bp_rp: String(star.bp_rp ?? ""),
        source_id: star.source_id,
      });
      setPredictionResult(null);
      setPredictionError("");
    } catch (error) {
      setPredictionError(error instanceof Error ? error.message : "Could not load the example features.");
    }
  };

  const submitPrediction = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setPredictionLoading(true);
    setPredictionError("");
    setPredictionResult(null);
    try {
      const input = {} as PredictionInput;
      for (const feature of INITIAL_FEATURES) {
        const value = predictionValues[feature.key];
        if (value === undefined || value.trim() === "" || !Number.isFinite(Number(value))) {
          throw new Error(`Enter a valid number for ${feature.label}.`);
        }
        (input as unknown as Record<string, unknown>)[feature.key] = Number(value);
      }
      if (predictionValues.source_id) input.source_id = predictionValues.source_id;
      const result = await predictStar(input);
      setPredictionResult(result);
    } catch (error) {
      setPredictionError(error instanceof Error ? error.message : "Prediction could not be completed.");
    } finally {
      setPredictionLoading(false);
    }
  };

  const title = VIEW_LABELS[view];
  const description: Record<View, string> = {
    overview: "A high-level view of the Gaia DR3 clustering catalogue and its model outputs.",
    map: "Rotate the physical 3D stellar field or switch to the UMAP projection to inspect individual observations.",
    clusters: "Compare density-based groups and K-Means partitions in the saved catalogue.",
    lookup: "Retrieve a catalogue observation using its Gaia DR3 source identifier.",
    predict: "Assign an engineered six-feature observation to the existing fitted models.",
  };

  return (
    <div className={`app-shell ${view === "overview" ? "immersive-home" : ""} ${lightsOut ? "lights-out" : ""}`}>
      <Sidebar view={view} onChange={setView} backendOnline={Boolean(backendOnline)} />
      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumb"><span>GAIA ATLAS</span><b>/</b><strong>{title.toUpperCase()}</strong></div>
          <div className="topbar-right">
            <div className={`connection-pill ${backendOnline ? "connected" : ""}`}><span className="status-dot" />{backendOnline ? "LIVE API" : globalLoading ? "CONNECTING" : "API OFFLINE"}</div>
            <div className="topbar-version">DR3 <i /> LOCAL WORKSPACE</div>
            <button
              type="button"
              className="theme-toggle"
              onClick={() => setLightsOut((current) => !current)}
              aria-pressed={lightsOut}
              aria-label={lightsOut ? "Switch off Lights out theme" : "Switch on Lights out theme"}
              title={lightsOut ? "Return to the standard theme" : "Switch to the deeper cosmic theme"}
            >
              <span aria-hidden="true">{lightsOut ? "☼" : "✦"}</span>{lightsOut ? "Lights on" : "Lights out"}
            </button>
            <div className="avatar-mark">G</div>
          </div>
        </header>

        <div className={`page-content ${view === "overview" ? "immersive-page" : ""}`}>
          {view !== "overview" && (
            <div className="page-heading">
              <div><div className="eyebrow">STELLAR CATALOGUE / DATA RELEASE 3</div><h1>{title}<span className="heading-period">.</span></h1><p>{description[view]}</p></div>
              <button className="button-secondary refresh-button" onClick={() => void refreshData()} disabled={globalLoading}><Icon name="refresh" /> <span>{globalLoading ? "Loading" : "Refresh data"}</span></button>
            </div>
          )}

          {backendError && (
            <div className="notice-banner"><span className="notice-symbol">!</span><div><strong>Gaia API connection issue</strong><p>{backendError}. Check that the FastAPI Docker service is running on port 8000.</p></div><button onClick={() => setBackendError("")}>Dismiss</button></div>
          )}

          {view === "overview" && (
            <div className="stellar-landing">
              <header className="stellar-landing-nav">
                <button className="landing-brand" onClick={() => setView("overview")} aria-label="Gaia Stellar Explorer home">
                  <span className="landing-brand-mark"><Icon name="orbit" /></span>
                  <span className="landing-brand-copy"><strong>GAIA <i>/</i> STELLAR EXPLORER</strong><small>AN INTERACTIVE STELLAR ATLAS</small></span>
                </button>
                <nav className="landing-nav-links" aria-label="Explore the catalogue">
                  <button className="active" onClick={() => setView("overview")}>Discover</button>
                  <button onClick={() => setView("map")}>3D atlas</button>
                  <button onClick={() => setView("clusters")}>Clusters</button>
                  <button onClick={() => setView("lookup")}>Source archive</button>
                </nav>
                <div className="landing-nav-actions">
                  <span className={`landing-live ${backendOnline ? "is-live" : ""}`}><i />{backendOnline ? "DATA CONNECTED" : "CONNECTING"}</span>
                  <button
                    type="button"
                    className="theme-toggle"
                    onClick={() => setLightsOut((current) => !current)}
                    aria-pressed={lightsOut}
                    aria-label={lightsOut ? "Switch off Lights out theme" : "Switch on Lights out theme"}
                    title={lightsOut ? "Return to the standard theme" : "Switch to the deeper cosmic theme"}
                  >
                    <span aria-hidden="true">{lightsOut ? "☼" : "✦"}</span>{lightsOut ? "Lights on" : "Lights out"}
                  </button>
                  <button className="landing-model-link" onClick={() => setView("predict")}>Model lab <span>↗</span></button>
                </div>
              </header>

              <section className="stellar-intro">
                <div className="stellar-intro-copy">
                  <div className="landing-kicker"><i /> GAIA DATA RELEASE 3 <span>·</span> SELECTED STELLAR SAMPLE</div>
                  <h1>Somewhere <em>between</em><br />the stars.</h1>
                  <p>A closer look at the structure of our stellar neighbourhood. Explore 249,295 observations from a Gaia DR3 line of sight through spatial positions, dimensionality reduction and unsupervised clustering.</p>
                  <div className="landing-hero-actions">
                    <button className="landing-primary" onClick={() => setView("map")}>Enter the atlas <span>↗</span></button>
                    <button className="landing-secondary" onClick={() => setView("clusters")}>Discover cluster structure <span>→</span></button>
                  </div>
                  <div className="landing-statline">
                    <div><strong>{count(summary?.total_observations)}</strong><span>GAIA SOURCES</span></div>
                    <div><strong>{count(summary?.kmeans.cluster_count)}</strong><span>K-MEANS PARTITIONS</span></div>
                    <div><strong>{count(summary?.hdbscan.cluster_count_excluding_noise)}</strong><span>HDBSCAN GROUPS</span></div>
                  </div>
                </div>
                <aside className="landing-field-note">
                  <div className="field-note-top"><span>OBSERVATION WINDOW</span><span>FIELD / 01</span></div>
                  <div className="field-note-title">Orion–Taurus<br />line of sight</div>
                  <div className="field-note-coordinates">
                    <div><span>RIGHT ASCENSION</span><strong>50° — 70°</strong></div>
                    <div><span>DECLINATION</span><strong>−5° — +5°</strong></div>
                    <div><span>REFERENCE FRAME</span><strong>ICRS</strong></div>
                    <div><span>POSITION UNIT</span><strong>Parsecs</strong></div>
                  </div>
                  <div className="field-note-bottom"><span><i /> CURATED GAIA DR3 SAMPLE</span><span>01 / 01</span></div>
                </aside>
              </section>

              <section className="stellar-field-section">
                <div className="stellar-section-heading">
                  <div><div className="landing-kicker"><i /> THE STELLAR FIELD <span>·</span> LIVE CATALOGUE</div><h2>Move through the field.</h2><p>Rotate the spatial view, change the clustering lens and select a source to inspect its catalogue record.</p></div>
                  <button className="landing-text-link" onClick={() => setView("map")}>Open full atlas <span>↗</span></button>
                </div>
                <div className="landing-space-view">
                  <SpatialExplorer points={points} algorithm={algorithm} onAlgorithmChange={setAlgorithm} total={pointTotal} loaded={points.length} loadingFull={loadingFull} loadProgress={loadProgress} onLoadFull={() => void loadFullCatalogue()} onSelectPoint={(point) => void selectPoint(point)} />
                </div>
                <div className="field-disclaimer"><span>COORDINATE SPACE <strong>Heliocentric Cartesian · ICRS</strong></span><span>COLOUR <strong>{algorithm === "kmeans" ? "K-Means assignment" : "HDBSCAN assignment"}</strong></span><span>HDBSCAN −1 <strong>Noise / unassigned</strong></span></div>
              </section>

              <section className="stellar-discovery-section">
                <div className="stellar-section-heading discovery-heading"><div><div className="landing-kicker"><i /> CHOOSE YOUR WAY IN</div><h2>Explore the catalogue.</h2><p>Four ways to move from the stellar field into the underlying data and models.</p></div></div>
                <div className="stellar-discovery-grid">
                  <button className="stellar-discovery-card discovery-spatial" onClick={() => setView("map")}><span className="discovery-card-index">01 / THE ATLAS</span><span className="discovery-card-symbol">◌</span><h3>Spatial explorer</h3><p>Navigate the observed sample in heliocentric Cartesian space or switch to its separate UMAP projection.</p><span className="discovery-card-action">ENTER FIELD <b>↗</b></span></button>
                  <button className="stellar-discovery-card discovery-clusters" onClick={() => setView("clusters")}><span className="discovery-card-index">02 / STRUCTURE</span><span className="discovery-card-symbol">✳</span><h3>Cluster atlas</h3><p>Compare four K-Means partitions with 26 non-noise HDBSCAN groups and noise observations.</p><span className="discovery-card-action">EXPLORE GROUPS <b>↗</b></span></button>
                  <button className="stellar-discovery-card discovery-sources" onClick={() => setView("lookup")}><span className="discovery-card-index">03 / CATALOGUE</span><span className="discovery-card-symbol">⌕</span><h3>Source archive</h3><p>Retrieve an individual Gaia DR3 source and inspect its astrometric, photometric and model fields.</p><span className="discovery-card-action">FIND A SOURCE <b>↗</b></span></button>
                  <button className="stellar-discovery-card discovery-model" onClick={() => setView("predict")}><span className="discovery-card-index">04 / INFERENCE</span><span className="discovery-card-symbol">⟡</span><h3>Model laboratory</h3><p>Submit six engineered features for approximate assignment to the existing fitted clusters.</p><span className="discovery-card-action">RUN INFERENCE <b>↗</b></span></button>
                </div>
              </section>

              <footer className="stellar-landing-footer"><span>GAIA STELLAR EXPLORER <i /> DATA RELEASE 3</span><span>AN EXPLORATION INTERFACE FOR AN UNSUPERVISED CLUSTERING STUDY</span><span>NOT AN ASTROPHYSICAL CLASSIFICATION</span></footer>
            </div>
          )}

          {view === "map" && (
            <>
              <div className="map-context-bar"><div><span className="context-beacon" />{activeCluster ? <><strong>{activeCluster.label}</strong><span> · {count(activeCluster.star_count)} catalogue observations</span></> : <><strong>Full catalogue</strong><span> · {count(summary?.total_observations)} observations</span></>}</div><div className="map-mode-actions"><div className="segmented-control atlas-mode-switch"><button className={mapMode === "spatial" ? "segment active" : "segment"} onClick={() => setMapMode("spatial")}>Spatial · 3D</button><button className={mapMode === "umap" ? "segment active" : "segment"} onClick={() => setMapMode("umap")}>UMAP · 2D</button></div><button className="button-secondary compact" onClick={() => void openAllStars()}>Show all observations</button></div></div>
              {mapMode === "spatial" ? <SpatialExplorer points={points} algorithm={algorithm} onAlgorithmChange={setAlgorithm} total={pointTotal} loaded={points.length} loadingFull={loadingFull} loadProgress={loadProgress} onLoadFull={() => void loadFullCatalogue()} onSelectPoint={(point) => void selectPoint(point)} /> : <UmapExplorer points={points} algorithm={algorithm} onAlgorithmChange={setAlgorithm} total={pointTotal} loaded={points.length} loadingFull={loadingFull} loadProgress={loadProgress} onLoadFull={() => void loadFullCatalogue()} onSelectPoint={(point) => void selectPoint(point)} />}
              <div className="map-insight-grid"><div className="insight-tile"><span>{mapMode === "spatial" ? "COORDINATE FRAME" : "EMBEDDING METHOD"}</span><strong>{mapMode === "spatial" ? "ICRS · Cartesian 3D" : "UMAP · 2D"}</strong><small>{mapMode === "spatial" ? "Heliocentric X, Y, Z positions in parsecs" : "n_neighbors = 50 · min_dist = 0.1"}</small></div><div className="insight-tile"><span>POINT COLOUR</span><strong>{algorithm === "kmeans" ? "K-Means assignment" : "HDBSCAN assignment"}</strong><small>{algorithm === "kmeans" ? "Four algorithmic partitions" : "26 groups plus noise / unassigned"}</small></div><div className="insight-tile"><span>INTERACTION</span><strong>Explore individual sources</strong><small>{mapMode === "spatial" ? "Drag to rotate · scroll to zoom · click a star" : "Hover to inspect · click to open the source record"}</small></div></div>
            </>
          )}

          {view === "clusters" && (
            <>
              <div className="cluster-intro"><div><h2>Two ways to organise the field.</h2><p>K-Means partitions the standardized feature space into four groups. HDBSCAN identifies density-based groups in the UMAP representation and leaves low-density observations unassigned.</p></div><div className="segmented-control large"><button className={algorithm === "kmeans" ? "segment active" : "segment"} onClick={() => setAlgorithm("kmeans")}>K-Means <small>4 groups</small></button><button className={algorithm === "hdbscan" ? "segment active" : "segment"} onClick={() => setAlgorithm("hdbscan")}>HDBSCAN <small>26 groups</small></button></div></div>
              <div className="cluster-statline"><div><span>GROUPS DISPLAYED</span><strong>{visibleClusters.length}</strong></div><div><span>OBSERVATIONS</span><strong>{count(summary?.total_observations)}</strong></div><div><span>ALGORITHM</span><strong>{algorithm.toUpperCase()}</strong></div></div>
              <div className="cluster-grid">{visibleClusters.map((cluster, index) => <ClusterMiniCard key={cluster.cluster_id} cluster={cluster} algorithm={algorithm} onClick={() => void openCluster(cluster, algorithm)} />)}</div>
              <p className="scientific-caveat">Cluster labels are algorithmic assignments. They do not establish astrophysical identity, gravitational binding or a one-to-one correspondence between K-Means and HDBSCAN labels. HDBSCAN label −1 denotes noise / unassigned observations under the fitted density criteria.</p>
            </>
          )}

          {view === "lookup" && (
            <div className="lookup-layout">
              <section className="lookup-search-card"><div className="eyebrow">CATALOGUE QUERY</div><h2>Find a stellar source.</h2><p>Enter a Gaia DR3 source identifier to retrieve the saved astrometric, photometric, kinematic and clustering record.</p><form onSubmit={submitLookup} className="lookup-form"><label htmlFor="source-id">GAIA DR3 SOURCE ID</label><div className="search-input-wrap"><Icon name="search" /><input id="source-id" inputMode="numeric" value={starQuery} onChange={(event) => setStarQuery(event.target.value)} placeholder="e.g. 2964390723303808" /><button type="submit" disabled={lookupLoading}>{lookupLoading ? "Searching…" : "Search source"}</button></div></form><button className="text-action example-link" onClick={() => void loadExampleStar()}>Load a verified example source <span>→</span></button>{lookupError && <div className="inline-error">{lookupError}</div>}</section>
              {selectedStar ? <StarDetails star={selectedStar} /> : <section className="lookup-empty"><div className="empty-orbit">◎</div><strong>{lookupLoading ? "Querying the catalogue" : "Your source record will appear here"}</strong><span>Search a Gaia DR3 source ID or load the verified example.</span></section>}
            </div>
          )}

          {view === "predict" && (
            <div className="prediction-layout">
              <section className="prediction-form-card"><div className="eyebrow">EXISTING MODEL INFERENCE</div><h2>Assign a new observation.</h2><p>Supply the six engineered features in the same units and order used during model training. This endpoint assigns the observation to existing fitted clusters; it does not discover new clusters.</p><div className="prediction-actions"><button className="text-action" onClick={() => void fillExamplePrediction()}>Fill verified example <span>↓</span></button><span className="feature-order-note">6 FEATURES REQUIRED</span></div><form onSubmit={submitPrediction} className="feature-form"><div className="feature-grid">{INITIAL_FEATURES.map((feature) => <label className="feature-field" key={feature.key}><span>{feature.label}<i>{feature.unit}</i></span><input type="number" step="any" required value={predictionValues[feature.key] ?? ""} placeholder={feature.placeholder} onChange={(event) => setPredictionValues((current) => ({ ...current, [feature.key]: event.target.value }))} /></label>)}</div><button className="button-primary prediction-submit" type="submit" disabled={predictionLoading}>{predictionLoading ? "Running inference…" : "Run cluster assignment"} <span>↗</span></button></form>{predictionError && <div className="inline-error">{predictionError}</div>}</section>
              <section className="prediction-result-card"><div className="eyebrow">INFERENCE OUTPUT</div><h3>Assignment result</h3>{predictionResult ? <><div className="prediction-result-grid"><div className="prediction-result-metric"><span>K-MEANS</span><strong>{predictionResult.items[0]?.kmeans_cluster ?? "—"}</strong><small>Partition assignment</small></div><div className="prediction-result-metric"><span>HDBSCAN</span><strong>{predictionResult.items[0]?.hdbscan_cluster === -1 ? "Noise" : predictionResult.items[0]?.hdbscan_cluster ?? "—"}</strong><small>{predictionResult.items[0]?.hdbscan_label ?? "Density assignment"}</small></div></div><div className="result-strength"><div><span>HDBSCAN MEMBERSHIP STRENGTH</span><strong>{number(predictionResult.items[0]?.hdbscan_membership_strength, 4)}</strong></div><div className="strength-track"><i style={{ width: `${Math.max(0, Math.min(100, (predictionResult.items[0]?.hdbscan_membership_strength ?? 0) * 100))}%` }} /></div></div><div className="embedding-result"><span>NEW UMAP COORDINATE</span><strong>{number(predictionResult.items[0]?.umap_1, 4)} <i>/</i> {number(predictionResult.items[0]?.umap_2, 4)}</strong></div><p className="subtle-note">Approximate assignment to existing fitted clusters. Membership strength is not a calibrated probability.</p></> : <div className="prediction-placeholder"><span className="prediction-orbit">✳</span><strong>Awaiting an observation</strong><p>Complete the six feature fields or load the verified example to run inference against the saved models.</p></div>}</section>
            </div>
          )}

          <footer className="page-footer"><span>GAIA STELLAR EXPLORER <i /> AN INTERACTIVE VIEW OF A GAIA DR3 CLUSTERING PIPELINE</span><span>UNSUPERVISED LEARNING <i /> SIX-DIMENSIONAL INPUT SPACE</span></footer>
        </div>
      </main>
      {toast && <div className="toast-message"><span>✓</span>{toast}</div>}
    </div>
  );
}
