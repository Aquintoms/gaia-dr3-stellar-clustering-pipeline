"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { PointerEvent } from "react";
import type { Algorithm, StarPoint } from "@/lib/api";

const KMEANS_COLORS = ["#9b8cff", "#45d6c5", "#ffb86b", "#ff7fa7"];
const HDBSCAN_COLORS = [
  "#74d9cf", "#a995ff", "#f4b86a", "#f184ad", "#79b7ff", "#d1a0ff",
  "#9fd477", "#e88b74", "#70c7e8", "#dfca75", "#9c9eea", "#e79acb",
  "#85d1a0", "#efaa7f", "#78a6db", "#c5a0e8", "#a7c56e", "#e47d8e",
  "#70c2b7", "#dca76e", "#8eaae8", "#c18bd6", "#7fcca0", "#e28d68",
  "#8bbbe0", "#c2b2f2",
];

function colorFor(point: StarPoint, algorithm: Algorithm) {
  if (algorithm === "kmeans") return KMEANS_COLORS[((point.kmeans_label % 4) + 4) % 4];
  if (point.hdbscan_label === -1) return "#667386";
  return HDBSCAN_COLORS[((point.hdbscan_label % HDBSCAN_COLORS.length) + HDBSCAN_COLORS.length) % HDBSCAN_COLORS.length];
}

function compactNumber(value: number) {
  return Number(value).toFixed(1);
}

interface UmapExplorerProps {
  points: StarPoint[];
  algorithm: Algorithm;
  onAlgorithmChange: (algorithm: Algorithm) => void;
  total: number;
  loaded: number;
  loadingFull: boolean;
  loadProgress: number;
  onLoadFull: () => void;
  onSelectPoint?: (point: StarPoint) => void;
}

interface HoveredPoint {
  point: StarPoint;
  x: number;
  y: number;
}

export default function UmapExplorer({
  points,
  algorithm,
  onAlgorithmChange,
  total,
  loaded,
  loadingFull,
  loadProgress,
  onLoadFull,
  onSelectPoint,
}: UmapExplorerProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null);
  const geometryRef = useRef({ left: 54, top: 20, plotWidth: 0, plotHeight: 0, minX: -1, maxX: 1, minY: -1, maxY: 1 });
  const [size, setSize] = useState({ width: 900, height: 470 });
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [hovered, setHovered] = useState<HoveredPoint | null>(null);

  const bounds = useMemo(() => {
    if (!points.length) return { minX: -10, maxX: 10, minY: -10, maxY: 10 };
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const point of points) {
      if (Number.isFinite(point.umap_1) && Number.isFinite(point.umap_2)) {
        minX = Math.min(minX, point.umap_1);
        maxX = Math.max(maxX, point.umap_1);
        minY = Math.min(minY, point.umap_2);
        maxY = Math.max(maxY, point.umap_2);
      }
    }
    if (!Number.isFinite(minX)) return { minX: -10, maxX: 10, minY: -10, maxY: 10 };
    const padX = Math.max((maxX - minX) * 0.035, 0.5);
    const padY = Math.max((maxY - minY) * 0.06, 0.5);
    return { minX: minX - padX, maxX: maxX + padX, minY: minY - padY, maxY: maxY + padY };
  }, [points]);

  useEffect(() => {
    const wrapper = wrapperRef.current;
    if (!wrapper) return;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) setSize({ width: Math.max(320, entry.contentRect.width), height: Math.max(300, entry.contentRect.height) });
    });
    observer.observe(wrapper);
    return () => observer.disconnect();
  }, []);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.floor(size.width * dpr);
    canvas.height = Math.floor(size.height * dpr);
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, size.width, size.height);

    const left = 58;
    const right = 22;
    const top = 22;
    const bottom = 46;
    const plotWidth = Math.max(1, size.width - left - right);
    const plotHeight = Math.max(1, size.height - top - bottom);
    geometryRef.current = { left, top, plotWidth, plotHeight, ...bounds };

    context.fillStyle = "#0a111c";
    context.fillRect(0, 0, size.width, size.height);
    context.save();
    context.strokeStyle = "rgba(143, 163, 190, 0.10)";
    context.lineWidth = 1;
    context.font = "10px var(--font-geist-mono), monospace";
    context.fillStyle = "#66758a";
    for (let i = 0; i <= 5; i++) {
      const x = left + (plotWidth * i) / 5;
      const y = top + (plotHeight * i) / 5;
      context.beginPath(); context.moveTo(x, top); context.lineTo(x, top + plotHeight); context.stroke();
      context.beginPath(); context.moveTo(left, y); context.lineTo(left + plotWidth, y); context.stroke();
      const xValue = bounds.minX + ((bounds.maxX - bounds.minX) * i) / 5;
      const yValue = bounds.maxY - ((bounds.maxY - bounds.minY) * i) / 5;
      context.textAlign = "center";
      context.fillText(compactNumber(xValue), x, top + plotHeight + 18);
      context.textAlign = "right";
      context.fillText(compactNumber(yValue), left - 10, y + 3);
    }
    context.fillStyle = "#8593a8";
    context.font = "10px var(--font-geist-mono), monospace";
    context.textAlign = "center";
    context.fillText("UMAP 1", left + plotWidth / 2, size.height - 8);
    context.save();
    context.translate(14, top + plotHeight / 2);
    context.rotate(-Math.PI / 2);
    context.fillText("UMAP 2", 0, 0);
    context.restore();

    context.beginPath();
    context.rect(left, top, plotWidth, plotHeight);
    context.clip();
    const centerX = left + plotWidth / 2;
    const centerY = top + plotHeight / 2;
    for (const point of points) {
      if (!Number.isFinite(point.umap_1) || !Number.isFinite(point.umap_2)) continue;
      const baseX = left + ((point.umap_1 - bounds.minX) / (bounds.maxX - bounds.minX)) * plotWidth;
      const baseY = top + (1 - (point.umap_2 - bounds.minY) / (bounds.maxY - bounds.minY)) * plotHeight;
      const x = centerX + (baseX - centerX) * zoom + pan.x;
      const y = centerY + (baseY - centerY) * zoom + pan.y;
      context.fillStyle = colorFor(point, algorithm);
      context.globalAlpha = algorithm === "hdbscan" && point.hdbscan_label === -1 ? 0.42 : 0.82;
      context.fillRect(x, y, points.length > 50000 ? 1.6 : 2.4, points.length > 50000 ? 1.6 : 2.4);
    }
    context.globalAlpha = 1;
    context.restore();
  }, [algorithm, bounds, pan, points, size, zoom]);

  useEffect(() => { draw(); }, [draw]);

  const findNearest = (mouseX: number, mouseY: number) => {
    const g = geometryRef.current;
    const centerX = g.left + g.plotWidth / 2;
    const centerY = g.top + g.plotHeight / 2;
    let nearest: StarPoint | null = null;
    let nearestDistance = 12 * 12;
    let nearestX = 0;
    let nearestY = 0;
    for (const point of points) {
      if (!Number.isFinite(point.umap_1) || !Number.isFinite(point.umap_2)) continue;
      const baseX = g.left + ((point.umap_1 - g.minX) / (g.maxX - g.minX)) * g.plotWidth;
      const baseY = g.top + (1 - (point.umap_2 - g.minY) / (g.maxY - g.minY)) * g.plotHeight;
      const x = centerX + (baseX - centerX) * zoom + pan.x;
      const y = centerY + (baseY - centerY) * zoom + pan.y;
      const distance = (x - mouseX) ** 2 + (y - mouseY) ** 2;
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearest = point;
        nearestX = x;
        nearestY = y;
      }
    }
    return nearest ? { point: nearest, x: nearestX, y: nearestY } : null;
  };

  const handlePointerDown = (event: PointerEvent<HTMLCanvasElement>) => {
    if (event.button !== 0) return;
    dragRef.current = { x: event.clientX, y: event.clientY, panX: pan.x, panY: pan.y };
    event.currentTarget.setPointerCapture(event.pointerId);
    setHovered(null);
  };

  const handlePointerMove = (event: PointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    if (drag) {
      setPan({ x: drag.panX + event.clientX - drag.x, y: drag.panY + event.clientY - drag.y });
      return;
    }
    const rect = event.currentTarget.getBoundingClientRect();
    const found = findNearest(event.clientX - rect.left, event.clientY - rect.top);
    if (found) setHovered({ ...found, x: event.clientX - rect.left, y: event.clientY - rect.top });
    else setHovered(null);
  };

  const handlePointerUp = (event: PointerEvent<HTMLCanvasElement>) => {
    if (!dragRef.current) return;
    const moved = Math.abs(event.clientX - dragRef.current.x) + Math.abs(event.clientY - dragRef.current.y) > 4;
    dragRef.current = null;
    if (!moved) {
      const rect = event.currentTarget.getBoundingClientRect();
      const found = findNearest(event.clientX - rect.left, event.clientY - rect.top);
      if (found) onSelectPoint?.(found.point);
    }
  };

  return (
    <section className="plot-card">
      <div className="plot-toolbar">
        <div>
          <div className="eyebrow">DIMENSIONALITY REDUCTION</div>
          <h3 className="section-title">UMAP stellar projection</h3>
          <p className="section-subtitle">Explore the two-dimensional embedding of the six-feature stellar representation.</p>
        </div>
        <div className="segmented-control" role="group" aria-label="Colour points by clustering algorithm">
          <button className={algorithm === "kmeans" ? "segment active" : "segment"} onClick={() => onAlgorithmChange("kmeans")}>K-Means</button>
          <button className={algorithm === "hdbscan" ? "segment active" : "segment"} onClick={() => onAlgorithmChange("hdbscan")}>HDBSCAN</button>
        </div>
      </div>

      <div className="plot-stage">
        <div className="plot-floating-label"><span className="live-dot" /> {algorithm === "kmeans" ? "FOUR PARTITIONS" : "DENSITY-BASED GROUPS"}</div>
        <div className="plot-controls">
          <button aria-label="Zoom in" onClick={() => setZoom((value) => Math.min(8, value * 1.25))}>+</button>
          <button aria-label="Zoom out" onClick={() => setZoom((value) => Math.max(0.7, value / 1.25))}>−</button>
          <button aria-label="Reset view" onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}>⌂</button>
        </div>
        <div className="plot-canvas-wrap" ref={wrapperRef}>
          {points.length ? (
            <canvas
              ref={canvasRef}
              className="umap-canvas"
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerCancel={() => { dragRef.current = null; }}
              onPointerLeave={() => { if (!dragRef.current) setHovered(null); }}
              onWheel={(event) => { event.preventDefault(); setZoom((value) => Math.max(0.7, Math.min(8, value * (event.deltaY < 0 ? 1.12 : 0.89)))); }}
            />
          ) : (
            <div className="plot-empty"><span className="empty-orbit">◎</span><strong>Waiting for the Gaia API</strong><span>Connect the local FastAPI service to load the stellar projection.</span></div>
          )}
          {hovered && (
            <div className="plot-tooltip" style={{ left: Math.min(hovered.x + 14, size.width - 220), top: Math.max(hovered.y - 82, 12) }}>
              <div className="tooltip-kicker">GAIA SOURCE</div>
              <strong>{hovered.point.source_id}</strong>
              <div className="tooltip-row"><span>{algorithm === "kmeans" ? "K-Means" : "HDBSCAN"}</span><b>{algorithm === "kmeans" ? hovered.point.kmeans_label : hovered.point.hdbscan_label === -1 ? "Noise" : hovered.point.hdbscan_label}</b></div>
              <div className="tooltip-row"><span>BP − RP</span><b>{hovered.point.bp_rp?.toFixed(3) ?? "—"}</b></div>
              <div className="tooltip-row"><span>UMAP</span><b>{hovered.point.umap_1.toFixed(2)}, {hovered.point.umap_2.toFixed(2)}</b></div>
              <span className="tooltip-hint">Click point for star details</span>
            </div>
          )}
        </div>
        <div className="plot-footer">
          <span>{loaded.toLocaleString()} / {total.toLocaleString()} observations loaded</span>
          <span>Scroll to zoom · Drag to pan · Hover for details</span>
        </div>
      </div>

      <div className="plot-bottom">
        <div className="plot-legend">
          {algorithm === "kmeans" ? (
            KMEANS_COLORS.map((color, index) => <span className="legend-item" key={index}><i style={{ background: color }} />Cluster {index}</span>)
          ) : (
            <><span className="legend-item"><i style={{ background: HDBSCAN_COLORS[0] }} />Density clusters</span><span className="legend-item"><i style={{ background: "#667386" }} />Noise / unassigned</span></>
          )}
        </div>
        <div className="plot-load-action">
          {loadingFull ? (
            <span className="load-progress">Loading catalogue… {Math.round(loadProgress)}%</span>
          ) : loaded < total ? (
            <button className="button-secondary compact" onClick={onLoadFull}>Load full catalogue <span>↗</span></button>
          ) : (
            <span className="all-loaded"><i /> Full catalogue loaded</span>
          )}
        </div>
      </div>
      <p className="plot-note">UMAP is a two-dimensional representation used for exploration and HDBSCAN clustering. Apparent proximity in this projection should not, by itself, be interpreted as physical proximity or gravitational binding.</p>
    </section>
  );
}
