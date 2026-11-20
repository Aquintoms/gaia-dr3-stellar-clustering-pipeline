"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { PointerEvent, WheelEvent } from "react";
import type { Algorithm, StarPoint } from "@/lib/api";

const KMEANS = ["#9d8cff", "#4fd7c4", "#f1bd77", "#ee8cab"];
const HDBSCAN = ["#70d8ce", "#a997ff", "#f2bd78", "#ee8bad", "#7cb7ff", "#d4a2f8", "#a4d77a", "#e88d76", "#70c7e7", "#e0cb78", "#a3a4ed", "#e79bc9", "#86d1a1", "#efaa7f", "#78a7dc", "#c6a1e8", "#a7c56e", "#e47e8e", "#70c2b8", "#dca76e", "#8eaae8", "#c18bd6", "#7fcca0", "#e28d68", "#8bbbe0", "#c2b2f2"];

type Projected = { point: StarPoint; x: number; y: number; depth: number };

function colorFor(point: StarPoint, algorithm: Algorithm) {
  if (algorithm === "kmeans") return KMEANS[((point.kmeans_label % 4) + 4) % 4];
  if (point.hdbscan_label === -1) return "#69778a";
  return HDBSCAN[((point.hdbscan_label % HDBSCAN.length) + HDBSCAN.length) % HDBSCAN.length];
}

function fmt(value: number) {
  return value.toLocaleString(undefined, { maximumFractionDigits: 1 });
}

interface SpatialExplorerProps {
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

export default function SpatialExplorer({ points, algorithm, onAlgorithmChange, total, loaded, loadingFull, loadProgress, onLoadFull, onSelectPoint }: SpatialExplorerProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dragRef = useRef<{ x: number; y: number; yaw: number; pitch: number; moved: boolean } | null>(null);
  const renderedRef = useRef<Projected[]>([]);
  const [size, setSize] = useState({ width: 1000, height: 520 });
  const [yaw, setYaw] = useState(.42);
  const [pitch, setPitch] = useState(-.22);
  const [zoom, setZoom] = useState(1);
  const [hovered, setHovered] = useState<{ item: Projected; x: number; y: number } | null>(null);

  const geometry = useMemo(() => {
    const valid = points.filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.z));
    if (!valid.length) return { valid, cx: 0, cy: 0, cz: 0, radius: 1 };
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (const p of valid) {
      minX = Math.min(minX, p.x!); maxX = Math.max(maxX, p.x!);
      minY = Math.min(minY, p.y!); maxY = Math.max(maxY, p.y!);
      minZ = Math.min(minZ, p.z!); maxZ = Math.max(maxZ, p.z!);
    }
    const span = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1);
    return { valid, cx: (minX + maxX) / 2, cy: (minY + maxY) / 2, cz: (minZ + maxZ) / 2, radius: span / 2 };
  }, [points]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const observer = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (rect) setSize({ width: Math.max(320, rect.width), height: Math.max(320, rect.height) });
    });
    observer.observe(host);
    return () => observer.disconnect();
  }, []);

  const project = useCallback((x: number, y: number, z: number) => {
    const nx = (x - geometry.cx) / geometry.radius;
    const ny = (y - geometry.cy) / geometry.radius;
    const nz = (z - geometry.cz) / geometry.radius;
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    const cp = Math.cos(pitch), sp = Math.sin(pitch);
    const rx = nx * cy - nz * sy;
    const rz = nx * sy + nz * cy;
    const ry = ny * cp - rz * sp;
    const depth = ny * sp + rz * cp;
    const perspective = 2.7 / (2.7 + depth * .42);
    const scale = Math.min(size.width * .43, size.height * .43) * zoom;
    return { x: size.width * .52 + rx * scale * perspective, y: size.height * .49 - ry * scale * perspective, depth };
  }, [geometry, pitch, size.height, size.width, yaw, zoom]);

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

    const background = context.createRadialGradient(size.width * .52, size.height * .47, 8, size.width * .52, size.height * .47, size.width * .65);
    background.addColorStop(0, "#111d2d");
    background.addColorStop(.6, "#09121e");
    background.addColorStop(1, "#060b12");
    context.fillStyle = background;
    context.fillRect(0, 0, size.width, size.height);

    // Faint reference axes use the same heliocentric Cartesian coordinates as the catalogue.
    const axisLength = geometry.radius * 1.28;
    const origin = project(geometry.cx, geometry.cy, geometry.cz);
    const axisDefs = [
      { end: project(geometry.cx + axisLength, geometry.cy, geometry.cz), label: "X · pc", color: "rgba(110, 207, 212, .52)" },
      { end: project(geometry.cx, geometry.cy + axisLength, geometry.cz), label: "Y · pc", color: "rgba(176, 154, 255, .56)" },
      { end: project(geometry.cx, geometry.cy, geometry.cz + axisLength), label: "Z · pc", color: "rgba(236, 184, 126, .54)" },
    ];
    for (const axis of axisDefs) {
      context.beginPath(); context.moveTo(origin.x, origin.y); context.lineTo(axis.end.x, axis.end.y);
      context.strokeStyle = axis.color; context.lineWidth = 1; context.stroke();
      context.fillStyle = axis.color; context.font = "10px var(--font-geist-mono), monospace";
      context.fillText(axis.label, axis.end.x + 6, axis.end.y - 5);
    }

    // A faint projected reference ring gives the cloud a spatial frame, not an astronomical orbit claim.
    context.beginPath();
    context.ellipse(size.width * .52, size.height * .49, size.width * .31 * zoom, size.height * .13 * zoom, -.12 + yaw * .14, 0, Math.PI * 2);
    context.strokeStyle = "rgba(132, 160, 194, .10)"; context.setLineDash([3, 8]); context.stroke(); context.setLineDash([]);

    const projected: Projected[] = [];
    context.save();
    context.beginPath(); context.rect(0, 0, size.width, size.height); context.clip();
    for (const point of geometry.valid) {
      const item = { point, ...project(point.x!, point.y!, point.z!) };
      projected.push(item);
      const alpha = Math.max(.24, Math.min(.94, .78 - item.depth * .18));
      context.globalAlpha = algorithm === "hdbscan" && item.point.hdbscan_label === -1 ? alpha * .36 : alpha;
      context.fillStyle = colorFor(item.point, algorithm);
      const radius = geometry.valid.length > 50000 ? 1.05 : 1.55;
      context.beginPath(); context.arc(item.x, item.y, radius, 0, Math.PI * 2); context.fill();
    }
    context.restore();
    context.globalAlpha = 1;
    renderedRef.current = projected;

    const vignette = context.createRadialGradient(size.width * .52, size.height * .48, size.width * .13, size.width * .52, size.height * .48, size.width * .65);
    vignette.addColorStop(.55, "rgba(3, 7, 13, 0)"); vignette.addColorStop(1, "rgba(3, 7, 13, .38)");
    context.fillStyle = vignette; context.fillRect(0, 0, size.width, size.height);
  }, [algorithm, geometry, project, size.height, size.width, yaw]);

  useEffect(() => { draw(); }, [draw]);

  const nearest = (x: number, y: number) => {
    let best: Projected | null = null;
    let distance = 15 * 15;
    const rendered = renderedRef.current;
    const stride = Math.max(1, Math.ceil(rendered.length / 60000));
    for (let index = 0; index < rendered.length; index += stride) {
      const item = rendered[index];
      const d = (item.x - x) ** 2 + (item.y - y) ** 2;
      if (d < distance) { distance = d; best = item; }
    }
    return best;
  };

  const onPointerDown = (event: PointerEvent<HTMLCanvasElement>) => {
    if (event.button !== 0) return;
    dragRef.current = { x: event.clientX, y: event.clientY, yaw, pitch, moved: false };
    event.currentTarget.setPointerCapture(event.pointerId);
    setHovered(null);
  };

  const onPointerMove = (event: PointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left, y = event.clientY - rect.top;
    if (drag) {
      const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
      setYaw(drag.yaw + dx * .008);
      setPitch(Math.max(-1.15, Math.min(1.15, drag.pitch + dy * .006)));
      return;
    }
    const found = nearest(x, y);
    setHovered(found ? { item: found, x, y } : null);
  };

  const onPointerUp = (event: PointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    dragRef.current = null;
    if (!drag || drag.moved) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const found = nearest(event.clientX - rect.left, event.clientY - rect.top);
    if (found) onSelectPoint?.(found.point);
  };

  const onWheel = (event: WheelEvent<HTMLCanvasElement>) => {
    event.preventDefault();
    setZoom((current) => Math.max(.55, Math.min(3.2, current * (event.deltaY < 0 ? 1.09 : .92))));
  };

  const reset = () => { setYaw(.42); setPitch(-.22); setZoom(1); setHovered(null); };
  const validCount = geometry.valid.length;

  return (
    <section className="spatial-card">
      <div className="spatial-toolbar">
        <div className="spatial-title-block"><span className="eyebrow">HELIOCENTRIC CARTESIAN FIELD · ICRS</span><strong>Three-dimensional stellar positions</strong><small>Physical catalogue coordinates · parsecs</small></div>
        <div className="spatial-toolbar-actions">
          <div className="segmented-control spatial-algorithm-switch">
            <button className={algorithm === "kmeans" ? "segment active" : "segment"} onClick={() => onAlgorithmChange("kmeans")}>K-Means</button>
            <button className={algorithm === "hdbscan" ? "segment active" : "segment"} onClick={() => onAlgorithmChange("hdbscan")}>HDBSCAN</button>
          </div>
          <button className="spatial-reset" onClick={reset} title="Reset view">Reset view</button>
        </div>
      </div>
      <div className="spatial-stage" ref={hostRef}>
        <canvas ref={canvasRef} className="spatial-canvas" onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerCancel={() => { dragRef.current = null; }} onWheel={onWheel} onPointerLeave={() => { if (!dragRef.current) setHovered(null); }} aria-label="Interactive 3D plot of Gaia DR3 heliocentric Cartesian stellar positions. Drag to rotate, scroll to zoom, click a point to inspect its source." />
        <div className="spatial-stage-label"><span className="hero-field-beacon" /> {validCount.toLocaleString()} POINTS RENDERED <i /> {algorithm.toUpperCase()}</div>
        <div className="spatial-stage-controls"><button onClick={() => setZoom((z) => Math.min(3.2, z * 1.18))} aria-label="Zoom in">+</button><button onClick={() => setZoom((z) => Math.max(.55, z / 1.18))} aria-label="Zoom out">−</button></div>
        {hovered && <div className="spatial-tooltip" style={{ left: Math.min(Math.max(12, hovered.x + 16), size.width - 210), top: Math.max(12, hovered.y - 12) }}><span>GAIA DR3 SOURCE</span><strong>{hovered.item.point.source_id}</strong><small>K-Means {hovered.item.point.kmeans_label} · HDBSCAN {hovered.item.point.hdbscan_label === -1 ? "noise" : hovered.item.point.hdbscan_label}</small><small>Click to inspect record</small></div>}
        {!validCount && <div className="spatial-empty"><strong>Waiting for spatial coordinates</strong><span>Refresh the catalogue or check the API connection.</span></div>}
      </div>
      <div className="spatial-footer">
        <div className="spatial-legend"><span><i className="legend-dot legend-violet"/> X axis</span><span><i className="legend-dot legend-mint"/> Y axis</span><span><i className="legend-dot legend-amber"/> Z axis</span><span><i className="legend-dot legend-muted"/> HDBSCAN noise</span></div>
        <div className="spatial-counts"><strong>{loaded.toLocaleString()}</strong> / {total.toLocaleString()} observations</div>
      </div>
      <div className="spatial-actions">
        <span>DRAG TO ROTATE <i/> SCROLL TO ZOOM <i/> CLICK A STAR TO INSPECT</span>
        {loaded < total ? <button className="button-secondary compact" onClick={onLoadFull} disabled={loadingFull}>{loadingFull ? `Loading ${Math.round(loadProgress)}%…` : "Load full catalogue ↗"}</button> : <span className="spatial-complete">FULL CATALOGUE LOADED</span>}
      </div>
      {loadingFull && <div className="spatial-progress"><i style={{ width: `${loadProgress}%` }} /></div>}
      <p className="spatial-science-note">Axes show heliocentric Cartesian positions in the ICRS frame. Point colours represent algorithmic cluster assignments; HDBSCAN label −1 denotes noise / unassigned observations, not a physical stellar class.</p>
    </section>
  );
}
