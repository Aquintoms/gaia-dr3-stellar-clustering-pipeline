"use client";

import { useEffect, useRef } from "react";
import type { Algorithm, StarPoint } from "@/lib/api";

const KMEANS = ["#9c8cff", "#54d8c5", "#f0bd78", "#f08cac"];
const HDBSCAN = ["#72d8ce", "#aa98ff", "#f2bd78", "#ed8caf", "#79b8ff", "#d6a4fa", "#a5d77c", "#e88b75", "#6fc8e8", "#dfcc7c"];

function pointColor(point: StarPoint, algorithm: Algorithm) {
  if (algorithm === "kmeans") return KMEANS[((point.kmeans_label % 4) + 4) % 4];
  if (point.hdbscan_label === -1) return "#748093";
  return HDBSCAN[((point.hdbscan_label % HDBSCAN.length) + HDBSCAN.length) % HDBSCAN.length];
}

export default function StarfieldPreview({ points, algorithm }: { points: StarPoint[]; algorithm: Algorithm }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const hostRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const host = hostRef.current;
    const canvas = canvasRef.current;
    if (!host || !canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    let width = 0;
    let height = 0;
    const draw = () => {
      const rect = host.getBoundingClientRect();
      width = Math.max(280, rect.width);
      height = Math.max(230, rect.height);
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      context.clearRect(0, 0, width, height);
      const glow = context.createRadialGradient(width * .55, height * .48, 4, width * .55, height * .48, width * .55);
      glow.addColorStop(0, "rgba(86, 111, 166, .20)");
      glow.addColorStop(.55, "rgba(62, 79, 126, .08)");
      glow.addColorStop(1, "rgba(6, 10, 18, 0)");
      context.fillStyle = glow;
      context.fillRect(0, 0, width, height);

      const finite = points.filter((p) => Number.isFinite(p.umap_1) && Number.isFinite(p.umap_2));
      if (!finite.length) {
        // Quiet decorative field while the API is connecting; never presented as catalogue data.
        for (let i = 0; i < 95; i++) {
          const x = ((i * 67.37) % 997) / 997 * width;
          const y = ((i * 37.91) % 991) / 991 * height;
          context.fillStyle = `rgba(190, 205, 230, ${.18 + (i % 5) * .08})`;
          context.beginPath(); context.arc(x, y, i % 11 === 0 ? 1.5 : .7, 0, Math.PI * 2); context.fill();
        }
        return;
      }

      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (const p of finite) {
        minX = Math.min(minX, p.umap_1!); maxX = Math.max(maxX, p.umap_1!);
        minY = Math.min(minY, p.umap_2!); maxY = Math.max(maxY, p.umap_2!);
      }
      const rangeX = Math.max(maxX - minX, 1);
      const rangeY = Math.max(maxY - minY, 1);
      const pad = 24;
      const scale = Math.min((width - pad * 2) / rangeX, (height - pad * 2) / rangeY);
      const offsetX = (width - rangeX * scale) / 2;
      const offsetY = (height - rangeY * scale) / 2;

      // The glow is built from real UMAP observations returned by the Gaia API.
      for (let i = 0; i < finite.length; i++) {
        const point = finite[i];
        const x = offsetX + (point.umap_1! - minX) * scale;
        const y = height - offsetY - (point.umap_2! - minY) * scale;
        const r = finite.length > 50000 ? 1.0 : 1.45;
        context.globalAlpha = algorithm === "hdbscan" && point.hdbscan_label === -1 ? .30 : .76;
        context.fillStyle = pointColor(point, algorithm);
        context.fillRect(x, y, r, r);
      }
      context.globalAlpha = 1;
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(host);
    return () => observer.disconnect();
  }, [points, algorithm]);

  return (
    <div className="hero-field" ref={hostRef} aria-label="UMAP field preview drawn from Gaia catalogue observations">
      <canvas ref={canvasRef} className="hero-field-canvas" />
      <div className="hero-field-vignette" />
      <div className="hero-field-caption"><span className="hero-field-beacon" /> LIVE CATALOGUE FIELD <i /> UMAP / 2D</div>
      <div className="hero-field-coordinate">RA 50°—70° <i /> DEC −5°—+5°</div>
    </div>
  );
}
