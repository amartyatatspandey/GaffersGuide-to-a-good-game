"use client";
import { useEffect, useRef, useState } from "react";
import { normalizePitchMeters, projectMetersToCanvas } from "@/lib/trackingAdapter";
import type { TrackingFrame } from "@/lib/types/trackingTypes";
import { drawPitchBackground } from "./pitch";
import type { JobPlayerMappings } from "@/lib/playerMappingUtils";

interface RadarCanvasProps {
  frame: TrackingFrame | undefined;
  savedMappings?: JobPlayerMappings | null;
}

const TEAM_0_COLOR = "#3b82f6"; // Blue
const TEAM_1_COLOR = "#ef4444"; // Red
const UNKNOWN_COLOR = "#6b7280";
const BALL_COLOR = "#ffffff";

export default function RadarCanvas({
  frame,
  savedMappings = null,
}: RadarCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 518 });

  useEffect(() => {
    const updateSize = () => {
      if (!containerRef.current) return;
      const { width } = containerRef.current.getBoundingClientRect();
      // Restore standard pitch aspect ratio 105:68
      const height = Math.floor((width * 68) / 105);
      setDimensions({ width, height });
    };

    updateSize();
    window.addEventListener("resize", updateSize);
    return () => window.removeEventListener("resize", updateSize);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = dimensions.width * dpr;
    canvas.height = dimensions.height * dpr;
    ctx.scale(dpr, dpr);

    const { width, height } = dimensions;
    const padding = Math.max(12, width * 0.04); 

    drawPitchBackground(ctx, width, height, padding);

    if (!frame) return;

    // Draw Players
    for (const player of frame.players) {
      if (player.id === 75 || player.id === 76 || player.id === 79) continue;
      const normalized = normalizePitchMeters(player.x_pitch, player.y_pitch);
      if (!normalized) continue;

      const { xPx, yPx } = projectMetersToCanvas(
        normalized.xM,
        normalized.yM,
        width,
        height,
        padding,
      );

      const color =
        player.team_id === "team_0"
          ? TEAM_0_COLOR
          : player.team_id === "team_1"
            ? TEAM_1_COLOR
            : UNKNOWN_COLOR;

      ctx.beginPath();
      ctx.arc(xPx, yPx, width * 0.008, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      
      ctx.strokeStyle = "white";
      ctx.lineWidth = 1;
      ctx.stroke();

      // Mapped label: jersey number if available, otherwise raw ID
      const mappedEntry = savedMappings?.mappings?.[String(player.id)];
      const label = mappedEntry ? `#${mappedEntry.number}` : (player.id ? player.id.toString() : "");
      ctx.fillStyle = "white";
      ctx.font = `bold ${Math.max(8, width * 0.012)}px Inter, sans-serif`;
      ctx.textAlign = "center";
      ctx.fillText(label, xPx, yPx - (width * 0.015));
    }

    // Draw Ball
    if (frame.ball_xy) {
      const [bx, by] = frame.ball_xy;
      const normalized = normalizePitchMeters(bx, by);
      if (normalized) {
        const { xPx, yPx } = projectMetersToCanvas(
          normalized.xM,
          normalized.yM,
          width,
          height,
          padding,
        );

        ctx.fillStyle = BALL_COLOR;
        ctx.beginPath();
        ctx.arc(xPx, yPx, width * 0.006, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "black";
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }
  }, [frame, dimensions, savedMappings]);

  return (
    <div ref={containerRef} className="w-full">
      <canvas
        ref={canvasRef}
        style={{
          width: "100%",
          height: "auto",
          display: "block",
          aspectRatio: "105 / 68"
        }}
        className="rounded-lg border border-gray-800 bg-[#061a0d] shadow-lg"
      />
    </div>
  );
}
