import { useEffect, useRef } from "react";
import { normalizePitchMeters, projectMetersToCanvas } from "@/lib/trackingAdapter";
import type { TrackingFrame } from "@/lib/types/trackingTypes";
import { drawPitchBackground } from "./pitch";

interface RadarCanvasProps {
  frame: TrackingFrame | undefined;
  width?: number;
  height?: number;
  padding?: number;
}

const TEAM_0_COLOR = "#3b82f6"; // blue
const TEAM_1_COLOR = "#ef4444"; // red
const UNKNOWN_COLOR = "#9ca3af";
const BALL_COLOR = "#fde047";
const BALL_STROKE = "#111827";

export default function RadarCanvas({
  frame,
  width = 800,
  height = 520,
  padding = 20,
}: RadarCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    drawPitchBackground(ctx, width, height, padding);

    if (!frame) return;

    for (const player of frame.players) {
      const normalized = normalizePitchMeters(player.x_pitch, player.y_pitch);
      if (!normalized) continue;

      const { xPx, yPx } = projectMetersToCanvas(
        normalized.xM,
        normalized.yM,
        width,
        height,
        padding,
      );
      ctx.fillStyle =
        player.team_id === "team_0"
          ? TEAM_0_COLOR
          : player.team_id === "team_1"
            ? TEAM_1_COLOR
            : UNKNOWN_COLOR;
      ctx.beginPath();
      ctx.arc(xPx, yPx, 5, 0, Math.PI * 2);
      ctx.fill();
    }

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
        ctx.strokeStyle = BALL_STROKE;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(xPx, yPx, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    }
  }, [frame, width, height, padding]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="w-full rounded border border-gray-700 bg-[#111a12]"
    />
  );
}

