import type { TrackingFrame, TrackingPayload } from "@/lib/types/trackingTypes";

export const PITCH_LENGTH_M = 105;
export const PITCH_WIDTH_M = 68;
const LEGACY_SCALE = 10;

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * Accept both modern meter coordinates and legacy 10x radar units.
 */
export function normalizePitchMeters(
  xPitch: number | null,
  yPitch: number | null,
): { xM: number; yM: number } | null {
  if (xPitch == null || yPitch == null) return null;
  
  let xM = xPitch;
  let yM = yPitch;
  
  // If coordinates look like they are in legacy 10x units (e.g. 525 instead of 52.5)
  if (Math.abs(xM) > PITCH_LENGTH_M + 5 || Math.abs(yM) > PITCH_WIDTH_M + 5) {
    xM = xM / LEGACY_SCALE;
    yM = yM / LEGACY_SCALE;
  }

  // No longer shifting centered coordinates as it caused regressions
  // with standard 0-105m tracking data.

  return {
    xM: clamp(xM, 0, PITCH_LENGTH_M),
    yM: clamp(yM, 0, PITCH_WIDTH_M),
  };
}

export function projectMetersToCanvas(
  xM: number,
  yM: number,
  widthPx: number,
  heightPx: number,
  paddingPx: number,
): { xPx: number; yPx: number } {
  const innerW = Math.max(1, widthPx - paddingPx * 2);
  const innerH = Math.max(1, heightPx - paddingPx * 2);
  const xPx = paddingPx + (clamp(xM, 0, PITCH_LENGTH_M) / PITCH_LENGTH_M) * innerW;
  const yPx = paddingPx + (clamp(yM, 0, PITCH_WIDTH_M) / PITCH_WIDTH_M) * innerH;
  return { xPx, yPx };
}

export interface FrameLookup {
  totalFrames: number;
  getFrameByIndex: (frameIndex: number) => TrackingFrame | undefined;
}

export function buildFrameLookup(payload: TrackingPayload): FrameLookup {
  const frames = payload.frames ?? [];
  const frameByIdx = new Map<number, TrackingFrame>();
  for (const frame of frames) frameByIdx.set(frame.frame_idx, frame);
  const totalFrames = frames.length;
  return {
    totalFrames,
    getFrameByIndex: (frameIndex: number) =>
      frameByIdx.get(frameIndex) ?? frames[frameIndex],
  };
}

/** Convert API payload from radar pixel space to meter space for the canvas pipeline. */
export function adaptTrackingPayload(payload: TrackingPayload): TrackingPayload {
  const t = payload.telemetry;
  if (t?.player_position_space !== "radar_pixels") {
    return payload;
  }
  const ppm = t.radar_pixels_per_meter ?? 10;
  if (!Number.isFinite(ppm) || ppm <= 0) {
    return payload;
  }
  const frames = (payload.frames ?? []).map((fr) => ({
    ...fr,
    players: fr.players.map((p) => ({
      ...p,
      x_pitch:
        p.x_pitch != null && p.y_pitch != null ? p.x_pitch / ppm : p.x_pitch,
      y_pitch:
        p.x_pitch != null && p.y_pitch != null ? p.y_pitch / ppm : p.y_pitch,
    })),
    ball_xy:
      fr.ball_xy != null
        ? ([fr.ball_xy[0] / ppm, fr.ball_xy[1] / ppm] as [number, number])
        : null,
  }));
  return {
    ...payload,
    telemetry: {
      ...t,
      player_position_space: "meters",
    },
    frames,
  };
}
