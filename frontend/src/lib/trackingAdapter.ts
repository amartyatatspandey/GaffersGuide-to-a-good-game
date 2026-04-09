import type { TrackingFrame, TrackingPayload } from "./trackingTypes";

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
  if (xM > PITCH_LENGTH_M + 1 || yM > PITCH_WIDTH_M + 1) {
    xM = xM / LEGACY_SCALE;
    yM = yM / LEGACY_SCALE;
  }
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

