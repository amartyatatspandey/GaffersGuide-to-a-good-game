export interface TrackingPlayer {
  id: number | null;
  team_id: "team_0" | "team_1" | null;
  x_pitch: number | null;
  y_pitch: number | null;
  x_canvas: number;
  y_canvas: number;
}

export interface TrackingFrame {
  frame_idx: number;
  players: TrackingPlayer[];
  ball_xy: [number, number] | null;
  used_optical_flow_fallback: boolean;
}

export interface TrackingPayload {
  telemetry: {
    total_frames_processed: number;
    frames_optical_flow_fallback: number;
  };
  frames: TrackingFrame[];
}

