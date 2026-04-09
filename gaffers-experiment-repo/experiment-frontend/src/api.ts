export type DecoderMode = "opencv" | "pyav";
export type RuntimeTarget = "nvidia" | "apple_mps" | "cpu_fallback";
export type HardwareProfile = "l4" | "a10" | "a100" | "mps" | "cpu";
export type QualityMode = "fast" | "balanced" | "high";
export type ChunkingPolicy = "none" | "fixed" | "auto";
export type SlaTier = "tier_10m" | "tier_5m";
export type JobStatus = "pending" | "processing" | "done" | "error" | string;
export type JobTelemetry = {
  queue_wait_ms?: number;
  decode_ms?: number;
  infer_ms?: number;
  post_ms?: number;
  frames_processed?: number;
  effective_fps?: number;
  reid_invocations?: number;
  reid_ms?: number;
  id_switch_rate?: number;
};

const API_BASE = (import.meta.env.VITE_EXP_API_URL ?? "http://127.0.0.1:8100").replace(
  /\/$/,
  ""
);
const WS_BASE = (import.meta.env.VITE_EXP_WS_URL ?? "ws://127.0.0.1:8100").replace(/\/$/, "");

export function wsJobUrl(jobId: string): string {
  return `${WS_BASE}/ws/exp/jobs/${encodeURIComponent(jobId)}`;
}

export async function uploadJob(
  file: File,
  decoderMode: DecoderMode,
  options: {
    runtimeTarget: RuntimeTarget;
    hardwareProfile: HardwareProfile;
    qualityMode: QualityMode;
    chunkingPolicy: ChunkingPolicy;
    maxParallelChunks: number;
    targetSlaTier: SlaTier;
  }
): Promise<{ job_id: string }> {
  const form = new FormData();
  form.append("file", file, file.name);
  form.append("cv_engine", options.runtimeTarget === "nvidia" ? "cloud" : "local");
  form.append("llm_engine", "local");
  form.append("decoder_mode", decoderMode);
  form.append("runtime_target", options.runtimeTarget);
  form.append("hardware_profile", options.hardwareProfile);
  form.append("quality_mode", options.qualityMode);
  form.append("chunking_policy", options.chunkingPolicy);
  form.append("max_parallel_chunks", String(options.maxParallelChunks));
  form.append("target_sla_tier", options.targetSlaTier);
  form.append("idempotency_key", `${file.name}-${file.size}-${crypto.randomUUID()}`);
  const res = await fetch(`${API_BASE}/api/exp/jobs`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
  return (await res.json()) as { job_id: string };
}

export type JobState = {
  job_id: string;
  status: JobStatus;
  current_step: string;
  stage?: string;
  runtime_target?: RuntimeTarget;
  hardware_profile?: HardwareProfile;
  chunks?: Array<{ chunk_id: string; start_frame: number; end_frame: number }>;
  telemetry?: JobTelemetry;
  error?: string | null;
};

export async function fetchJobState(jobId: string): Promise<JobState> {
  const res = await fetch(`${API_BASE}/api/exp/jobs/${encodeURIComponent(jobId)}`);
  if (!res.ok) throw new Error(`Job state failed: ${res.status}`);
  return (await res.json()) as JobState;
}

export async function fetchAdvice(jobId: string): Promise<unknown> {
  const res = await fetch(`${API_BASE}/api/exp/coach/advice?job_id=${encodeURIComponent(jobId)}`);
  if (!res.ok) throw new Error(`Advice failed: ${res.status}`);
  return await res.json();
}

export async function fetchTracking(jobId: string): Promise<unknown> {
  const res = await fetch(`${API_BASE}/api/exp/jobs/${encodeURIComponent(jobId)}/tracking`);
  if (!res.ok) throw new Error(`Tracking failed: ${res.status}`);
  return await res.json();
}

export async function sendChat(jobId: string, message: string): Promise<{ reply: string }> {
  const res = await fetch(`${API_BASE}/api/exp/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_id: jobId, message }),
  });
  if (!res.ok) throw new Error(`Chat failed: ${res.status}`);
  return (await res.json()) as { reply: string };
}

export async function fetchReports(): Promise<unknown> {
  const res = await fetch(`${API_BASE}/api/exp/reports`);
  if (!res.ok) throw new Error(`Reports failed: ${res.status}`);
  return await res.json();
}

export async function fetchMetrics(): Promise<unknown> {
  const res = await fetch(`${API_BASE}/api/exp/metrics`);
  if (!res.ok) throw new Error(`Metrics failed: ${res.status}`);
  return await res.json();
}
