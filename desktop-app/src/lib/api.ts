export type JobStatus = "pending" | "processing" | "done" | "error";

export interface CreateJobResponse {
  job_id: string;
  status: JobStatus;
  cv_engine: "local" | "cloud";
  llm_engine: "local" | "cloud";
}

export interface JobProgressMessage {
  job_id: string;
  status: JobStatus;
  current_step: string;
  result_path?: string | null;
  tracking_overlay_path?: string | null;
  tracking_data_path?: string | null;
  error?: string | null;
}

export interface JobArtifactsResponse {
  job_id: string;
  status: JobStatus;
  report_path: string | null;
  tracking_overlay_path: string | null;
  tracking_data_path: string | null;
  report_state: "ready" | "not_ready";
  tracking_state: "ready" | "not_ready";
  overlay_state: "ready" | "not_ready" | "unavailable";
  overlay_reason?: string | null;
}

export interface BetaJobResponse {
  job_id: string;
  status: JobStatus;
  current_step: string;
  result_path: string | null;
  tracking_overlay_path: string | null;
  tracking_data_path: string | null;
  error: string | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface ReportEntry {
  job_id: string;
  created_at: string;
  report_filename: string;
}

export interface ReportsResponse {
  reports: ReportEntry[];
}

export interface CoachingAdviceItem {
  frame_idx: number;
  team: string;
  flaw: string;
  severity: string;
  evidence: string;
  matched_philosophy_author: string;
  fc25_player_roles?: string[] | null;
  tactical_instruction?: string | null;
  tactical_instruction_steps: string[];
  llm_error?: string | null;
}

export interface CoachAdviceResponse {
  generated_at: string;
  pipeline: Record<string, string>;
  advice_items: CoachingAdviceItem[];
}

export interface ChatResponse {
  reply: string;
}

const DEFAULT_API_BASE = "http://127.0.0.1:8000";

export function getApiBaseUrl(): string {
  // Prefer runtime config injected by Electron preload (works in packaged builds
  // where NEXT_PUBLIC_* is fixed at Next build time).
  if (
    typeof window !== "undefined" &&
    (window as typeof window & { desktopConfig?: { backendUrl?: string } }).desktopConfig
      ?.backendUrl
  ) {
    return (window as typeof window & { desktopConfig: { backendUrl: string } }).desktopConfig
      .backendUrl;
  }
  return process.env.NEXT_PUBLIC_BACKEND_URL ?? DEFAULT_API_BASE;
}

function getWsBaseUrl(apiBase: string): string {
  if (apiBase.startsWith("https://")) {
    return apiBase.replace("https://", "wss://");
  }
  if (apiBase.startsWith("http://")) {
    return apiBase.replace("http://", "ws://");
  }
  return `ws://${apiBase}`;
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Request failed (${response.status}): ${detail}`);
  }
  return (await response.json()) as T;
}

export interface CreateBetaJobOptions {
  cvEngine?: "local" | "cloud";
  llmEngine?: "local" | "cloud";
}

export async function createBetaJob(
  file: File,
  options: CreateBetaJobOptions = {},
): Promise<CreateJobResponse> {
  const { cvEngine = "local", llmEngine = "local" } = options;
  const formData = new FormData();
  formData.append("file", file);
  formData.append("cv_engine", cvEngine);
  formData.append("llm_engine", llmEngine);
  formData.append("idempotency_key", `desktop-${Date.now()}-${file.name}`);

  const response = await fetch(`${getApiBaseUrl()}/api/v1beta/jobs`, {
    method: "POST",
    body: formData,
  });
  return parseResponse<CreateJobResponse>(response);
}

export function subscribeJobProgress(
  jobId: string,
  onMessage: (message: JobProgressMessage) => void,
  onError: (error: Error) => void,
): () => void {
  const ws = new WebSocket(`${getWsBaseUrl(getApiBaseUrl())}/ws/v1beta/jobs/${jobId}`);
  ws.onmessage = (event: MessageEvent<string>) => {
    try {
      onMessage(JSON.parse(event.data) as JobProgressMessage);
    } catch (error) {
      onError(new Error(`Failed to parse websocket message: ${String(error)}`));
    }
  };
  ws.onerror = () => {
    onError(new Error("Job progress websocket failed."));
  };

  return () => {
    if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
      ws.close();
    }
  };
}

export async function getBetaJob(jobId: string): Promise<BetaJobResponse> {
  const response = await fetch(`${getApiBaseUrl()}/api/v1beta/jobs/${jobId}`);
  return parseResponse<BetaJobResponse>(response);
}

export async function getBetaArtifacts(jobId: string): Promise<JobArtifactsResponse> {
  const response = await fetch(`${getApiBaseUrl()}/api/v1beta/jobs/${jobId}/artifacts`);
  return parseResponse<JobArtifactsResponse>(response);
}

export async function getReports(): Promise<ReportsResponse> {
  const response = await fetch(`${getApiBaseUrl()}/api/v1/reports`);
  return parseResponse<ReportsResponse>(response);
}

export async function getCoachAdvice(
  jobId: string,
  options: { llmEngine?: "local" | "cloud" } = {},
): Promise<CoachAdviceResponse> {
  const url = new URL(`${getApiBaseUrl()}/api/v1/coach/advice`);
  url.searchParams.set("job_id", jobId);
  if (options.llmEngine) {
    url.searchParams.set("llm_engine", options.llmEngine);
  }
  const response = await fetch(url.toString());
  return parseResponse<CoachAdviceResponse>(response);
}

export async function sendChat(
  jobId: string,
  message: string,
  llmEngine?: "local" | "cloud",
): Promise<ChatResponse> {
  const response = await fetch(`${getApiBaseUrl()}/api/v1/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      job_id: jobId,
      message,
      llm_engine: llmEngine ?? null,
    }),
  });
  return parseResponse<ChatResponse>(response);
}

export interface BackendHealthResult {
  reachable: boolean;
  status?: string;
}

export interface LocalLlmPreflightResponse {
  configured_base_url: string;
  configured_model: string;
  daemon_reachable: boolean;
  model_present: boolean;
  generation_ok: boolean;
  error?: string | null;
  hint?: string | null;
}

export interface TrackingPlayerPoint {
  x_pitch?: number | null;
  y_pitch?: number | null;
  team_id?: number | null;
}

export interface TrackingFrame {
  frame_idx: number;
  players: TrackingPlayerPoint[];
}

export interface TrackingDataResponse {
  telemetry?: Record<string, unknown>;
  frames: TrackingFrame[];
}

export async function checkBackendHealth(): Promise<BackendHealthResult> {
  try {
    const response = await fetch(`${getApiBaseUrl()}/health`, { signal: AbortSignal.timeout(4000) });
    if (!response.ok) {
      return { reachable: false };
    }
    const body = (await response.json()) as { status?: string };
    return { reachable: true, status: body.status };
  } catch {
    return { reachable: false };
  }
}

export async function getBetaJobOverlay(jobId: string): Promise<string> {
  return `${getApiBaseUrl()}/api/v1beta/jobs/${jobId}/overlay`;
}

export function getBetaSourceVideoUrl(jobId: string): string {
  return `${getApiBaseUrl()}/api/v1beta/jobs/${jobId}/source-video`;
}

export async function getRecentAnalysis(): Promise<ReportsResponse> {
  return getReports();
}

export async function getBetaTracking(jobId: string): Promise<TrackingDataResponse> {
  const response = await fetch(`${getApiBaseUrl()}/api/v1beta/jobs/${jobId}/tracking`);
  return parseResponse<TrackingDataResponse>(response);
}

export async function getLocalLlmPreflight(): Promise<LocalLlmPreflightResponse> {
  const response = await fetch(`${getApiBaseUrl()}/api/v1/llm/local/preflight`);
  return parseResponse<LocalLlmPreflightResponse>(response);
}
