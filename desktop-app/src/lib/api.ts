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
  report_path?: string | null;
  tracking_overlay_path?: string | null;
  tracking_data_path?: string | null;
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

export async function createBetaJob(file: File): Promise<CreateJobResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("cv_engine", "cloud");
  formData.append("llm_engine", "cloud");
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

export async function getBetaArtifacts(jobId: string): Promise<JobArtifactsResponse> {
  const response = await fetch(`${getApiBaseUrl()}/api/v1beta/jobs/${jobId}/artifacts`);
  return parseResponse<JobArtifactsResponse>(response);
}

export async function getReports(): Promise<ReportsResponse> {
  const response = await fetch(`${getApiBaseUrl()}/api/v1/reports`);
  return parseResponse<ReportsResponse>(response);
}

export async function getCoachAdvice(jobId: string): Promise<CoachAdviceResponse> {
  const url = new URL(`${getApiBaseUrl()}/api/v1/coach/advice`);
  url.searchParams.set("job_id", jobId);
  const response = await fetch(url.toString());
  return parseResponse<CoachAdviceResponse>(response);
}

export async function sendChat(jobId: string, message: string): Promise<ChatResponse> {
  const response = await fetch(`${getApiBaseUrl()}/api/v1/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      job_id: jobId,
      message,
    }),
  });
  return parseResponse<ChatResponse>(response);
}
