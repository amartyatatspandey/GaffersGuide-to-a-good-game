import { getApiBaseUrl, getAuthHeaders } from '@/lib/apiBase';
import { debugSessionLog } from '@/lib/debugSessionLog';
import type { TrackingPayload } from '@/lib/types/trackingTypes';

export interface JobResponse {
  job_id: string;
  status: string;
}

/** Local Modal-less dev: default `local` unless `NEXT_PUBLIC_CV_ENGINE=cloud`. */
const defaultCvEngine =
  process.env.NEXT_PUBLIC_CV_ENGINE === 'cloud' ? 'cloud' : 'local';

/** Default Llama/Ollama path unless user chose cloud in Engine Settings. */
export async function createJob(
  file: File,
  llmEngine: 'local' | 'cloud' = 'local',
  qualityProfile: string = 'balanced',
  chunkingInterval: string = '15-minute intervals',
): Promise<JobResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('cv_engine', defaultCvEngine);
  formData.append('llm_engine', llmEngine);
  formData.append('quality_profile', qualityProfile);
  formData.append('chunking_interval', chunkingInterval);

  const base = getApiBaseUrl();
  const res = await fetch(`${base}/api/v1/jobs`, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: formData,
  });

  // #region agent log
  debugSessionLog({
    sessionId: 'bb63ae',
    hypothesisId: 'H3-H4',
    location: 'jobs.ts:createJob',
    message: 'createJob response',
    data: {
      apiBaseHost: (() => {
        try {
          return new URL(base).host;
        } catch {
          return 'invalid-url';
        }
      })(),
      ok: res.ok,
      status: res.status,
      cv_engine: defaultCvEngine,
    },
  });
  // #endregion

  if (!res.ok) {
    throw new Error(`Failed to create job: ${res.statusText}`);
  }

  return res.json();
}

const _sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

/**
 * Fetch tracking JSON. Retries on HTTP 425 while the pipeline is still writing
 * `{job_id}_tracking_data.json` (see FastAPI `get_job_tracking`).
 */
export async function getTracking(jobId: string): Promise<TrackingPayload> {
  const base = getApiBaseUrl();
  const url = `${base}/api/v1/jobs/${jobId}/tracking`;
  const maxAttempts = 48;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const res = await fetch(url, { headers: getAuthHeaders() });
    // #region agent log
    debugSessionLog({
      sessionId: 'bb63ae',
      runId: 'post-fix',
      hypothesisId: 'H1-H3',
      location: 'jobs.ts:getTracking',
      message: 'getTracking response',
      data: {
        jobIdPrefix: jobId.slice(0, 8),
        attempt,
        ok: res.ok,
        status: res.status,
      },
    });
    // #endregion

    if (res.ok) {
      return (await res.json()) as TrackingPayload;
    }

    if (res.status === 425 && attempt < maxAttempts - 1) {
      await _sleep(Math.min(2000, 350 + attempt * 120));
      continue;
    }

    throw new Error(
      res.status === 425
        ? 'Tracking file was not ready after waiting for the pipeline (HTTP 425).'
        : `Tracking not ready or not found (HTTP ${res.status})`,
    );
  }

  throw new Error('Tracking not ready or not found');
}

export async function listReports(): Promise<any[]> {
  const base = getApiBaseUrl();
  const res = await fetch(`${base}/api/v1/elite/reports`, { headers: getAuthHeaders() });
  if (!res.ok) throw new Error('Failed to list reports');
  return res.json();
}

export async function getReport(reportId: string): Promise<any> {
  const base = getApiBaseUrl();
  const res = await fetch(`${base}/api/v1/elite/reports/${reportId}`, { headers: getAuthHeaders() });
  if (!res.ok) throw new Error('Failed to fetch report');
  return res.json();
}

export async function saveReport(reportData: any): Promise<{ status: string; filename: string }> {
  const base = getApiBaseUrl();
  const res = await fetch(`${base}/api/v1/elite/reports/save`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
    body: JSON.stringify(reportData),
  });
  if (!res.ok) throw new Error('Failed to save report');
  return res.json();
}

export async function deleteReport(reportId: string): Promise<{ status: string }> {
  const base = getApiBaseUrl();
  const res = await fetch(`${base}/api/v1/elite/reports/${reportId}`, {
    method: 'DELETE',
    headers: getAuthHeaders(),
  });
  if (!res.ok) throw new Error('Failed to delete report');
  return res.json();
}

export async function listMatches(search?: string, sort?: string): Promise<any[]> {
  const base = getApiBaseUrl();
  const searchParam = search ? `search=${encodeURIComponent(search)}` : '';
  const sortParam = sort ? `sort=${sort}` : '';
  const params = [searchParam, sortParam].filter(Boolean).join('&');
  const url = `${base}/api/v1/matches${params ? `?${params}` : ''}`;
  
  const res = await fetch(url, { headers: getAuthHeaders() });
  if (!res.ok) throw new Error('Failed to fetch matches');
  return res.json();
}

export async function deleteMatch(matchId: string): Promise<{ status: string }> {
  const base = getApiBaseUrl();
  const res = await fetch(`${base}/api/v1/matches/${matchId}`, {
    method: 'DELETE',
    headers: getAuthHeaders(),
  });
  if (!res.ok) throw new Error('Failed to delete match');
  return res.json();
}

export async function reanalyzeMatch(matchId: string): Promise<{ status: string; job_id: string }> {
  const base = getApiBaseUrl();
  const res = await fetch(`${base}/api/v1/matches/${matchId}/reanalyze`, {
    method: 'POST',
    headers: getAuthHeaders(),
  });
  if (!res.ok) throw new Error('Failed to start reanalysis');
  return res.json();
}


export async function getJobEvents(jobId: string): Promise<any> {
  const base = getApiBaseUrl();
  const res = await fetch(`${base}/api/v1/jobs/${jobId}/events`, {
    headers: getAuthHeaders(),
  });
  if (!res.ok) throw new Error('Failed to fetch job events');
  return res.json();
}

export async function getTacticalTimeline(jobId: string): Promise<any[]> {
  const base = getApiBaseUrl();
  const res = await fetch(`${base}/api/v1/jobs/${jobId}/timeline`, {
    headers: getAuthHeaders(),
  });
  if (!res.ok) throw new Error('Failed to fetch tactical timeline');
  return res.json();
}

