import { getApiBaseUrl } from '@/lib/apiBase';
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
): Promise<JobResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('cv_engine', defaultCvEngine);
  formData.append('llm_engine', llmEngine);

  const base = getApiBaseUrl();
  const res = await fetch(`${base}/api/v1/jobs`, {
    method: 'POST',
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
    const res = await fetch(url);
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
