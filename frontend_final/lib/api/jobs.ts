import { resolveApiEndpoints } from '@/lib/apiBase';
import { debugSessionLog } from '@/lib/debugSessionLog';
import type { TrackingPayload } from '@/lib/types/trackingTypes';

export interface JobResponse {
  job_id: string;
  status: string;
}

type ParsedFetchError = {
  message: string;
  code: string | null;
};

async function readFetchErrorMessage(res: Response): Promise<ParsedFetchError> {
  const text = await res.text();
  if (!text.trim()) {
    return {
      message: res.statusText || `HTTP ${res.status}`,
      code: null,
    };
  }
  try {
    const parsed = JSON.parse(text) as { code?: unknown; detail?: unknown };
    const d = parsed.detail;
    const topLevelCode = typeof parsed.code === 'string' ? parsed.code : null;
    const detailCode =
      d && typeof d === 'object' && 'code' in d && typeof (d as { code?: unknown }).code === 'string'
        ? ((d as { code?: string }).code ?? null)
        : null;
    const code = detailCode ?? topLevelCode;
    if (typeof d === 'string') {
      return { message: d, code };
    }
    if (d && typeof d === 'object' && 'message' in d) {
      const m = (d as { message?: unknown }).message;
      if (typeof m === 'string' && m.length > 0) {
        return { message: m, code };
      }
    }
  } catch {
    /* not JSON */
  }
  return {
    message: text.length > 600 ? `${text.slice(0, 600)}…` : text,
    code: null,
  };
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

  const { httpBase, host, source } = resolveApiEndpoints();
  const res = await fetch(`${httpBase}/api/v1/jobs`, {
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
      apiBaseHost: host,
      apiBaseSource: source,
      ok: res.ok,
      status: res.status,
      cv_engine: defaultCvEngine,
    },
  });
  // #endregion

  if (!res.ok) {
    const detail = await readFetchErrorMessage(res);
    debugSessionLog({
      sessionId: 'bb63ae',
      hypothesisId: 'H3-H4',
      location: 'jobs.ts:createJob',
      message: 'createJob error',
      data: {
        status: res.status,
        code: detail.code,
        apiBaseHost: host,
      },
    });
    const codeSuffix = detail.code ? ` [${detail.code}]` : '';
    throw new Error(
      `Failed to create job (${res.status}${codeSuffix}): ${detail.message}`,
    );
  }

  return res.json() as Promise<JobResponse>;
}

const _sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

/**
 * Fetch tracking JSON. Retries on HTTP 425 while the pipeline is still writing
 * `{job_id}_tracking_data.json` (see FastAPI `get_job_tracking`).
 */
export async function getTracking(jobId: string): Promise<TrackingPayload> {
  const { httpBase, host } = resolveApiEndpoints();
  const url = `${httpBase}/api/v1/jobs/${jobId}/tracking`;
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
        apiBaseHost: host,
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

    const detail = await readFetchErrorMessage(res);
    throw new Error(
      res.status === 425
        ? 'Tracking file was not ready after waiting for the pipeline (HTTP 425).'
        : `Tracking not ready or not found (HTTP ${res.status}${detail.code ? ` [${detail.code}]` : ''}): ${detail.message}`,
    );
  }

  throw new Error('Tracking not ready or not found');
}
