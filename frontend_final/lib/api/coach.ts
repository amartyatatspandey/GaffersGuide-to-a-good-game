import { getApiBaseUrl } from "@/lib/apiBase";

export interface CoachingAdviceItem {
  frame_idx: number;
  team: string;
  flaw: string;
  severity: string;
  evidence: string;
  matched_philosophy_author: string;
  fc25_player_roles: string[] | null;
  tactical_instruction: string | null;
  tactical_instruction_steps: string[];
  llm_error: string | null;
}

export interface CoachAdviceResponse {
  generated_at: string;
  pipeline: Record<string, unknown>;
  advice_items: CoachingAdviceItem[];
}

const _sleep = (ms: number): Promise<void> =>
  new Promise<void>((r) => {
    setTimeout(r, ms);
  });

function coachAdviceUrl(
  jobId: string,
  llmEngine: "local" | "cloud",
  skipLlm: boolean,
): string {
  const base = getApiBaseUrl();
  return `${base}/api/v1/coach/advice?${new URLSearchParams({
    job_id: jobId,
    llm_engine: llmEngine,
    skip_llm: skipLlm ? "true" : "false",
  }).toString()}`;
}

/**
 * Load job-scoped coaching advice from the FastAPI pipeline (report JSON + optional local LLM refresh).
 * Retries on HTTP 425 while the report is still being written.
 * If `llm_engine=local` and Ollama is unreachable (HTTP 424/500), retries once with `skip_llm=true`
 * so the timeline still loads from the on-disk report.
 */
export async function getCoachAdvice(
  jobId: string,
  llmEngine: "local" | "cloud" = "local",
): Promise<CoachAdviceResponse> {
  const maxAttempts = 36;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const res = await fetch(coachAdviceUrl(jobId, llmEngine, false));

    if (res.ok) {
      return (await res.json()) as CoachAdviceResponse;
    }

    const text = await res.text().catch(() => "");

    if (res.status === 425 && attempt < maxAttempts - 1) {
      await _sleep(Math.min(2000, 400 + attempt * 150));
      continue;
    }

    if (
      llmEngine === "local" &&
      (res.status === 424 || res.status === 500)
    ) {
      const resSkip = await fetch(coachAdviceUrl(jobId, llmEngine, true));
      if (resSkip.ok) {
        return (await resSkip.json()) as CoachAdviceResponse;
      }
    }

    throw new Error(
      res.status === 425
        ? "Coach report was not ready after waiting (HTTP 425)."
        : `Coach advice failed (HTTP ${res.status}): ${text.slice(0, 200)}`,
    );
  }

  throw new Error("Coach advice not available.");
}

export interface CoachChatResponse {
  reply: string;
}

export async function postCoachChat(params: {
  message: string;
  jobId: string | null;
  llmEngine: "local" | "cloud";
}): Promise<CoachChatResponse> {
  const base = getApiBaseUrl();
  const res = await fetch(`${base}/api/v1/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message: params.message,
      job_id: params.jobId,
      llm_engine: params.llmEngine,
    }),
  });

  const data = (await res.json().catch(() => ({}))) as CoachChatResponse & {
    detail?: unknown;
  };

  if (!res.ok) {
    const detail =
      typeof data.detail === "string"
        ? data.detail
        : JSON.stringify(data.detail ?? {}).slice(0, 400);
    throw new Error(`Chat failed (HTTP ${res.status}): ${detail}`);
  }

  return { reply: data.reply ?? "" };
}
