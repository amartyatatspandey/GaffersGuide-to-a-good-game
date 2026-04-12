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

// #region agent log
function _coachDebug(
  hypothesisId: string,
  location: string,
  message: string,
  data: Record<string, unknown>,
): void {
  fetch("http://127.0.0.1:7265/ingest/b94af6c0-0f3f-4385-ab39-095f9a480704", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Debug-Session-Id": "bb63ae",
    },
    body: JSON.stringify({
      sessionId: "bb63ae",
      hypothesisId,
      location,
      message,
      data,
      timestamp: Date.now(),
    }),
  }).catch(() => {});
}
// #endregion

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

async function _coachFetch(url: string): Promise<Response> {
  try {
    return await fetch(url);
  } catch (e) {
    const inner = e instanceof Error ? e.message : String(e);
    // #region agent log
    _coachDebug("A", "coach.ts:_coachFetch", "fetch_rejected", {
      inner,
      urlHost: (() => {
        try {
          return new URL(url).host;
        } catch {
          return "bad-url";
        }
      })(),
    });
    // #endregion
    const base = getApiBaseUrl();
    throw new Error(
      `Coach request could not reach the API (${inner}). ` +
        `Confirm FastAPI is running at ${base}. ` +
        `Safari/WebKit often reports this as "Load failed" when the connection drops or times out ` +
        `(a long local-Ollama refresh on /coach/advice can cause that).`,
    );
  }
}

/**
 * Load job-scoped coaching advice from the FastAPI pipeline (report JSON + optional local LLM refresh).
 * Polls with `skip_llm=true` so HTTP 425 wait loops do not block on Ollama.
 * For `llm_engine=local`, attempts one enrichment request (`skip_llm=false`); if it fails or times out,
 * returns the fast disk-backed response so the timeline still renders.
 */
export async function getCoachAdvice(
  jobId: string,
  llmEngine: "local" | "cloud" = "local",
): Promise<CoachAdviceResponse> {
  const maxAttempts = 36;
  // #region agent log
  _coachDebug("E", "coach.ts:getCoachAdvice", "enter", {
    jobIdPrefix: jobId.slice(0, 8),
    llmEngine,
  });
  // #endregion

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const res = await _coachFetch(coachAdviceUrl(jobId, llmEngine, true));
    // #region agent log
    _coachDebug("D", "coach.ts:getCoachAdvice", "poll_skip_llm_response", {
      attempt,
      status: res.status,
      ok: res.ok,
    });
    // #endregion

    if (res.ok) {
      let fast: CoachAdviceResponse;
      try {
        fast = (await res.json()) as CoachAdviceResponse;
      } catch (parseErr) {
        // #region agent log
        _coachDebug("E", "coach.ts:getCoachAdvice", "poll_json_parse_failed", {
          err: parseErr instanceof Error ? parseErr.message : String(parseErr),
        });
        // #endregion
        throw new Error(
          "Coach advice response was not valid JSON (check API base URL and that FastAPI served this request).",
        );
      }
      if (llmEngine !== "local") {
        return fast;
      }
      const items = fast.advice_items ?? [];
      const fullyHydrated =
        items.length > 0 &&
        items.every(
          (it) =>
            typeof it.tactical_instruction === "string" &&
            it.tactical_instruction.trim().length > 0,
        );
      // #region agent log
      _coachDebug("C", "coach.ts:getCoachAdvice", "enrich_gate", {
        itemCount: items.length,
        fullyHydrated,
        skipEnrichReason:
          items.length === 0 ? "empty_report" : fullyHydrated ? "already_hydrated" : "needs_enrich",
      });
      // #endregion
      if (items.length === 0 || fullyHydrated) {
        return fast;
      }
      try {
        // #region agent log
        _coachDebug("A", "coach.ts:getCoachAdvice", "enrich_fetch_start", {
          adviceLen: items.length,
        });
        // #endregion
        const richRes = await _coachFetch(coachAdviceUrl(jobId, "local", false));
        // #region agent log
        _coachDebug("B", "coach.ts:getCoachAdvice", "enrich_fetch_response", {
          status: richRes.status,
          ok: richRes.ok,
        });
        // #endregion
        if (richRes.ok) {
          try {
            return (await richRes.json()) as CoachAdviceResponse;
          } catch (parseRich) {
            // #region agent log
            _coachDebug("E", "coach.ts:getCoachAdvice", "enrich_json_parse_failed", {
              err: parseRich instanceof Error ? parseRich.message : String(parseRich),
            });
            // #endregion
            return fast;
          }
        }
        if (richRes.status === 424 || richRes.status === 500) {
          const resSkip = await _coachFetch(coachAdviceUrl(jobId, "local", true));
          if (resSkip.ok) {
            return (await resSkip.json()) as CoachAdviceResponse;
          }
        }
        return fast;
      } catch (enrichErr) {
        // #region agent log
        _coachDebug("E", "coach.ts:getCoachAdvice", "enrich_catch_fallback_fast", {
          err: enrichErr instanceof Error ? enrichErr.message : String(enrichErr),
        });
        // #endregion
        return fast;
      }
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
      const resSkip = await _coachFetch(coachAdviceUrl(jobId, llmEngine, true));
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
