import { getApiBaseUrl, getAuthHeaders } from '@/lib/apiBase';
import type { FixtureSearchResult, FixtureLineup } from '@/types/lineup';

/**
 * Football data service (API-Football via our own backend).
 * Calls our `/api/v1/fixtures/*` endpoints — never API-Football directly.
 * Follows the same fetch pattern as `lib/api/jobs.ts`.
 *
 * All three functions:
 *   • wrap network errors (TypeError from fetch) with a friendly message
 *   • extract the backend's `detail` field from non-2xx JSON bodies
 *     so FastAPI's human-readable error messages reach the UI intact
 */

const UNREACHABLE = 'Football data service unreachable. Check your connection or retry.';

async function _safeJson(res: Response): Promise<{ detail?: string }> {
  return res.json().catch(() => ({}));
}

export async function searchFixtures(query: string): Promise<FixtureSearchResult[]> {
  const url = `${getApiBaseUrl()}/api/v1/fixtures/search?q=${encodeURIComponent(query)}`;
  let res: Response;
  try {
    res = await fetch(url, { headers: getAuthHeaders() });
  } catch {
    throw new Error(UNREACHABLE);
  }
  if (!res.ok) {
    const body = await _safeJson(res);
    throw new Error(body.detail ?? `Failed to search fixtures (${res.status}).`);
  }
  return res.json();
}

/**
 * Fetch fixture metadata by ID (competition, date, venue, teams).
 * Returns the same shape as FixtureSearchResult.
 * Useful for a confirmation panel that shows fixture details without a full lineup fetch.
 */
export async function getFixtureDetail(fixtureId: string): Promise<FixtureSearchResult> {
  const url = `${getApiBaseUrl()}/api/v1/fixtures/${encodeURIComponent(fixtureId)}`;
  let res: Response;
  try {
    res = await fetch(url, { headers: getAuthHeaders() });
  } catch {
    throw new Error(UNREACHABLE);
  }
  if (!res.ok) {
    const body = await _safeJson(res);
    throw new Error(body.detail ?? `Failed to fetch fixture (${res.status}).`);
  }
  return res.json();
}

/**
 * Fetch official starting lineup and bench for both teams.
 * Throws with the backend's `detail` message on non-2xx responses
 * (lineup not yet released, plan restriction, fixture not found, etc.).
 */
export async function getFixtureLineup(fixtureId: string): Promise<FixtureLineup> {
  const url = `${getApiBaseUrl()}/api/v1/fixtures/${encodeURIComponent(fixtureId)}/lineup`;
  let res: Response;
  try {
    res = await fetch(url, { headers: getAuthHeaders() });
  } catch {
    throw new Error(UNREACHABLE);
  }
  if (!res.ok) {
    const body = await _safeJson(res);
    throw new Error(body.detail ?? res.statusText);
  }
  return res.json();
}
