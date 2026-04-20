/**
 * NDJSON append via same-origin `/api/debug-log` (writes in dev; route no-ops in production).
 */
const RECENT_EVENT_MS = 2000;
const recentEvents = new Map<string, number>();

function buildEventKey(entry: Record<string, unknown>): string {
  const location = typeof entry.location === 'string' ? entry.location : 'unknown';
  const message = typeof entry.message === 'string' ? entry.message : 'unknown';
  const data = entry.data;
  let dataKey = '';
  if (data && typeof data === 'object') {
    const maybeData = data as Record<string, unknown>;
    const status = typeof maybeData.status === 'string' ? maybeData.status : '';
    const currentStep =
      typeof maybeData.current_step === 'string' ? maybeData.current_step : '';
    const host = typeof maybeData.wsHost === 'string' ? maybeData.wsHost : '';
    const apiHost =
      typeof maybeData.apiBaseHost === 'string' ? maybeData.apiBaseHost : '';
    dataKey = `${status}|${currentStep}|${host}|${apiHost}`;
  }
  return `${location}|${message}|${dataKey}`;
}

export function debugSessionLog(entry: Record<string, unknown>): void {
  const key = buildEventKey(entry);
  const now = Date.now();
  const last = recentEvents.get(key) ?? 0;
  if (now - last < RECENT_EVENT_MS) {
    return;
  }
  recentEvents.set(key, now);
  const payload = JSON.stringify({ ...entry, timestamp: Date.now() });
  fetch('/api/debug-log', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: payload,
  }).catch(() => {});
}
