/**
 * NDJSON append via same-origin `/api/debug-log` (writes in dev; route no-ops in production).
 */
export function debugSessionLog(entry: Record<string, unknown>): void {
  const payload = JSON.stringify({ ...entry, timestamp: Date.now() });
  fetch("/api/debug-log", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: payload,
  }).catch(() => {});
}
