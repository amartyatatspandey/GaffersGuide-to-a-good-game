/**
 * Resolves the base API URL and WebSocket URL from the environment.
 */
export function getApiBaseUrl(): string {
  // Typical default for local python/fastapi backend is 8000
  return process.env.NEXT_PUBLIC_API_BASE || 'http://127.0.0.1:8000';
}

export function getWsBaseUrl(): string {
  const httpUrl = getApiBaseUrl();
  if (httpUrl.startsWith('https://')) {
    return httpUrl.replace('https://', 'wss://');
  }
  return httpUrl.replace('http://', 'ws://');
}
