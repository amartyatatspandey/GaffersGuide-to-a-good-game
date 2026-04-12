declare global {
  interface Window {
    /** Set by Electron preload (`electron/preload.cjs`) when packaged or dev. */
    gaffersGuide?: { getApiBase: () => string };
  }
}

/**
 * Resolves the base API URL for HTTP and WebSocket calls.
 *
 * Order: Electron `window.gaffersGuide.getApiBase()` (non-empty) wins, then
 * `NEXT_PUBLIC_API_BASE`, then local FastAPI default.
 */
export function getApiBaseUrl(): string {
  if (typeof window !== 'undefined') {
    const bridge = window.gaffersGuide?.getApiBase;
    if (typeof bridge === 'function') {
      const fromElectron = bridge();
      if (typeof fromElectron === 'string' && fromElectron.trim().length > 0) {
        return fromElectron.trim().replace(/\/$/, '');
      }
    }
  }
  const env = process.env.NEXT_PUBLIC_API_BASE;
  if (typeof env === 'string' && env.trim().length > 0) {
    return env.trim().replace(/\/$/, '');
  }
  return 'http://127.0.0.1:8000';
}

export function getWsBaseUrl(): string {
  const httpUrl = getApiBaseUrl();
  if (httpUrl.startsWith('https://')) {
    return httpUrl.replace('https://', 'wss://');
  }
  return httpUrl.replace('http://', 'ws://');
}
