declare global {
  interface Window {
    /** Set by Electron preload (`electron/preload.cjs`) when packaged or dev. */
    gaffersGuide?: { getApiBase: () => string };
  }
}

export type ApiEndpoints = {
  httpBase: string;
  wsBase: string;
  host: string;
  source: 'electron' | 'env' | 'default';
};

function sanitizeBaseUrl(raw: string): string {
  const trimmed = raw.trim().replace(/\/+$/, '');
  if (!trimmed) {
    throw new Error('API base URL is empty.');
  }
  const parsed = new URL(trimmed);
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new Error(
      `Unsupported API protocol "${parsed.protocol}". Use http:// or https://.`,
    );
  }
  return parsed.toString().replace(/\/+$/, '');
}

/**
 * Resolves the base API URL for HTTP and WebSocket calls.
 *
 * Order: Electron `window.gaffersGuide.getApiBase()` (non-empty) wins, then
 * `NEXT_PUBLIC_API_BASE`, then local FastAPI default.
 */
export function getApiBaseUrl(): string {
  return resolveApiEndpoints().httpBase;
}

export function getWsBaseUrl(): string {
  return resolveApiEndpoints().wsBase;
}

export function resolveApiEndpoints(): ApiEndpoints {
  let candidate = 'http://127.0.0.1:8000';
  let source: ApiEndpoints['source'] = 'default';

  if (typeof window !== 'undefined') {
    const bridge = window.gaffersGuide?.getApiBase;
    if (typeof bridge === 'function') {
      const fromElectron = bridge();
      if (typeof fromElectron === 'string' && fromElectron.trim().length > 0) {
        candidate = fromElectron;
        source = 'electron';
      }
    }
  }

  if (source === 'default') {
    const env = process.env.NEXT_PUBLIC_API_BASE;
    if (typeof env === 'string' && env.trim().length > 0) {
      candidate = env;
      source = 'env';
    }
  }

  const httpBase = sanitizeBaseUrl(candidate);
  const parsed = new URL(httpBase);
  const wsProtocol = parsed.protocol === 'https:' ? 'wss:' : 'ws:';

  return {
    httpBase,
    wsBase: `${wsProtocol}//${parsed.host}`,
    host: parsed.host,
    source,
  };
}
