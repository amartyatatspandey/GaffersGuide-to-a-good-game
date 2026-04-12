export {};

declare global {
  interface Window {
    desktopWindow?: {
      minimize: () => void;
      maximize: () => void;
      close: () => void;
    };
    desktopConfig?: {
      /** Backend FastAPI base URL, injected by Electron preload at runtime. */
      backendUrl: string;
      isDev: boolean;
    };
  }
}
