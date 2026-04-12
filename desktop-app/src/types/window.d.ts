export {};

declare global {
  interface Window {
    desktopWindow?: {
      minimize: () => void;
      maximize: () => void;
      close: () => void;
    };
  }
}
