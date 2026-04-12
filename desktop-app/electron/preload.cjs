/* eslint-disable @typescript-eslint/no-require-imports */
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("desktopWindow", {
  minimize: () => ipcRenderer.send("window:minimize"),
  maximize: () => ipcRenderer.send("window:maximize"),
  close: () => ipcRenderer.send("window:close"),
});

// Expose runtime config so the Next.js renderer can read the backend URL
// even in packaged Electron where NEXT_PUBLIC_* is baked at build time.
contextBridge.exposeInMainWorld("desktopConfig", {
  backendUrl: process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8000",
  isDev: process.env.NODE_ENV !== "production",
});
