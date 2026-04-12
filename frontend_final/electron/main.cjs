/**
 * Electron shell for Gaffer's Guide workspace (Next.js).
 * Dev: loads http://127.0.0.1:3000/workspace (run `npm run dev` first).
 * Prod: run `next build && next start` then point loadURL at that origin, or use loadFile with static export (not configured here).
 */
const { app, BrowserWindow, ipcMain } = require("electron");
const path = require("path");
const fs = require("fs");

function readConfig(configPath) {
  if (!fs.existsSync(configPath)) return { apiBase: "", workspaceUrl: "" };
  try {
    const raw = fs.readFileSync(configPath, "utf8");
    const parsed = JSON.parse(raw);
    return {
      apiBase: typeof parsed.apiBase === "string" ? parsed.apiBase.trim() : "",
      workspaceUrl:
        typeof parsed.workspaceUrl === "string" ? parsed.workspaceUrl.trim() : "",
    };
  } catch {
    return { apiBase: "", workspaceUrl: "" };
  }
}

function configPath() {
  return app.isPackaged
    ? path.join(process.resourcesPath, "config.json")
    : path.join(__dirname, "..", "resources", "config.json");
}

function resolveApiBase() {
  const envOverride = process.env.GAFFERS_GUIDE_API_URL?.trim();
  if (envOverride) return envOverride.replace(/\/$/, "");
  const { apiBase } = readConfig(configPath());
  if (apiBase) return apiBase.replace(/\/$/, "");
  return "";
}

function resolveWorkspaceUrl() {
  const env = process.env.GAFFERS_GUIDE_WORKSPACE_URL?.trim();
  if (env) return env;
  const { workspaceUrl } = readConfig(configPath());
  if (workspaceUrl) return workspaceUrl;
  return process.env.ELECTRON_START_URL || "http://127.0.0.1:3000/workspace";
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  const loadUrl = resolveWorkspaceUrl();
  win.loadURL(loadUrl);
  if (!app.isPackaged) {
    win.webContents.openDevTools({ mode: "detach" });
  }
}

app.whenReady().then(() => {
  const resolved = resolveApiBase();
  ipcMain.on("gg:get-api-base", (event) => {
    event.returnValue = resolved;
  });
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
