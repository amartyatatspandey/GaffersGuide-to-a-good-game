/**
 * Electron shell for Gaffer's Guide workspace (Next.js).
 * Dev: loads http://127.0.0.1:3000/workspace (run `npm run dev` first).
 * Prod: run `next build && next start` then point loadURL at that origin, or use loadFile with static export (not configured here).
 */
const { app, BrowserWindow, ipcMain } = require("electron");
const path = require("path");
const fs = require("fs");
const { exec, spawn } = require("child_process");

const ALLOWED_PROFILES = new Set(["fast", "balanced", "high_res", "sahi"]);
let activePipelineProcess = null;

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

  ipcMain.handle("check-engine-status", async () => {
    return await new Promise((resolve) => {
      exec("gaffers-guide --version", (error, stdout) => {
        if (error) {
          resolve(false);
          return;
        }
        resolve(Boolean(stdout && stdout.trim().length > 0));
      });
    });
  });

  ipcMain.handle("run-pipeline", async (event, payload) => {
    if (activePipelineProcess) {
      return { started: false, error: "Pipeline already running" };
    }

    const videoPath =
      typeof payload?.videoPath === "string" ? payload.videoPath.trim() : "";
    const outputDir =
      typeof payload?.outputDir === "string" ? payload.outputDir.trim() : "";
    const profile =
      typeof payload?.profile === "string" ? payload.profile.trim() : "";

    if (!videoPath || !outputDir || !profile) {
      return { started: false, error: "Missing required arguments" };
    }
    if (!ALLOWED_PROFILES.has(profile)) {
      return { started: false, error: "Invalid quality profile" };
    }

    const child = spawn(
      "gaffers-guide",
      [
        "run",
        "--video",
        videoPath,
        "--output",
        outputDir,
        "--quality-profile",
        profile,
      ],
      { stdio: ["ignore", "pipe", "pipe"] }
    );
    activePipelineProcess = child;

    child.stdout.on("data", (chunk) => {
      event.sender.send("gg:pipeline-log", {
        stream: "stdout",
        message: chunk.toString(),
      });
    });

    child.stderr.on("data", (chunk) => {
      event.sender.send("gg:pipeline-log", {
        stream: "stderr",
        message: chunk.toString(),
      });
    });

    child.on("error", (err) => {
      event.sender.send("gg:pipeline-exit", {
        code: null,
        signal: null,
        error: err.message,
      });
      activePipelineProcess = null;
    });

    child.on("close", (code, signal) => {
      event.sender.send("gg:pipeline-exit", {
        code,
        signal,
        error: null,
      });
      activePipelineProcess = null;
    });

    return { started: true, pid: child.pid ?? null };
  });

  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
