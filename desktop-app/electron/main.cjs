/* eslint-disable @typescript-eslint/no-require-imports */
const { app, BrowserWindow, ipcMain } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const net = require("net");

const isDev = !app.isPackaged;
const NEXT_PORT = process.env.NEXT_PORT || "3000";
const NEXT_HOST = process.env.NEXT_HOST || "127.0.0.1";
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8000";
const nextUrl = `http://${NEXT_HOST}:${NEXT_PORT}/workspace`;

let nextServerProcess = null;

function waitForPort(port, host, timeoutMs = 45000) {
  const startedAt = Date.now();
  return new Promise((resolve, reject) => {
    const tryConnect = () => {
      const socket = net.createConnection({ port: Number(port), host }, () => {
        socket.end();
        resolve();
      });
      socket.on("error", () => {
        socket.destroy();
        if (Date.now() - startedAt > timeoutMs) {
          reject(new Error(`Timed out waiting for ${host}:${port}`));
          return;
        }
        setTimeout(tryConnect, 300);
      });
    };
    tryConnect();
  });
}

function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 1600,
    height: 980,
    minWidth: 1200,
    minHeight: 760,
    backgroundColor: "#0a0f0a",
    title: "Gaffer's Guide",
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  if (isDev) {
    mainWindow.webContents.openDevTools({ mode: "detach" });
  }

  mainWindow.loadURL(nextUrl);
}

function startNextServer() {
  if (isDev) {
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    const appRoot = path.resolve(__dirname, "..");
    nextServerProcess = spawn(
      process.platform === "win32" ? "npm.cmd" : "npm",
      ["run", "start:web"],
      {
        cwd: appRoot,
        stdio: "inherit",
        env: {
          ...process.env,
          NEXT_PORT,
          NEXT_HOST,
          NEXT_PUBLIC_BACKEND_URL: BACKEND_URL,
        },
      },
    );

    nextServerProcess.on("error", reject);
    nextServerProcess.on("exit", (code) => {
      if (code !== 0) {
        reject(new Error(`Next server exited early with code ${code}`));
      }
    });

    waitForPort(NEXT_PORT, NEXT_HOST)
      .then(resolve)
      .catch(reject);
  });
}

function registerWindowControls() {
  ipcMain.on("window:minimize", (event) => {
    BrowserWindow.fromWebContents(event.sender)?.minimize();
  });
  ipcMain.on("window:maximize", (event) => {
    const win = BrowserWindow.fromWebContents(event.sender);
    if (!win) return;
    if (win.isMaximized()) {
      win.unmaximize();
      return;
    }
    win.maximize();
  });
  ipcMain.on("window:close", (event) => {
    BrowserWindow.fromWebContents(event.sender)?.close();
  });
}

app.whenReady().then(async () => {
  registerWindowControls();
  try {
    await startNextServer();
    createWindow();
  } catch (error) {
    console.error("Failed to launch desktop shell:", error);
    app.quit();
  }
});

app.on("before-quit", () => {
  if (nextServerProcess) {
    nextServerProcess.kill();
    nextServerProcess = null;
  }
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});
