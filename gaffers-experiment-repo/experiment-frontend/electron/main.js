/* eslint-disable @typescript-eslint/no-require-imports */
const { app, BrowserWindow } = require("electron");
const path = require("path");
const fs = require("fs");

const DEV_URL = process.env.ELECTRON_START_URL || "http://localhost:5173";

function createWindow(startUrl) {
  const win = new BrowserWindow({
    width: 1280,
    height: 840,
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });
  win.loadURL(startUrl);
}

async function bootstrap() {
  if (!app.isPackaged) {
    createWindow(DEV_URL);
    return;
  }
  const distHtml = path.resolve(__dirname, "..", "dist", "index.html");
  if (!fs.existsSync(distHtml)) {
    throw new Error(`Built frontend not found: ${distHtml}`);
  }
  createWindow(`file://${distHtml}`);
}

app.whenReady().then(bootstrap);
app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) bootstrap();
});
