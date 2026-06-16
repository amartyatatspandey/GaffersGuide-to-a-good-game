const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("gaffersGuide", {
  getApiBase: () => ipcRenderer.sendSync("gg:get-api-base"),
  windowControls: {
    minimize: () => ipcRenderer.send("gg:window-minimize"),
    maximize: () => ipcRenderer.send("gg:window-maximize"),
    close: () => ipcRenderer.send("gg:window-close"),
  }
});
