const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("gaffersGuide", {
  getApiBase: () => ipcRenderer.sendSync("gg:get-api-base"),
});
