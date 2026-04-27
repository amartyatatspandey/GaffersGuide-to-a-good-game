const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("gaffersGuide", {
  getApiBase: () => ipcRenderer.sendSync("gg:get-api-base"),
  checkEngineStatus: () => ipcRenderer.invoke("check-engine-status"),
  runPipeline: (videoPath, outputDir, profile) =>
    ipcRenderer.invoke("run-pipeline", { videoPath, outputDir, profile }),
  onPipelineLog: (handler) => {
    const listener = (_event, payload) => handler(payload);
    ipcRenderer.on("gg:pipeline-log", listener);
    return () => ipcRenderer.removeListener("gg:pipeline-log", listener);
  },
  onPipelineExit: (handler) => {
    const listener = (_event, payload) => handler(payload);
    ipcRenderer.on("gg:pipeline-exit", listener);
    return () => ipcRenderer.removeListener("gg:pipeline-exit", listener);
  },
});
