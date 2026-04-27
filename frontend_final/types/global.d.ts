export {};

type PipelineProfile = "fast" | "balanced" | "high_res" | "sahi";

type PipelineLogPayload = {
  stream: "stdout" | "stderr";
  message: string;
};

type PipelineExitPayload = {
  code: number | null;
  signal: NodeJS.Signals | null;
  error: string | null;
};

type RunPipelineResult = {
  started: boolean;
  pid?: number | null;
  error?: string;
};

interface GaffersGuideBridge {
  getApiBase: () => string;
  checkEngineStatus: () => Promise<boolean>;
  runPipeline: (
    videoPath: string,
    outputDir: string,
    profile: PipelineProfile
  ) => Promise<RunPipelineResult>;
  onPipelineLog: (handler: (payload: PipelineLogPayload) => void) => () => void;
  onPipelineExit: (handler: (payload: PipelineExitPayload) => void) => () => void;
}

declare global {
  interface Window {
    gaffersGuide?: GaffersGuideBridge;
  }
}
