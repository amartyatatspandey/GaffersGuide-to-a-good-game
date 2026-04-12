"use client";
import React, { createContext, useCallback, useContext, useMemo, useState } from "react";

export type CvEngine = "local" | "cloud";
export type LlmEngine = "local" | "cloud";
export type Connectivity = "online" | "offline";

export interface EngineConfig {
  cvEngine: CvEngine;
  llmEngine: LlmEngine;
  connectivity: Connectivity;
  setCvEngine: (v: CvEngine) => void;
  setLlmEngine: (v: LlmEngine) => void;
  setConnectivity: (v: Connectivity) => void;
}

const DEFAULT_ENGINE_CONFIG: Omit<EngineConfig, "setCvEngine" | "setLlmEngine" | "setConnectivity"> =
  {
    cvEngine: "local",
    llmEngine: "local",
    connectivity: "online",
  };

const EngineContext = createContext<EngineConfig>({
  ...DEFAULT_ENGINE_CONFIG,
  setCvEngine: () => undefined,
  setLlmEngine: () => undefined,
  setConnectivity: () => undefined,
});

export function EngineProvider({ children }: { children: React.ReactNode }): React.JSX.Element {
  const [cvEngine, setCvEngineState] = useState<CvEngine>(DEFAULT_ENGINE_CONFIG.cvEngine);
  const [llmEngine, setLlmEngineState] = useState<LlmEngine>(DEFAULT_ENGINE_CONFIG.llmEngine);
  const [connectivity, setConnectivityState] = useState<Connectivity>(
    DEFAULT_ENGINE_CONFIG.connectivity,
  );

  const setCvEngine = useCallback((v: CvEngine) => setCvEngineState(v), []);
  const setLlmEngine = useCallback((v: LlmEngine) => setLlmEngineState(v), []);
  const setConnectivity = useCallback((v: Connectivity) => setConnectivityState(v), []);

  const value = useMemo(
    () => ({ cvEngine, llmEngine, connectivity, setCvEngine, setLlmEngine, setConnectivity }),
    [cvEngine, llmEngine, connectivity, setCvEngine, setLlmEngine, setConnectivity],
  );

  return <EngineContext.Provider value={value}>{children}</EngineContext.Provider>;
}

export function useEngineConfig(): EngineConfig {
  return useContext(EngineContext);
}
