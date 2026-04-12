"use client";
import React from "react";
import { Cloud, Cpu, Lock, Server, X } from "lucide-react";

import { type CvEngine, type LlmEngine, useEngineConfig } from "@/context/EngineContext";
import { getLocalLlmPreflight, type LocalLlmPreflightResponse } from "@/lib/api";

export function EngineSettingsModal({
  isOpen,
  onClose,
}: {
  isOpen: boolean;
  onClose: () => void;
}): React.JSX.Element | null {
  const { cvEngine, llmEngine, connectivity, setCvEngine, setLlmEngine, setConnectivity } =
    useEngineConfig();
  const [llmCheck, setLlmCheck] = React.useState<LocalLlmPreflightResponse | null>(null);
  const [llmCheckLoading, setLlmCheckLoading] = React.useState(false);
  const [llmCheckError, setLlmCheckError] = React.useState<string | null>(null);

  const engineType: CvEngine = cvEngine;
  const setEngineType = (v: CvEngine): void => {
    setCvEngine(v);
    setLlmEngine(v as LlmEngine);
  };

  const runLocalCheck = async (): Promise<void> => {
    setLlmCheckLoading(true);
    setLlmCheckError(null);
    try {
      const payload = await getLocalLlmPreflight();
      setLlmCheck(payload);
    } catch (error) {
      setLlmCheckError(error instanceof Error ? error.message : "Local LLM preflight failed.");
    } finally {
      setLlmCheckLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm font-sans transition-all duration-300">
      <div className="bg-[#111a12] border border-gray-800 rounded-2xl w-full max-w-xl shadow-[0_0_50px_rgba(0,0,0,0.8)] overflow-hidden relative">
        {/* Header */}
        <div className="flex justify-between items-center bg-[#0a0f0a] px-6 py-4 border-b border-gray-900">
          <h2 className="text-sm font-bold text-gray-300 font-mono tracking-widest uppercase flex items-center gap-2">
            <Cpu size={16} className="text-emerald-500" /> AI Engine Configuration
          </h2>
          <button onClick={onClose} className="text-gray-500 hover:text-white transition-colors">
            <X size={18} />
          </button>
        </div>

        {/* Body */}
        <div className="p-6">
          <p className="text-sm text-gray-400 mb-6">
            Configure your primary inference engine and connectivity state to dictate how telemetry
            is processed.
          </p>

          {/* Hybrid Toggle: Engine Type */}
          <div className="mb-4 text-[10px] font-bold text-gray-500 uppercase tracking-widest font-mono">
            1. Inference Architecture
          </div>
          <div className="flex gap-2 p-1 bg-[#0a0f0a] border border-gray-800 rounded-xl mb-6">
            <button
              onClick={() => setEngineType("cloud")}
              className={`flex-1 py-3 px-4 rounded-lg flex items-center justify-center gap-2 text-sm font-bold transition-all border ${
                engineType === "cloud"
                  ? "bg-blue-900/40 text-blue-400 border-blue-800 shadow-[0_0_15px_rgba(59,130,246,0.2)]"
                  : "text-gray-500 border-transparent hover:text-gray-300 hover:bg-gray-800/50"
              }`}
            >
              <Cloud size={16} /> Cloud Engine
            </button>
            <button
              onClick={() => setEngineType("local")}
              className={`flex-1 py-3 px-4 rounded-lg flex items-center justify-center gap-2 text-sm font-bold transition-all border ${
                engineType === "local"
                  ? "bg-emerald-900/40 text-emerald-400 border-emerald-800 shadow-[0_0_15px_rgba(16,185,129,0.2)]"
                  : "text-gray-500 border-transparent hover:text-gray-300 hover:bg-gray-800/50"
              }`}
            >
              <Lock size={16} /> Local Engine
            </button>
          </div>

          {/* Connectivity Toggle */}
          <div className="mb-4 text-[10px] font-bold text-gray-500 uppercase tracking-widest font-mono">
            2. Connectivity Mode
          </div>
          <div className="flex gap-2 p-1 bg-[#0a0f0a] border border-gray-800 rounded-xl mb-6">
            <button
              onClick={() => setConnectivity("online")}
              className={`flex-1 py-2 px-4 rounded-lg flex items-center justify-center gap-2 text-xs font-bold transition-all border ${
                connectivity === "online"
                  ? "bg-gray-800 text-gray-200 border-gray-600"
                  : "text-gray-600 border-transparent hover:text-gray-400 hover:bg-gray-900"
              }`}
            >
              Online Mode
            </button>
            <button
              onClick={() => setConnectivity("offline")}
              className={`flex-1 py-2 px-4 rounded-lg flex items-center justify-center gap-2 text-xs font-bold transition-all border ${
                connectivity === "offline"
                  ? "bg-gray-800 text-gray-200 border-gray-600"
                  : "text-gray-600 border-transparent hover:text-gray-400 hover:bg-gray-900"
              }`}
            >
              Offline Mode
            </button>
          </div>

          {/* Info panel */}
          <div className="min-h-[120px] p-5 bg-[#0a0f0a]/50 rounded-xl border border-gray-800/50">
            {engineType === "cloud" && connectivity === "online" && (
              <div className="animate-fade-in-up py-4 text-center">
                <Cloud size={32} className="mx-auto text-blue-500/50 mb-3" />
                <h4 className="text-sm font-bold text-blue-400 mb-1">Cloud Engine — Online</h4>
                <p className="text-xs text-gray-500 max-w-sm mx-auto">
                  Requires{" "}
                  <span className="font-mono text-gray-400">MODAL_WEBHOOK_URL</span> on the
                  backend. CV and LLM run on cloud infrastructure.
                </p>
              </div>
            )}
            {engineType === "cloud" && connectivity === "offline" && (
              <div className="animate-fade-in-up py-4 text-center">
                <Cloud size={32} className="mx-auto text-indigo-500/50 mb-3" />
                <h4 className="text-sm font-bold text-indigo-400 mb-1">
                  Cloud Engine — Offline
                </h4>
                <p className="text-xs text-gray-500 max-w-sm mx-auto">
                  Cloud engines require an internet connection. Switch to Local Engine for
                  offline operation.
                </p>
              </div>
            )}
            {engineType === "local" && connectivity === "online" && (
              <div className="animate-fade-in-up py-4 text-center">
                <Server size={32} className="mx-auto text-cyan-500/50 mb-3" />
                <h4 className="text-sm font-bold text-cyan-400 mb-1">
                  Local Engine — Online
                </h4>
                <p className="text-xs text-gray-500 max-w-sm mx-auto">
                  CV runs the on-device pipeline. LLM routes to Ollama if available, otherwise
                  falls back to cloud.
                </p>
                <div className="mt-3 flex justify-center">
                  <button
                    type="button"
                    onClick={() => void runLocalCheck()}
                    disabled={llmCheckLoading}
                    className="rounded border border-cyan-700 px-3 py-1 text-xs text-cyan-300 hover:bg-cyan-900/20 disabled:opacity-60"
                  >
                    {llmCheckLoading ? "Checking Ollama..." : "Check Local LLM"}
                  </button>
                </div>
                {llmCheck && (
                  <div className="mt-3 rounded border border-gray-700 bg-black/20 p-2 text-left text-[11px] text-gray-300">
                    <div>Daemon: {llmCheck.daemon_reachable ? "reachable" : "offline"}</div>
                    <div>
                      Model `{llmCheck.configured_model}`: {llmCheck.model_present ? "installed" : "missing"}
                    </div>
                    <div>Generation: {llmCheck.generation_ok ? "ok" : "failed"}</div>
                    {llmCheck.hint && <div className="text-amber-300 mt-1">{llmCheck.hint}</div>}
                  </div>
                )}
                {llmCheckError && (
                  <div className="mt-2 text-xs text-red-400">{llmCheckError}</div>
                )}
              </div>
            )}
            {engineType === "local" && connectivity === "offline" && (
              <div className="animate-fade-in-up py-4 text-center">
                <Server size={32} className="mx-auto text-emerald-500/50 mb-3" />
                <h4 className="text-sm font-bold text-emerald-400 mb-1">
                  Local Engine — Offline (Airgap)
                </h4>
                <div className="mt-3 p-3 bg-amber-900/20 border border-amber-800/50 rounded-lg flex items-start gap-3 text-left">
                  <span className="text-amber-500 mt-0.5">⚠</span>
                  <p className="text-xs text-amber-200/80 leading-relaxed font-mono">
                    <strong>Airgap Active:</strong> Running entirely on local device hardware.
                    Ensure you have 16 GB+ RAM available.
                  </p>
                </div>
                <div className="mt-3 rounded border border-gray-700 bg-black/20 p-2 text-xs text-gray-300 text-left">
                  <div className="font-semibold mb-1">Local LLM setup checklist</div>
                  <div>1. Install Ollama from ollama.com/download</div>
                  <div>2. Run `ollama pull llama3`</div>
                  <div>3. Start daemon with `ollama serve`</div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="bg-[#0a0f0a] border-t border-gray-900 px-6 py-4 flex justify-between items-center">
          <div className="font-mono text-[10px] uppercase text-gray-600 tracking-widest">
            Pipeline:{" "}
            <span
              className={
                engineType === "cloud" && connectivity === "online"
                  ? "text-blue-500"
                  : engineType === "cloud" && connectivity === "offline"
                    ? "text-indigo-400"
                    : engineType === "local" && connectivity === "online"
                      ? "text-cyan-400"
                      : "text-emerald-500"
              }
            >
              {engineType} • {llmEngine} llm • {connectivity}
            </span>
          </div>
          <button
            onClick={onClose}
            className={`px-6 py-2 rounded font-bold text-sm tracking-wide transition-colors ${
              engineType === "cloud"
                ? "bg-blue-600 hover:bg-blue-500 text-white"
                : "bg-emerald-600 hover:bg-emerald-500 text-white"
            }`}
          >
            Apply &amp; Close
          </button>
        </div>
      </div>
    </div>
  );
}
