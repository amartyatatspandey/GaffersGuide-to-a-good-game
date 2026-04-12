"use client";
import React, { useEffect, useState } from 'react';
import { X, Cloud, Lock, Cpu, Server } from 'lucide-react';

const LS_ENGINE = "gaffer-engine-type";
const LS_CONNECTIVITY = "gaffer-connectivity";
const LS_OLLAMA_MODEL = "gaffer-ollama-model";

export function EngineSettingsModal({ isOpen, onClose }: { isOpen: boolean, onClose: () => void }) {
  const [engineType, setEngineType] = useState<'cloud' | 'local'>('local');
  const [connectivity, setConnectivity] = useState<'online' | 'offline'>('online');
  const [ollamaModel, setOllamaModel] = useState<string>("llama3");

  useEffect(() => {
    if (!isOpen || typeof window === "undefined") return;
    const eng = localStorage.getItem(LS_ENGINE);
    if (eng === "local" || eng === "cloud") setEngineType(eng);
    const con = localStorage.getItem(LS_CONNECTIVITY);
    if (con === "online" || con === "offline") setConnectivity(con);
    const m = localStorage.getItem(LS_OLLAMA_MODEL);
    if (m && m.trim()) setOllamaModel(m.trim());
  }, [isOpen]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    localStorage.setItem(LS_ENGINE, engineType);
  }, [engineType]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    localStorage.setItem(LS_CONNECTIVITY, connectivity);
  }, [connectivity]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    localStorage.setItem(LS_OLLAMA_MODEL, ollamaModel.trim() || "llama3");
  }, [ollamaModel]);

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
          <p className="text-sm text-gray-400 mb-6">Configure your primary inference engine and connectivity state to dictate how telemetry is processed.</p>
          
          {/* Hybrid Toggle: Engine Type */}
          <div className="mb-4 text-[10px] font-bold text-gray-500 uppercase tracking-widest font-mono">1. Inference Architecture</div>
          <div className="flex gap-2 p-1 bg-[#0a0f0a] border border-gray-800 rounded-xl mb-6">
            <button 
              onClick={() => setEngineType('cloud')}
              className={`flex-1 py-3 px-4 rounded-lg flex items-center justify-center gap-2 text-sm font-bold transition-all border ${
                engineType === 'cloud' 
                  ? 'bg-blue-900/40 text-blue-400 border-blue-800 shadow-[0_0_15px_rgba(59,130,246,0.2)]' 
                  : 'text-gray-500 border-transparent hover:text-gray-300 hover:bg-gray-800/50'
              }`}
            >
              <Cloud size={16} /> Cloud Engine
            </button>
            <button 
              onClick={() => setEngineType('local')}
              className={`flex-1 py-3 px-4 rounded-lg flex items-center justify-center gap-2 text-sm font-bold transition-all border ${
                engineType === 'local' 
                  ? 'bg-emerald-900/40 text-emerald-400 border-emerald-800 shadow-[0_0_15px_rgba(16,185,129,0.2)]' 
                  : 'text-gray-500 border-transparent hover:text-gray-300 hover:bg-gray-800/50'
              }`}
            >
              <Lock size={16} /> Local Engine
            </button>
          </div>

          {/* Connectivity Toggle */}
          <div className="mb-4 text-[10px] font-bold text-gray-500 uppercase tracking-widest font-mono">2. Connectivity Mode</div>
          <div className="flex gap-2 p-1 bg-[#0a0f0a] border border-gray-800 rounded-xl mb-6">
            <button 
              onClick={() => setConnectivity('online')}
              className={`flex-1 py-2 px-4 rounded-lg flex items-center justify-center gap-2 text-xs font-bold transition-all border ${
                connectivity === 'online' 
                  ? 'bg-gray-800 text-gray-200 border-gray-600' 
                  : 'text-gray-600 border-transparent hover:text-gray-400 hover:bg-gray-900'
              }`}
            >
              Online Mode
            </button>
            <button 
              onClick={() => setConnectivity('offline')}
              className={`flex-1 py-2 px-4 rounded-lg flex items-center justify-center gap-2 text-xs font-bold transition-all border ${
                connectivity === 'offline' 
                  ? 'bg-gray-800 text-gray-200 border-gray-600' 
                  : 'text-gray-600 border-transparent hover:text-gray-400 hover:bg-gray-900'
              }`}
            >
              Offline Mode
            </button>
          </div>

          {/* Configuration Inputs */}
          <div className="min-h-[140px] p-5 bg-[#0a0f0a]/50 rounded-xl border border-gray-800/50">
            {engineType === 'cloud' && connectivity === 'online' && (
              <div className="animate-fade-in-up">
                <label className="block text-xs font-bold text-gray-500 uppercase tracking-widest mb-2 font-mono">OpenAI / Gemini API Key</label>
                <div className="relative">
                  <input 
                    type="password" 
                    placeholder="sk-..." 
                    className="w-full bg-[#0a0f0a] border border-gray-700 rounded-lg px-4 py-3 text-gray-200 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500/50 transition-all font-mono text-sm"
                  />
                  <Server size={16} className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-600" />
                </div>
                <p className="text-xs text-blue-500/70 mt-3 flex items-center gap-1 font-mono"><Cloud size={12}/> Live streaming telemetry to cloud endpoints.</p>
              </div>
            )}

            {engineType === 'cloud' && connectivity === 'offline' && (
              <div className="animate-fade-in-up py-4 text-center">
                 <Cloud size={32} className="mx-auto text-indigo-500/50 mb-3" />
                 <h4 className="text-sm font-bold text-indigo-400 mb-1">Cloud Mock / Cached Sync</h4>
                 <p className="text-xs text-gray-500 max-w-sm mx-auto">Operating without internet. Telemetry is parsing against the last synced high-fidelity cached payloads.</p>
              </div>
            )}

            {engineType === 'local' && connectivity === 'online' && (
              <div className="animate-fade-in-up py-4 space-y-3 text-center">
                 <Server size={32} className="mx-auto text-cyan-500/50 mb-3" />
                 <h4 className="text-sm font-bold text-cyan-400 mb-1">Distributed Local Node</h4>
                 <p className="text-xs text-gray-500 max-w-sm mx-auto">Chat uses Ollama at <span className="font-mono text-gray-400">OLLAMA_BASE_URL</span> (default <span className="font-mono">http://127.0.0.1:11434</span>).</p>
                 <div className="text-left pt-2">
                   <label className="block text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-1 font-mono">Ollama model name</label>
                   <input
                     type="text"
                     value={ollamaModel}
                     onChange={(e) => setOllamaModel(e.target.value)}
                     placeholder="llama3"
                     className="w-full bg-[#0a0f0a] border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 font-mono focus:outline-none focus:border-cyan-600"
                   />
                 </div>
              </div>
            )}

            {engineType === 'local' && connectivity === 'offline' && (
              <div className="animate-fade-in-up">
                <label className="block text-xs font-bold text-gray-500 uppercase tracking-widest mb-2 font-mono">Strict Airgap Checkpoint</label>
                <select
                  value={ollamaModel}
                  onChange={(e) => setOllamaModel(e.target.value)}
                  className="w-full bg-[#0a0f0a] border border-gray-700 rounded-lg px-4 py-3 text-gray-200 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500/50 transition-all font-mono text-sm appearance-none custom-select"
                >
                  <option value="llama3">llama3 (Recommended)</option>
                  <option value="llama3:8b">llama3:8b</option>
                  <option value="mistral">mistral</option>
                  <option value="phi3">phi3</option>
                </select>
                
                <div className="mt-4 p-3 bg-amber-900/20 border border-amber-800/50 rounded-lg flex items-start gap-3">
                  <span className="text-amber-500 mt-0.5">⚠️</span>
                  <p className="text-xs text-amber-200/80 leading-relaxed font-mono">
                    <strong>Airgap Active:</strong> Running entirely on local device hardware. Ensure you have 16GB+ RAM available to prevent thermal throttling.
                  </p>
                </div>
              </div>
            )}

          </div>
        </div>

        {/* Footer */}
        <div className="bg-[#0a0f0a] border-t border-gray-900 px-6 py-4 flex justify-between items-center">
          <div className="font-mono text-[10px] uppercase text-gray-600 tracking-widest">
            Target: <span className={
              engineType === 'cloud' && connectivity === 'online' ? 'text-blue-500' :
              engineType === 'cloud' && connectivity === 'offline' ? 'text-indigo-400' :
              engineType === 'local' && connectivity === 'online' ? 'text-cyan-400' : 'text-emerald-500'
            }>{engineType} • {connectivity}</span>
          </div>
          <button
            type="button"
            onClick={() => {
              if (typeof window !== "undefined") {
                window.dispatchEvent(new Event("gaffer-engine-changed"));
              }
              onClose();
            }}
            className={`rounded px-6 py-2 text-sm font-bold tracking-wide transition-colors ${
              engineType === 'cloud' ? 'bg-blue-600 hover:bg-blue-500 text-white' : 'bg-emerald-600 hover:bg-emerald-500 text-white'
            }`}
          >
            Deploy Pipeline
          </button>
        </div>
      </div>
    </div>
  );
}
