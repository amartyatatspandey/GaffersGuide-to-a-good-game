import React, { useEffect, useRef, useState } from "react";
import { UploadCloud, CheckCircle2, Loader2, Circle } from "lucide-react";

import { type CoachAdviceResponse, type JobArtifactsResponse } from "@/lib/api";
import { STEPS, useWebSocketProgress } from "@/hooks/useWebSocketProgress";

interface HopperProps {
  onComplete: (payload: {
    jobId: string;
    artifacts: JobArtifactsResponse | null;
    advice: CoachAdviceResponse | null;
  }) => void;
}

export function Hopper({ onComplete }: HopperProps) {
  const { currentStep, isProcessing, startProcessing, errorMessage, jobId, artifacts, advice } =
    useWebSocketProgress();
  const [selectedFileName, setSelectedFileName] = useState<string>("");
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (currentStep === "Completed" && jobId) {
      onComplete({ jobId, artifacts, advice });
    }
  }, [advice, artifacts, currentStep, jobId, onComplete]);

  const currentIndex =
    isProcessing || currentStep === "Completed"
      ? STEPS.indexOf(currentStep as (typeof STEPS)[number])
    : -1;

  if (currentStep === "Completed") {
    return null;
  }

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>): Promise<void> => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    setSelectedFileName(file.name);
    await startProcessing(file);
  };

  return (
    <div className="flex flex-col gap-6 p-6 w-full max-w-4xl mx-auto h-full justify-center">
      
      {/* Title */}
      <div className="text-center mb-4">
        <h1 className="text-3xl font-bold font-sans tracking-tight text-gray-200">Initialize Analysis Engine</h1>
        <p className="text-gray-500 font-mono text-sm mt-2">
          Local-first telemetry. Upload a high-fidelity match video to start backend analysis.
        </p>
      </div>

      <div 
        onClick={() => {
          if (!isProcessing) {
            fileInputRef.current?.click();
          }
        }}
        className={`w-full h-48 border-2 border-dashed rounded-xl flex flex-col items-center justify-center transition-all ${
          isProcessing 
            ? 'border-emerald-500/50 bg-[#111a12] shadow-[0_0_30px_rgba(16,185,129,0.05)]' 
            : 'border-gray-700 bg-[#0a0f0a] hover:border-emerald-500/70 hover:bg-[#111a12]/50 cursor-pointer'
        }`}
      >
        {!isProcessing ? (
          <>
            <UploadCloud size={40} className="text-gray-600 mb-3" />
            <h2 className="text-lg font-bold text-gray-300 tracking-tight font-sans">System Ready</h2>
            <p className="text-xs text-gray-500 mt-1 font-mono">
              Click to upload a `.mp4` and begin pipeline ingestion
            </p>
            {selectedFileName && (
              <p className="text-xs text-emerald-400 mt-2 font-mono">{selectedFileName}</p>
            )}
          </>
        ) : (
          <>
            <div className="text-emerald-500 mb-3">
              <Loader2 size={40} className="animate-spin" />
            </div>
            <h2 className="text-lg font-bold text-emerald-400 tracking-tight font-mono uppercase">Processing Video Chunk</h2>
          </>
        )}
      </div>
      <input
        ref={fileInputRef}
        type="file"
        accept="video/mp4"
        className="hidden"
        onChange={(event) => {
          void handleFileChange(event);
        }}
      />

      {errorMessage && (
        <div className="rounded-lg border border-red-900 bg-red-950/40 px-4 py-3 text-sm text-red-300">
          {errorMessage}
        </div>
      )}

      {/* Interval Selector */}
      {!isProcessing && (
        <div className="w-full flex items-center justify-center gap-4 animate-fade-in-up mt-2">
          <label className="text-xs font-bold text-gray-500 uppercase tracking-widest font-mono">Chunking Interval</label>
          <select className="bg-[#111a12] border border-gray-800 rounded-lg px-4 py-2 text-gray-300 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500/50 transition-all font-mono text-xs appearance-none cursor-pointer">
            <option>15-minute intervals</option>
            <option>30-minute intervals</option>
            <option>Full Halves (45 mins)</option>
          </select>
        </div>
      )}

      {/* Stepped Progress UI */}
      {isProcessing && (
        <div className="bg-[#111a12] border border-[#1a2420] rounded-xl p-6 shadow-2xl mt-4">
          <h3 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-6 border-b border-gray-800 pb-3">Telemetry Pipeline Tracker</h3>
          <div className="flex flex-col gap-5">
            {STEPS.map((step, idx) => {
              const matchesCurrent = step === currentStep;
              const isPast = idx < currentIndex || (currentStep as string) === 'Completed';
              
              return (
                <div key={step} className="flex items-center gap-4">
                  <div className="flex-shrink-0">
                    {isPast ? (
                      <CheckCircle2 size={18} className="text-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.3)] rounded-full" />
                    ) : matchesCurrent ? (
                      <Loader2 size={18} className="text-emerald-400 animate-spin" />
                    ) : (
                      <Circle size={18} className="text-gray-700" />
                    )}
                  </div>
                  <span className={`font-mono text-sm tracking-wide ${isPast ? 'text-gray-400' : matchesCurrent ? 'text-emerald-400 font-bold' : 'text-gray-700'}`}>
                    {step}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
