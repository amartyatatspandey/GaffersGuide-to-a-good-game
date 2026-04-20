import React, { useState, useEffect, useRef } from 'react';
import { useWebSocketProgress, STEPS } from '@/hooks/useWebSocketProgress';
import { UploadCloud, CheckCircle2, Loader2, Circle, AlertCircle } from 'lucide-react';
import { createJob } from '@/lib/api/jobs';

export function Hopper({ onComplete }: { onComplete: (jobId: string, file: File) => void }) {
  const {
    currentStep,
    isProcessing,
    error,
    connectionState,
    elapsedSeconds,
    startTracking,
  } = useWebSocketProgress();
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (currentStep === 'Completed' && activeJobId && file) {
      onComplete(activeJobId, file);
    }
  }, [currentStep, activeJobId, file, onComplete]);

  const stepIndexInMilestones = STEPS.indexOf(currentStep);
  const isSlowStep = elapsedSeconds >= 30;
  const isLocalLlmStep = currentStep.toLowerCase().includes('llm (local)');
  const currentIndex =
    isProcessing || currentStep === 'Completed'
      ? stepIndexInMilestones >= 0
        ? stepIndexInMilestones
        : -1
      : -1;

  if (currentStep === 'Completed') {
    return null;
  }

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;
    
    setFile(selectedFile);
    setIsUploading(true);
    
    try {
      const llmPref =
        typeof window !== 'undefined' && localStorage.getItem('gaffer-engine-type') === 'cloud'
          ? 'cloud'
          : 'local';
      const job = await createJob(selectedFile, llmPref);
      setActiveJobId(job.job_id);
      startTracking(job.job_id);
    } catch (err: unknown) {
      console.error(err);
      const message = err instanceof Error ? err.message : String(err);
      alert('Failed to upload video: ' + message);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="flex flex-col gap-6 p-6 w-full max-w-4xl mx-auto h-full justify-center">
      
      {/* Title */}
      <div className="text-center mb-4">
        <h1 className="text-3xl font-bold font-sans tracking-tight text-gray-200">Initialize Analysis Engine</h1>
        <p className="text-gray-500 font-mono text-sm mt-2">Local-first telemetry. Connect high-fidelity match footage to generate insights.</p>
      </div>

      {/* Drag & Drop Zone */}
      <div 
        onClick={() => { if(!isProcessing && !isUploading) fileInputRef.current?.click(); }}
        className={`w-full h-48 border-2 border-dashed rounded-xl flex flex-col items-center justify-center transition-all ${
          (isProcessing || isUploading)
            ? 'border-emerald-500/50 bg-[#111a12] shadow-[0_0_30px_rgba(16,185,129,0.05)]' 
            : 'border-gray-700 bg-[#0a0f0a] hover:border-emerald-500/70 hover:bg-[#111a12]/50 cursor-pointer'
        }`}
      >
        <input 
          type="file" 
          accept="video/mp4,.mp4"
          className="hidden" 
          ref={fileInputRef} 
          onChange={handleFileChange} 
        />
        
        {(!isProcessing && !isUploading) ? (
          <>
            <UploadCloud size={40} className="text-gray-600 mb-3" />
            <h2 className="text-lg font-bold text-gray-300 tracking-tight font-sans">System Ready</h2>
            <p className="text-xs text-gray-500 mt-1 font-mono">MP4 only — click to upload match footage</p>
          </>
        ) : (
          <>
            <div className="text-emerald-500 mb-3">
              <Loader2 size={40} className="animate-spin" />
            </div>
            <h2 className="text-lg font-bold text-emerald-400 tracking-tight font-mono uppercase">
              {isUploading ? 'Uploading Video...' : 'Processing Pipeline...'}
            </h2>
            {!isUploading ? (
              <p className="text-xs text-emerald-300/80 mt-1 font-mono">
                Connection: {connectionState}
              </p>
            ) : null}
          </>
        )}
      </div>

      {error && (
        <div className="text-red-400 flex items-center justify-center gap-2 mt-4 font-mono text-sm">
           <AlertCircle size={16} /> {error}
        </div>
      )}

      {/* Interval Selector */}
      {(!isProcessing && !isUploading) && (
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
          <h3 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-3 border-b border-gray-800 pb-3">Telemetry Pipeline Tracker</h3>
          <p className="text-[11px] font-mono text-emerald-300/90 mb-4">
            Backend step: <span className="text-gray-100">{currentStep}</span>
            {isProcessing ? (
              <span className="ml-2 text-gray-500">(running {elapsedSeconds}s)</span>
            ) : null}
            {stepIndexInMilestones === -1 && currentStep !== 'Pending' && currentStep !== 'Completed' ? (
              <span className="ml-2 text-gray-500">(not in milestone list)</span>
            ) : null}
          </p>
          {isSlowStep ? (
            <p className="text-[11px] font-mono text-amber-300/90 mb-4">
              {isLocalLlmStep
                ? 'Local LLM generation can take a while on large clips. Pipeline is still active.'
                : 'This step is taking longer than usual. If it keeps growing, check backend logs.'}
            </p>
          ) : null}
          <div className="flex flex-col gap-5">
            {STEPS.map((step, idx) => {
              const matchesCurrent = step === currentStep;
              const isPast =
                currentStep === 'Completed' ||
                (currentIndex >= 0 && idx < currentIndex);
              
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
