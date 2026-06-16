import React, { useState, useEffect, useRef } from 'react';
import { useWebSocketProgress, STEPS } from '@/hooks/useWebSocketProgress';
import { useChunkedUpload, type UploadPhase } from '@/hooks/useChunkedUpload';
import {
  UploadCloud,
  CheckCircle2,
  Loader2,
  Circle,
  AlertCircle,
  XCircle,
  RefreshCw,
  HardDrive,
  Wifi,
  Zap,
  Clock,
} from 'lucide-react';

export function Hopper({ onComplete }: { onComplete: (jobId: string, file: File) => void }) {
  const { currentStep, isProcessing, error: wsError, startTracking } = useWebSocketProgress();
  const chunked = useChunkedUpload();
  const [file, setFile] = useState<File | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // When pipeline processing completes, notify parent
  useEffect(() => {
    if (currentStep === 'Completed' && activeJobId && file) {
      onComplete(activeJobId, file);
    }
  }, [currentStep, activeJobId, file, onComplete]);

  // Track milestone progress for the pipeline phase
  const stepIndexInMilestones = STEPS.findIndex(s =>
    currentStep === s ||
    (s === 'Tracking Players' && currentStep.toLowerCase().includes('tracking')) ||
    (s === 'Spatial Math' && currentStep.toLowerCase().includes('spatial')) ||
    (s === 'Rule Engine' && (currentStep.toLowerCase().includes('rule') || currentStep.toLowerCase().includes('tactical'))) ||
    (s === 'Synthesizing Advice' && (currentStep.toLowerCase().includes('synth') || currentStep.toLowerCase().includes('llm')))
  );

  const currentIndex =
    isProcessing || currentStep === 'Completed'
      ? stepIndexInMilestones >= 0
        ? stepIndexInMilestones
        : (currentStep.toLowerCase().includes('tracking') ? 0 : -1)
      : -1;

  if (currentStep === 'Completed') {
    return null;
  }

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    setFile(selectedFile);

    const llmPref =
      typeof window !== 'undefined' && localStorage.getItem('gaffer-engine-type') === 'cloud'
        ? 'cloud'
        : 'local';

    const cvPref = process.env.NEXT_PUBLIC_CV_ENGINE === 'cloud' ? 'cloud' : 'local';

    const qualityPref =
      typeof window !== 'undefined' ? localStorage.getItem('gaffer-quality-profile') || 'balanced' : 'balanced';

    const chunkingPref =
      typeof window !== 'undefined' ? localStorage.getItem('gaffer-chunking-interval') || '15-minute intervals' : '15-minute intervals';

    try {
      const result = await chunked.startUpload(selectedFile, {
        cvEngine: cvPref,
        llmEngine: llmPref,
        qualityProfile: qualityPref,
        chunkingInterval: chunkingPref,
      });

      // Upload complete → job created → start WebSocket tracking
      setActiveJobId(result.job_id);
      startTracking(result.job_id);
    } catch (err: any) {
      if (err?.name === 'AbortError') return; // user cancelled
      console.error('Upload failed:', err);
    }
  };

  // Derived state
  const isUploading = chunked.phase !== 'idle' && chunked.phase !== 'done' && chunked.phase !== 'error';
  const showUploadProgress = chunked.phase === 'uploading' || chunked.phase === 'assembling' || chunked.phase === 'initializing';
  const showPipelineProgress = isProcessing && chunked.phase === 'done' && !wsError;
  const uploadError = chunked.phase === 'error' ? chunked.error : null;
  const pipelineError = wsError || null;
  const showError = !!(uploadError || pipelineError);
  const isIdle = chunked.phase === 'idle' && !isProcessing && !wsError;

  // Format file size for display
  const fileSizeDisplay = file
    ? file.size >= 1024 * 1024 * 1024
      ? `${(file.size / (1024 * 1024 * 1024)).toFixed(2)} GB`
      : `${(file.size / (1024 * 1024)).toFixed(0)} MB`
    : '';

  return (
    <div className="flex flex-col gap-6 p-6 w-full max-w-4xl mx-auto h-full justify-center">

      {/* Title */}
      <div className="text-center mb-4">
        <h1 className="text-3xl font-bold font-sans tracking-tight text-gray-200">Initialize Analysis Engine</h1>
        <p className="text-gray-500 font-mono text-sm mt-2">Local-first telemetry. Connect high-fidelity match footage to generate insights.</p>
      </div>

      {/* Drag & Drop Zone */}
      <div
        onClick={() => { if (isIdle) fileInputRef.current?.click(); }}
        className={`w-full h-48 border-2 border-dashed rounded-xl flex flex-col items-center justify-center transition-all ${
          isIdle
            ? 'border-gray-700 bg-[#0a0f0a] hover:border-emerald-500/70 hover:bg-[#111a12]/50 cursor-pointer'
            : 'border-emerald-500/50 bg-[#111a12] shadow-[0_0_30px_rgba(16,185,129,0.05)]'
        }`}
      >
        <input
          type="file"
          accept="video/mp4,video/quicktime,video/x-msvideo"
          className="hidden"
          ref={fileInputRef}
          onChange={handleFileChange}
        />

        {isIdle ? (
          <>
            <UploadCloud size={40} className="text-gray-600 mb-3" />
            <h2 className="text-lg font-bold text-gray-300 tracking-tight font-sans">System Ready</h2>
            <p className="text-xs text-gray-500 mt-1 font-mono">Click here to upload match footage</p>
            <p className="text-[10px] text-gray-600 mt-2 font-mono">Supports files up to 12 GB · Chunked upload with auto-resume</p>
          </>
        ) : showUploadProgress ? (
          <>
            <div className="text-emerald-500 mb-3">
              <UploadCloud size={40} className="animate-pulse" />
            </div>
            <h2 className="text-lg font-bold text-emerald-400 tracking-tight font-mono uppercase">
              {chunked.phase === 'initializing' ? 'Initializing Upload...' :
               chunked.phase === 'assembling' ? 'Assembling Video...' :
               'Uploading Video...'}
            </h2>
            {file && (
              <p className="text-[10px] text-gray-500 mt-1 font-mono">{file.name} · {fileSizeDisplay}</p>
            )}
          </>
        ) : (
          <>
            <div className="text-emerald-500 mb-3">
              <Loader2 size={40} className="animate-spin" />
            </div>
            <h2 className="text-lg font-bold text-emerald-400 tracking-tight font-mono uppercase">
              Processing Pipeline...
            </h2>
          </>
        )}
      </div>

      {/* ── Upload Progress Bar ──────────────────────────────────────── */}
      {showUploadProgress && (
        <div className="bg-[#111a12] border border-[#1a2420] rounded-xl p-6 shadow-2xl animate-fade-in-up">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xs font-bold text-gray-500 uppercase tracking-widest font-mono">
              Upload Progress
            </h3>
            <button
              onClick={() => chunked.cancelUpload()}
              className="flex items-center gap-1.5 text-[10px] font-mono text-red-400/70 hover:text-red-400 transition-colors uppercase tracking-wider"
            >
              <XCircle size={12} />
              Cancel
            </button>
          </div>

          {/* Progress bar */}
          <div className="relative w-full h-3 bg-gray-800/80 rounded-full overflow-hidden mb-4">
            <div
              className="absolute inset-y-0 left-0 rounded-full transition-all duration-300 ease-out"
              style={{
                width: `${Math.max(chunked.progress, 0.5)}%`,
                background: chunked.phase === 'assembling'
                  ? 'linear-gradient(90deg, #059669, #10b981, #34d399)'
                  : 'linear-gradient(90deg, #065f46, #059669, #10b981)',
              }}
            />
            {/* Shimmer effect */}
            {chunked.phase === 'uploading' && (
              <div
                className="absolute inset-y-0 left-0 rounded-full opacity-30"
                style={{
                  width: `${chunked.progress}%`,
                  background: 'linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.3) 50%, transparent 100%)',
                  animation: 'shimmer 2s ease-in-out infinite',
                }}
              />
            )}
          </div>

          {/* Stats row */}
          <div className="flex items-center justify-between text-[11px] font-mono">
            <div className="flex items-center gap-4">
              {/* Percentage */}
              <span className="text-emerald-400 font-bold text-sm tabular-nums">
                {chunked.progress.toFixed(1)}%
              </span>

              {/* Chunks */}
              <span className="flex items-center gap-1 text-gray-500">
                <HardDrive size={11} />
                <span className="tabular-nums">{chunked.chunksUploaded}/{chunked.totalChunks}</span>
                <span className="text-gray-700">chunks</span>
              </span>
            </div>

            <div className="flex items-center gap-4">
              {/* Speed */}
              <span className="flex items-center gap-1 text-gray-500">
                <Wifi size={11} />
                <span className="tabular-nums">{chunked.speed}</span>
              </span>

              {/* ETA */}
              <span className="flex items-center gap-1 text-gray-500">
                <Clock size={11} />
                <span className="tabular-nums">{chunked.timeRemaining}</span>
              </span>
            </div>
          </div>

          {/* Assembling indicator */}
          {chunked.phase === 'assembling' && (
            <div className="flex items-center gap-2 mt-4 text-emerald-400 text-xs font-mono">
              <Zap size={14} className="animate-pulse" />
              Reassembling video on server... This takes a moment for large files.
            </div>
          )}
        </div>
      )}

      {/* ── Error State ──────────────────────────────────────────────────── */}
      {showError && (
        <div className="bg-[#1a1212] border border-red-900/40 rounded-xl p-5 shadow-2xl">
          <div className="flex items-start gap-3">
            <AlertCircle size={18} className="text-red-400 flex-shrink-0 mt-0.5" />
            <div className="flex-1 min-w-0">
              <p className="text-red-400 font-mono text-sm font-bold mb-1">
                {uploadError ? 'Upload Failed' : 'Pipeline Error'}
              </p>
              <p className="text-red-300/70 font-mono text-xs break-words">
                {uploadError || pipelineError}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3 mt-4">
            {uploadError && (
              <button
                onClick={() => chunked.retryUpload()}
                className="flex items-center gap-1.5 px-4 py-2 bg-emerald-600/20 hover:bg-emerald-600/30 border border-emerald-600/40 rounded-lg text-emerald-400 text-xs font-mono font-bold uppercase tracking-wider transition-all"
              >
                <RefreshCw size={13} />
                Retry Upload
              </button>
            )}
            <button
              onClick={() => {
                chunked.reset();
                setFile(null);
                setActiveJobId(null);
              }}
              className="flex items-center gap-1.5 px-4 py-2 bg-gray-800/40 hover:bg-gray-800/60 border border-gray-700/40 rounded-lg text-gray-400 text-xs font-mono uppercase tracking-wider transition-all"
            >
              Start Over
            </button>
          </div>
        </div>
      )}

      {/* ── Configuration Selectors ──────────────────────────────────── */}
      {isIdle && (
        <div className="w-full flex flex-col sm:flex-row items-center justify-center gap-6 animate-fade-in-up mt-2">

          <div className="flex items-center gap-3">
            <label className="text-[10px] font-bold text-gray-500 uppercase tracking-widest font-mono">Quality Profile</label>
            <select
              defaultValue={typeof window !== 'undefined' ? localStorage.getItem('gaffer-quality-profile') || 'balanced' : 'balanced'}
              onChange={(e) => {
                if (typeof window !== 'undefined') localStorage.setItem('gaffer-quality-profile', e.target.value);
              }}
              className="bg-[#111a12] border border-gray-800 rounded-lg px-4 py-2 text-gray-300 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500/50 transition-all font-mono text-xs appearance-none cursor-pointer"
            >
              <option value="fast">Fast (Preview)</option>
              <option value="balanced">Balanced (Standard)</option>
              <option value="high_res">High Res (Detailed)</option>
              <option value="sahi">SAHI (Maximum Ball Recall)</option>
            </select>
          </div>

          <div className="flex items-center gap-3">
            <label className="text-[10px] font-bold text-gray-500 uppercase tracking-widest font-mono">Chunking Interval</label>
            <select
              defaultValue={typeof window !== 'undefined' ? localStorage.getItem('gaffer-chunking-interval') || '15-minute intervals' : '15-minute intervals'}
              onChange={(e) => {
                if (typeof window !== 'undefined') localStorage.setItem('gaffer-chunking-interval', e.target.value);
              }}
              className="bg-[#111a12] border border-gray-800 rounded-lg px-4 py-2 text-gray-300 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500/50 transition-all font-mono text-xs appearance-none cursor-pointer"
            >
              <option value="15-minute intervals">15-minute intervals</option>
              <option value="30-minute intervals">30-minute intervals</option>
              <option value="Full Halves (45 mins)">Full Halves (45 mins)</option>
            </select>
          </div>

        </div>
      )}

      {/* ── Pipeline Progress (post-upload processing) ───────────────── */}
      {showPipelineProgress && (
        <div className="bg-[#111a12] border border-[#1a2420] rounded-xl p-6 shadow-2xl mt-4">
          <h3 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-3 border-b border-gray-800 pb-3">Telemetry Pipeline Tracker</h3>
          <p className="text-[11px] font-mono text-emerald-300/90 mb-4">
            Backend step: <span className="text-gray-100">{currentStep}</span>
            {stepIndexInMilestones === -1 && currentStep !== 'Pending' && currentStep !== 'Completed' ? (
              <span className="ml-2 text-gray-500">(not in milestone list)</span>
            ) : null}
          </p>
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

      {/* Shimmer animation keyframes */}
      <style jsx>{`
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(200%); }
        }
      `}</style>
    </div>
  );
}
