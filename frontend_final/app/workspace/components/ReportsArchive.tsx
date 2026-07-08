"use client";
import React, { useState, useEffect, useRef } from 'react';

import { 
  Database, Search, FolderOpen, Calendar, ArrowRight, Loader2, 
  Trash2, Film, RefreshCw, AlertTriangle, AlertCircle, X
} from 'lucide-react';
import { listMatches, deleteMatch, reanalyzeMatch, getReport } from '@/lib/api/jobs';
import { getApiBaseUrl, getAuthHeaders } from '@/lib/apiBase';

// ── In-app error / confirm dialogs ────────────────────────────────────────────

function ErrorCard({
  title,
  message,
  onRetry,
  onDismiss,
}: {
  title: string;
  message: string;
  onRetry?: () => void;
  onDismiss: () => void;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-[#0d160e] border border-emerald-900/40 rounded-2xl shadow-2xl p-8 max-w-sm w-full mx-4 relative">
        <button
          onClick={onDismiss}
          className="absolute top-4 right-4 text-gray-600 hover:text-gray-400 transition-colors"
        >
          <X size={16} />
        </button>
        <div className="flex items-start gap-4 mb-6">
          <div className="p-2.5 bg-emerald-500/10 rounded-xl shrink-0">
            <AlertCircle size={20} className="text-emerald-500" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-gray-100 mb-1">{title}</h3>
            <p className="text-xs text-gray-400 leading-relaxed">{message}</p>
          </div>
        </div>
        <div className="flex gap-3">
          {onRetry && (
            <button
              onClick={() => { onDismiss(); onRetry(); }}
              className="flex-1 py-2.5 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-semibold hover:bg-emerald-500/20 transition-colors"
            >
              Retry
            </button>
          )}
          <button
            onClick={onDismiss}
            className="flex-1 py-2.5 rounded-xl bg-gray-900 border border-gray-800 text-gray-400 text-xs font-semibold hover:bg-gray-800 transition-colors"
          >
            {onRetry ? 'Back to Reports' : 'Dismiss'}
          </button>
        </div>
      </div>
    </div>
  );
}

function ConfirmCard({
  message,
  onConfirm,
  onCancel,
  confirmLabel = 'Confirm',
  danger = false,
}: {
  message: string;
  onConfirm: () => void;
  onCancel: () => void;
  confirmLabel?: string;
  danger?: boolean;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-[#0d160e] border border-gray-800 rounded-2xl shadow-2xl p-8 max-w-sm w-full mx-4">
        <div className="flex items-start gap-4 mb-6">
          <div className="p-2.5 bg-amber-500/10 rounded-xl shrink-0">
            <AlertTriangle size={20} className="text-amber-400" />
          </div>
          <p className="text-xs text-gray-300 leading-relaxed pt-1">{message}</p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={onConfirm}
            className={`flex-1 py-2.5 rounded-xl text-xs font-semibold transition-colors ${
              danger
                ? 'bg-red-500/10 border border-red-500/30 text-red-400 hover:bg-red-500/20'
                : 'bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/20'
            }`}
          >
            {confirmLabel}
          </button>
          <button
            onClick={onCancel}
            className="flex-1 py-2.5 rounded-xl bg-gray-900 border border-gray-800 text-gray-400 text-xs font-semibold hover:bg-gray-800 transition-colors"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Dialog state types ─────────────────────────────────────────────────────────

type ErrorState = { title: string; message: string; onRetry?: () => void } | null;
type ConfirmState = { message: string; onConfirm: () => void; confirmLabel?: string; danger?: boolean } | null;

// ── Main component ─────────────────────────────────────────────────────────────

export function ReportsArchive({ onOpenReport }: { onOpenReport: (report: any) => void }) {
  const [matches, setMatches] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState<'newest' | 'oldest'>('newest');
  const [downloadingVideo, setDownloadingVideo] = useState<string | null>(null);
  const [actionInProgress, setActionInProgress] = useState<string | null>(null);
  const [errorState, setErrorState] = useState<ErrorState>(null);
  const [confirmState, setConfirmState] = useState<ConfirmState>(null);

  const matchesRef = useRef<any[]>(matches);
  useEffect(() => { matchesRef.current = matches; }, [matches]);

  async function loadMatches(showLoader = true) {
    if (showLoader) setLoading(true);
    try {
      const data = await listMatches(searchTerm, sortBy);
      setMatches(data);
    } catch (error) {
      console.error("Failed to load matches archive:", error);
    } finally {
      if (showLoader) setLoading(false);
    }
  }

  useEffect(() => {
    loadMatches();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const interval = setInterval(() => {
      const activeJobs = matchesRef.current.some(
        (m) => m.status === 'processing' || m.status === 'pending'
      );
      if (activeJobs) {
        loadMatches(false);
      }
    }, 4000);
    return () => clearInterval(interval);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    loadMatches(true);
  }, [searchTerm, sortBy]); // eslint-disable-line react-hooks/exhaustive-deps

  async function handleOpen(matchId: string, status: string) {
    if (status !== 'completed' && status !== 'done') {
      setErrorState({
        title: 'Analysis in Progress',
        message: `This match is currently ${status}. Please wait for analysis to complete before opening.`,
      });
      return;
    }

    try {
      const fullReport = await getReport(matchId);
      onOpenReport(fullReport);
    } catch (error) {
      console.error("Failed to open report:", error);
      setErrorState({
        title: 'Unable to load analysis report.',
        message: 'The report may still be processing or could not be retrieved.',
        onRetry: () => handleOpen(matchId, status),
      });
    }
  }

  async function handleDelete(e: React.MouseEvent, matchId: string) {
    e.stopPropagation();
    setConfirmState({
      message: 'Are you sure you want to delete this match record and all associated files? This cannot be undone.',
      confirmLabel: 'Delete',
      danger: true,
      onConfirm: async () => {
        setConfirmState(null);
        setActionInProgress(matchId);
        try {
          await deleteMatch(matchId);
          await loadMatches(false);
        } catch (error) {
          console.error("Failed to delete match:", error);
          setErrorState({
            title: 'Delete Failed',
            message: 'Failed to delete the match record. Please try again.',
          });
        } finally {
          setActionInProgress(null);
        }
      },
    });
  }

  async function handleReanalyze(e: React.MouseEvent, matchId: string) {
    e.stopPropagation();
    setConfirmState({
      message: 'Are you sure you want to re-run the analysis for this match using the existing video file?',
      confirmLabel: 'Re-analyze',
      onConfirm: async () => {
        setConfirmState(null);
        setActionInProgress(matchId);
        try {
          await reanalyzeMatch(matchId);
          await loadMatches(false);
        } catch (error) {
          console.error("Failed to start reanalysis:", error);
          setErrorState({
            title: 'Reanalysis Failed',
            message: error instanceof Error ? error.message : 'Failed to trigger reanalysis. Ensure the source video file still exists in the uploads directory.',
          });
        } finally {
          setActionInProgress(null);
        }
      },
    });
  }

  async function handleDownloadVideo(e: React.MouseEvent, jobId: string) {
    e.stopPropagation();
    if (!jobId || jobId === 'unknown' || jobId === 'manual') {
      setErrorState({
        title: 'No Video Available',
        message: 'No video is associated with this match.',
      });
      return;
    }

    setDownloadingVideo(jobId);
    try {
      const base = getApiBaseUrl();
      const downloadUrl = `${base}/api/v1/elite/jobs/${jobId}/video/download`;
      const response = await fetch(downloadUrl, { headers: getAuthHeaders() });

      if (!response.ok) {
        throw new Error(`Video download failed: ${response.status}`);
      }

      const blob = await response.blob();
      const blobUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = blobUrl;
      link.setAttribute("download", `GaffersGuide_TacticalRadar_${jobId}.mp4`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      setTimeout(() => URL.revokeObjectURL(blobUrl), 5000);
    } catch (error) {
      console.error("Failed to download video:", error);
      setErrorState({
        title: 'Download Failed',
        message: 'Failed to download the video. The video may not have been generated for this analysis.',
      });
    } finally {
      setDownloadingVideo(null);
    }
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed':
      case 'done':
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            COMPLETED
          </span>
        );
      case 'pending':
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20 animate-pulse">
            QUEUED
          </span>
        );
      case 'processing':
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-blue-500/10 text-blue-400 border border-blue-500/20 flex items-center gap-1">
            <Loader2 size={10} className="animate-spin" /> ANALYZING
          </span>
        );
      case 'failed':
      case 'error':
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-red-500/10 text-red-400 border border-red-500/20">
            FAILED
          </span>
        );
      default:
        return (
          <span className="px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-gray-800 text-gray-400">
            UNKNOWN
          </span>
        );
    }
  };

  return (
    <>
      {/* In-app dialogs */}
      {errorState && (
        <ErrorCard
          title={errorState.title}
          message={errorState.message}
          onRetry={errorState.onRetry}
          onDismiss={() => setErrorState(null)}
        />
      )}
      {confirmState && (
        <ConfirmCard
          message={confirmState.message}
          confirmLabel={confirmState.confirmLabel}
          danger={confirmState.danger}
          onConfirm={confirmState.onConfirm}
          onCancel={() => setConfirmState(null)}
        />
      )}

      <div className="h-full w-full bg-[#050805] flex flex-col font-sans p-8 overflow-y-auto custom-scrollbar">
        {/* Header */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8 border-b border-gray-900 pb-6">
          <div>
            <h1 className="text-2xl font-bold text-gray-100 tracking-tight flex items-center gap-3">
              <FolderOpen className="text-emerald-500" />
              Match Reports
            </h1>
            <p className="text-sm text-gray-500 mt-1.5">
              Revisit completed analyses, download visual feeds, or re-run pipelines.
            </p>
          </div>

          {/* Search and Sort */}
          <div className="flex flex-wrap items-center gap-3 w-full md:w-auto">
            <div className="relative flex-1 md:w-60">
              <input
                id="match-search"
                type="text"
                placeholder="Search matches..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="bg-black/40 border border-gray-800 rounded-xl pl-10 pr-4 py-2 text-xs text-gray-300 focus:outline-none focus:border-emerald-500/50 transition-colors w-full font-mono placeholder-gray-700"
              />
              <Search size={13} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-gray-600" />
            </div>

            <select
              id="match-sort"
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="bg-black/40 border border-gray-800 rounded-xl px-3 py-2 text-xs font-mono text-gray-400 focus:outline-none focus:border-emerald-500/50 transition-colors"
            >
              <option value="newest">Newest first</option>
              <option value="oldest">Oldest first</option>
            </select>
          </div>
        </div>

        {loading ? (
          <div className="flex-1 flex flex-col items-center justify-center gap-3">
            <Loader2 className="animate-spin text-emerald-500" size={36} />
            <p className="text-gray-600 text-xs font-mono uppercase tracking-widest">Loading reports...</p>
          </div>
        ) : matches.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center border border-dashed border-gray-800 rounded-2xl p-12 gap-4">
            <Database className="text-gray-700" size={48} />
            <div className="text-center">
              <p className="text-gray-400 font-medium text-sm">No match reports yet</p>
              <p className="text-gray-600 text-xs mt-1">Upload match footage in the Dashboard to start tracking.</p>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
            {matches.map((match) => {
              const isCompleted = match.status === 'completed' || match.status === 'done';
              const isFailed = match.status === 'failed' || match.status === 'error';
              const isProcessing = match.status === 'processing' || match.status === 'pending';

              return (
                <div
                  key={match.id}
                  onClick={() => handleOpen(match.id, match.status)}
                  className={`group border rounded-2xl p-5 flex flex-col justify-between min-h-[260px] shadow-lg transition-all ${
                    isCompleted
                      ? 'bg-[#111a12]/10 border-gray-800 hover:border-emerald-500/35 hover:bg-[#152018]/30 cursor-pointer'
                      : isFailed
                      ? 'bg-red-950/5 border-red-900/20 hover:border-red-900/40 cursor-default'
                      : 'bg-blue-950/5 border-blue-900/20 hover:border-blue-900/40 cursor-default'
                  }`}
                >
                  {/* Card Header */}
                  <div>
                    <div className="flex justify-between items-start mb-4">
                      <div className="flex items-center gap-2">
                        <div className={`p-2 rounded-lg ${
                          isCompleted ? 'bg-emerald-500/10 text-emerald-500' : isFailed ? 'bg-red-500/10 text-red-500' : 'bg-blue-500/10 text-blue-500'
                        }`}>
                          <Database size={15} />
                        </div>
                        {getStatusBadge(match.status)}
                      </div>

                      <div className="flex items-center gap-1">
                        {isCompleted && (
                          <button
                            onClick={(e) => handleDownloadVideo(e, match.id)}
                            disabled={downloadingVideo === match.id}
                            className="text-gray-600 hover:text-emerald-400 p-1.5 rounded hover:bg-emerald-500/10 transition-colors disabled:opacity-40"
                            title="Download Tactical Video"
                          >
                            {downloadingVideo === match.id ? (
                              <Loader2 size={12} className="animate-spin" />
                            ) : (
                              <Film size={12} />
                            )}
                          </button>
                        )}
                        <button
                          onClick={(e) => handleReanalyze(e, match.id)}
                          disabled={actionInProgress === match.id || isProcessing}
                          className="text-gray-600 hover:text-emerald-400 p-1.5 rounded hover:bg-emerald-500/10 transition-colors disabled:opacity-40"
                          title="Re-run pipeline analysis"
                        >
                          {actionInProgress === match.id ? (
                            <Loader2 size={12} className="animate-spin" />
                          ) : (
                            <RefreshCw size={12} />
                          )}
                        </button>
                        <button
                          onClick={(e) => handleDelete(e, match.id)}
                          disabled={actionInProgress === match.id || isProcessing}
                          className="text-gray-600 hover:text-red-500 p-1.5 rounded hover:bg-red-500/10 transition-colors disabled:opacity-40"
                          title="Delete Match Record"
                        >
                          <Trash2 size={12} />
                        </button>
                      </div>
                    </div>

                    <h3 className="font-semibold text-gray-200 text-sm mb-1.5 truncate leading-snug" title={match.name}>
                      {match.name}
                    </h3>

                    <div className="flex flex-wrap items-center gap-x-3 text-[10px] font-mono text-gray-600 uppercase tracking-tight mb-4">
                      <span className="flex items-center gap-1"><Calendar size={10} /> {new Date(match.upload_date).toLocaleDateString()}</span>
                      <span>•</span>
                      <span>{match.quality_profile}</span>
                    </div>

                    {/* Core Metrics */}
                    {isCompleted && (
                      <div className="space-y-2 bg-black/40 border border-gray-900/60 rounded-xl p-3.5 mb-4 text-[10px] font-mono">
                        <div className="flex justify-between items-center">
                          <span className="text-gray-500 uppercase text-[9px]">Tactical Power (TPI)</span>
                          <span className="text-gray-300 font-bold">
                            Red <span className="text-emerald-400">{Math.round(match.tactical_power_red)}</span> / Blue <span className="text-emerald-400">{Math.round(match.tactical_power_blue)}</span>
                          </span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-gray-500 uppercase text-[9px]">Shape Compactness</span>
                          <span className="text-gray-300">
                            Red <span className="font-bold">{Math.round(match.compactness_red)}%</span> / Blue <span className="font-bold">{Math.round(match.compactness_blue)}%</span>
                          </span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-gray-500 uppercase text-[9px]">Transition Velocity</span>
                          <span className="text-gray-300">
                            Red <span className="font-bold">{Math.round(match.transition_speed_red)}</span> / Blue <span className="font-bold">{Math.round(match.transition_speed_blue)}</span>
                          </span>
                        </div>
                      </div>
                    )}

                    {/* Error State */}
                    {isFailed && (
                      <div className="bg-red-500/5 border border-red-500/10 rounded-xl p-3 mb-4 text-[10px] font-mono text-red-400/80 flex items-start gap-1.5">
                        <AlertTriangle size={12} className="shrink-0 mt-0.5 text-red-400" />
                        <div className="break-words max-h-16 overflow-y-auto">
                          {match.error_message || "Pipeline initialization error."}
                        </div>
                      </div>
                    )}

                    {/* Processing State */}
                    {isProcessing && (
                      <div className="bg-blue-500/5 border border-blue-500/10 rounded-xl p-3.5 mb-4 text-[10px] font-mono text-blue-400/80 flex items-center gap-2">
                        <Loader2 size={12} className="animate-spin text-blue-400" />
                        <span>Job status: {match.error_message || "Ingesting match feeds..."}</span>
                      </div>
                    )}
                  </div>

                  {isCompleted && (
                    <div className="mt-auto flex items-center justify-between border-t border-gray-900 pt-3">
                      <span className="text-[10px] font-bold text-emerald-500 uppercase tracking-widest">Open Analysis</span>
                      <ArrowRight size={14} className="text-gray-600 group-hover:text-emerald-400 transition-colors" />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </>
  );
}
