"use client";
import React, { useState, useEffect, useRef } from 'react';
import { Titlebar } from './components/Titlebar';
import { Sidebar } from './components/Sidebar';
import { Hopper } from './components/Hopper';
import { TacticalDashboard } from './components/TacticalDashboard';

import { ReportsArchive } from './components/ReportsArchive';

import { getTracking } from '@/lib/api/jobs';
import { getCoachAdvice } from '@/lib/api/coach';
import type { CoachAdviceResponse } from '@/lib/api/coach';
import { debugSessionLog } from '@/lib/debugSessionLog';

function classifyFrontendBackendError(err: unknown): string {
  const text = err instanceof Error ? err.message : String(err);
  const lowered = text.toLowerCase();
  if (
    lowered.includes('failed to fetch') ||
    lowered.includes('networkerror') ||
    lowered.includes('connection error')
  ) {
    return `Connectivity issue between frontend and backend: ${text}`;
  }
  return text;
}

export default function WorkspacePage() {
  const [ingestionComplete, setIngestionComplete] = useState(false);
  const [currentView, setCurrentView] = useState<'dashboard' | 'reports'>('dashboard');
  const [activeJob, setActiveJob] = useState<{ jobId: string; file: File; tracking: unknown } | null>(null);
  const [coachAdvice, setCoachAdvice] = useState<CoachAdviceResponse | null>(null);
  const [coachError, setCoachError] = useState<string | null>(null);
  const coachDelayedRefetchSent = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (!localStorage.getItem("gaffer-engine-type")) {
      localStorage.setItem("gaffer-engine-type", "local");
    }
    if (!localStorage.getItem("gaffer-ollama-model")) {
      localStorage.setItem("gaffer-ollama-model", "llama3");
    }
  }, []);

  const handleIngestionComplete = async (jobId: string, file: File) => {
    try {
      const tracking = await getTracking(jobId);
      // #region agent log
      debugSessionLog({
        sessionId: 'bb63ae',
        hypothesisId: 'H5',
        location: 'workspace/page.tsx:handleIngestionComplete',
        message: 'tracking ok',
        data: {
          jobIdPrefix: jobId.slice(0, 8),
          frameCount: Array.isArray(tracking?.frames) ? tracking.frames.length : -1,
        },
      });
      // #endregion
      setActiveJob({ jobId, file, tracking });
      setCoachAdvice(null);
      setCoachError(null);
      coachDelayedRefetchSent.current.clear();
      try {
        const llmPref =
          typeof window !== 'undefined' && localStorage.getItem('gaffer-engine-type') === 'cloud'
            ? 'cloud'
            : 'local';
        const advice = await getCoachAdvice(jobId, llmPref);
        setCoachAdvice(advice);
      } catch (ce) {
        const detail = classifyFrontendBackendError(ce);
        setCoachError(detail);
        debugSessionLog({
          sessionId: 'bb63ae',
          hypothesisId: 'H5',
          location: 'workspace/page.tsx:handleIngestionComplete',
          message: 'coach fetch failed',
          data: { jobIdPrefix: jobId.slice(0, 8), err: detail },
        });
      }
      setIngestionComplete(true);
    } catch (e) {
      console.error(e);
      const detail = classifyFrontendBackendError(e);
      // #region agent log
      debugSessionLog({
        sessionId: 'bb63ae',
        hypothesisId: 'H1-H5',
        location: 'workspace/page.tsx:handleIngestionComplete',
        message: 'tracking fetch failed',
        data: { jobIdPrefix: jobId.slice(0, 8), err: detail },
      });
      // #endregion
      setActiveJob({ jobId, file, tracking: null });
      setCoachAdvice(null);
      setCoachError(detail);
      setIngestionComplete(true);
    }
  };

  useEffect(() => {
    const jobId = activeJob?.jobId;
    if (!ingestionComplete || !jobId || coachError) return;
    if (coachAdvice === null) return;
    if ((coachAdvice.advice_items?.length ?? 0) > 0) return;
    if (coachDelayedRefetchSent.current.has(jobId)) return;
    let cancelled = false;
    const handle = window.setTimeout(() => {
      if (cancelled) return;
      coachDelayedRefetchSent.current.add(jobId);
      void (async () => {
        try {
          const llmPref =
            typeof window !== 'undefined' && localStorage.getItem('gaffer-engine-type') === 'cloud'
              ? 'cloud'
              : 'local';
          const advice = await getCoachAdvice(jobId, llmPref);
          setCoachAdvice(advice);
        } catch {
          /* ignore: primary coach path already surfaced errors */
        }
      })();
    }, 4000);
    return () => {
      cancelled = true;
      window.clearTimeout(handle);
    };
  }, [
    ingestionComplete,
    activeJob?.jobId,
    coachError,
    coachAdvice,
    coachAdvice?.advice_items?.length,
  ]);

  return (
    <div className="h-screen w-screen overflow-hidden flex flex-col bg-[#0a0f0a] text-gray-300 antialiased selection:bg-emerald-500/30 selection:text-emerald-300">
      <Titlebar />
      <div className="flex flex-1 overflow-hidden relative">
        <Sidebar currentView={currentView} setCurrentView={setCurrentView} />
        <div className="flex-1 relative flex flex-col min-w-0 bg-[#050805]">
          {currentView === 'reports' ? (
            <ReportsArchive />
          ) : !ingestionComplete ? (
            <Hopper onComplete={handleIngestionComplete} />
          ) : (
            <TacticalDashboard job={activeJob} coachAdvice={coachAdvice} coachError={coachError} />
          )}
        </div>
      </div>
    </div>
  );
}
