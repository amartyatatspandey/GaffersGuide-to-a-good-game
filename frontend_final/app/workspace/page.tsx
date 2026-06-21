"use client";
import React, { useState, useEffect, useRef } from 'react';
import { Titlebar } from './components/Titlebar';
import { Sidebar } from './components/Sidebar';
import { Hopper } from './components/Hopper';
import TacticalDashboard from './components/TacticalDashboard';
import { ReportsArchive } from './components/ReportsArchive';
import { DictionaryTab } from './components/DictionaryTab';
import { PlayerReports } from './components/PlayerReports';

import { getTracking } from '@/lib/api/jobs';
import { getCoachAdvice, getEnrichedReport } from '@/lib/api/coach';
import type { CoachAdviceResponse } from '@/lib/api/coach';
import { debugSessionLog } from '@/lib/debugSessionLog';

export default function WorkspacePage() {
  const [ingestionComplete, setIngestionComplete] = useState(false);
  const [currentView, setCurrentView] = useState<'dashboard' | 'reports' | 'players' | 'dictionary'>('dashboard');
  const [activeJob, setActiveJob] = useState<{ jobId: string; file: File; tracking: any; isHistorical?: boolean } | null>(null);
  const [coachAdvice, setCoachAdvice] = useState<CoachAdviceResponse | null>(null);
  const [coachError, setCoachError] = useState<string | null>(null);
  const coachDelayedRefetchSent = useRef<Set<string>>(new Set());

  // Dictionary remapping states
  const [useAltNames, setUseAltNames] = useState(false);
  const [dictionary, setDictionary] = useState<Record<string, string>>({});
  const [aiPromptOverride, setAiPromptOverride] = useState<string | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (!localStorage.getItem("gaffer-engine-type")) {
      localStorage.setItem("gaffer-engine-type", "cloud");
    }
    if (!localStorage.getItem("gaffer-ollama-model")) {
      localStorage.setItem("gaffer-ollama-model", "llama3");
    }

    // Load dictionary states
    const dict = localStorage.getItem("gaffer-dictionary");
    if (dict) {
      try {
        setDictionary(JSON.parse(dict));
      } catch (e) {
        console.error("Failed to parse dictionary", e);
      }
    }
    setUseAltNames(localStorage.getItem("gaffer-use-alt-names") === "true");
  }, []);

  const handleUpdateDictionary = (newDict: Record<string, string>) => {
    setDictionary(newDict);
    localStorage.setItem("gaffer-dictionary", JSON.stringify(newDict));
    if (typeof window !== "undefined") {
      window.dispatchEvent(new Event("gaffer-dictionary-changed"));
    }
  };

  const handleToggleAltNames = (val: boolean) => {
    setUseAltNames(val);
    localStorage.setItem("gaffer-use-alt-names", val ? "true" : "false");
    if (typeof window !== "undefined") {
      window.dispatchEvent(new Event("gaffer-dictionary-changed"));
    }
  };

  const handlePlayClipAndRedirect = (startTimeS: number) => {
    setCurrentView('dashboard');
    // We will find a way to notify the dashboard player. Custom event works beautifully!
    setTimeout(() => {
      if (typeof window !== "undefined") {
        const ev = new CustomEvent("gaffer-play-clip", { detail: { startTimeS } });
        window.dispatchEvent(ev);
      }
    }, 100);
  };

  const handleAskAIAndRedirect = (prompt: string) => {
    setAiPromptOverride(prompt);
    setCurrentView('dashboard');
  };

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
      setIngestionComplete(true);
      try {
        const llmPref =
          typeof window !== 'undefined' && localStorage.getItem('gaffer-engine-type') === 'cloud'
            ? 'cloud'
            : 'local';
        const advice = await getCoachAdvice(jobId, llmPref);
        try {
          const enrichedCards = await getEnrichedReport(jobId);
          if (enrichedCards && enrichedCards.length > 0) {
            advice.advice_items = enrichedCards;
          }
        } catch (enrichErr) {
          console.warn("Failed to load enriched report, using standard advice:", enrichErr);
        }
        setCoachAdvice(advice);
      } catch (ce) {
        setCoachError(ce instanceof Error ? ce.message : String(ce));
      }
    } catch (e) {
      console.error(e);
      // #region agent log
      debugSessionLog({
        sessionId: 'bb63ae',
        hypothesisId: 'H1-H5',
        location: 'workspace/page.tsx:handleIngestionComplete',
        message: 'tracking fetch failed',
        data: { jobIdPrefix: jobId.slice(0, 8), err: String(e) },
      });
      // #endregion
      setActiveJob({ jobId, file, tracking: null });
      setCoachAdvice(null);
      setCoachError(null);
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
          try {
            const enrichedCards = await getEnrichedReport(jobId);
            if (enrichedCards && enrichedCards.length > 0) {
              advice.advice_items = enrichedCards;
            }
          } catch (enrichErr) {
            console.warn("Delayed refetch failed to load enriched report:", enrichErr);
          }
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

  const handleOpenReport = (report: any) => {
    // Transform persistent report into active job state
    setActiveJob({
      jobId: report.job_id || report.id,
      file: new File([], report.video_title || 'Historical Match'),
      tracking: report.telemetry || null,
      isHistorical: true
    });
    setCoachAdvice({
      generated_at: report.metadata?.saved_at || new Date().toISOString(),
      pipeline: report.pipeline || {},
      advice_items: report.advice_items || []
    });
    setIngestionComplete(true);
    setCurrentView('dashboard');
  };

  return (
    <div className="h-screen w-screen overflow-hidden flex flex-col bg-[#0a0f0a] text-gray-300 antialiased selection:bg-emerald-500/30 selection:text-emerald-300">
      <Titlebar />
      <div className="flex flex-1 overflow-hidden relative">
        <Sidebar currentView={currentView} setCurrentView={setCurrentView} />
        <div className="flex-1 relative flex flex-col min-w-0 bg-[#050805]">
          {currentView === 'reports' ? (
            <ReportsArchive onOpenReport={handleOpenReport} />
          ) : currentView === 'dictionary' ? (
            <DictionaryTab 
              dictionary={dictionary} 
              useAltNames={useAltNames} 
              onUpdateDictionary={handleUpdateDictionary} 
              onToggleAltNames={handleToggleAltNames} 
            />
          ) : currentView === 'players' ? (
            <PlayerReports 
              job={activeJob} 
              useAltNames={useAltNames} 
              dictionary={dictionary} 
              onPlayClip={handlePlayClipAndRedirect} 
              onAskAI={handleAskAIAndRedirect} 
            />
          ) : !ingestionComplete ? (
            <Hopper onComplete={handleIngestionComplete} />
          ) : (
            <TacticalDashboard 
              job={activeJob} 
              coachAdvice={coachAdvice} 
              coachError={coachError}
              useAltNames={useAltNames}
              dictionary={dictionary}
              aiPromptOverride={aiPromptOverride}
              setAiPromptOverride={setAiPromptOverride}
            />
          )}
        </div>
      </div>
    </div>
  );
}
