"use client";
import React, { useState } from 'react';
import { Titlebar } from './components/Titlebar';
import { Sidebar } from './components/Sidebar';
import { Hopper } from './components/Hopper';
import { TacticalDashboard } from './components/TacticalDashboard';

import { ReportsArchive } from './components/ReportsArchive';

import { getTracking } from '@/lib/api/jobs';

export default function WorkspacePage() {
  const [ingestionComplete, setIngestionComplete] = useState(false);
  const [currentView, setCurrentView] = useState<'dashboard' | 'reports'>('dashboard');
  const [activeJob, setActiveJob] = useState<{jobId: string, file: File, tracking: any} | null>(null);

  const handleIngestionComplete = async (jobId: string, file: File) => {
    try {
      const tracking = await getTracking(jobId);
      setActiveJob({ jobId, file, tracking });
      setIngestionComplete(true);
    } catch (e) {
      console.error(e);
      setActiveJob({ jobId, file, tracking: null });
      setIngestionComplete(true);
    }
  };

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
            <TacticalDashboard job={activeJob} />
          )}
        </div>
      </div>
    </div>
  );
}
