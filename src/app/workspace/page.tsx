"use client";
import React, { useState } from 'react';
import { Titlebar } from './components/Titlebar';
import { Sidebar } from './components/Sidebar';
import { Hopper } from './components/Hopper';
import { TacticalDashboard } from './components/TacticalDashboard';

import { ReportsArchive } from './components/ReportsArchive';

export default function WorkspacePage() {
  const [ingestionComplete, setIngestionComplete] = useState(false);
  const [currentView, setCurrentView] = useState<'dashboard' | 'reports'>('dashboard');

  return (
    <div className="h-screen w-screen overflow-hidden flex flex-col bg-[#0a0f0a] text-gray-300 antialiased selection:bg-emerald-500/30 selection:text-emerald-300">
      <Titlebar />
      <div className="flex flex-1 overflow-hidden relative">
        <Sidebar currentView={currentView} setCurrentView={setCurrentView} />
        <div className="flex-1 relative flex flex-col min-w-0 bg-[#050805]">
          {currentView === 'reports' ? (
            <ReportsArchive />
          ) : !ingestionComplete ? (
            <Hopper onComplete={() => setIngestionComplete(true)} />
          ) : (
            <TacticalDashboard />
          )}
        </div>
      </div>
    </div>
  );
}
