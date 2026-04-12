"use client";
import React, { useState } from "react";

import { EngineProvider } from "@/context/EngineContext";
import { type CoachAdviceResponse, type JobArtifactsResponse } from "@/lib/api";

import { Hopper } from "./components/Hopper";
import { ReportsArchive } from "./components/ReportsArchive";
import { Sidebar } from "./components/Sidebar";
import { TacticalDashboard } from "./components/TacticalDashboard";
import { Titlebar } from "./components/Titlebar";

interface WorkspaceSession {
  jobId: string;
  artifacts: JobArtifactsResponse | null;
  advice: CoachAdviceResponse | null;
}

export default function WorkspacePage(): React.JSX.Element {
  const [ingestionComplete, setIngestionComplete] = useState(false);
  const [currentView, setCurrentView] = useState<"dashboard" | "reports">("dashboard");
  const [session, setSession] = useState<WorkspaceSession | null>(null);

  return (
    <EngineProvider>
    <div className="h-screen w-screen overflow-hidden flex flex-col bg-[#0a0f0a] text-gray-300 antialiased selection:bg-emerald-500/30 selection:text-emerald-300">
      <Titlebar />
      <div className="flex flex-1 overflow-hidden relative">
        <Sidebar currentView={currentView} setCurrentView={setCurrentView} />
        <div className="flex-1 relative flex flex-col min-w-0 bg-[#050805]">
          {currentView === "reports" ? (
            <ReportsArchive activeJobId={session?.jobId ?? null} />
          ) : !ingestionComplete ? (
            <Hopper
              onComplete={(payload) => {
                setSession(payload);
                setIngestionComplete(true);
              }}
            />
          ) : (
            <TacticalDashboard
              jobId={session?.jobId ?? null}
              initialAdvice={session?.advice ?? null}
              artifacts={session?.artifacts ?? null}
            />
          )}
        </div>
      </div>
    </div>
    </EngineProvider>
  );
}
