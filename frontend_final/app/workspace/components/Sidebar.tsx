"use client";
import React, { useState } from 'react';
import { LayoutDashboard, FileText, Settings, Database, Server, RefreshCw, Menu, ChevronLeft, ChevronRight } from 'lucide-react';
import { EngineSettingsModal } from './EngineSettingsModal';

export function Sidebar({ 
  currentView = 'dashboard', 
  setCurrentView = () => {} 
}: { 
  currentView?: 'dashboard' | 'reports', 
  setCurrentView?: (v: 'dashboard' | 'reports') => void 
}) {
  const [isEngineSettingsOpen, setIsEngineSettingsOpen] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(false);

  return (
    <>
      <div className={`${isCollapsed ? 'w-[68px]' : 'w-[240px]'} flex-shrink-0 bg-[#0a0f0a] border-r border-gray-900 flex flex-col h-full font-mono relative z-10 transition-all duration-300 overflow-hidden`}>
        
        {/* Toggle / Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-900/50">
           {!isCollapsed && <div className="text-[10px] font-bold text-gray-500 uppercase tracking-widest pl-1 whitespace-nowrap">Workspace</div>}
           <button 
             onClick={() => setIsCollapsed(!isCollapsed)}
             className={`p-1.5 rounded-md hover:bg-gray-800 text-gray-500 hover:text-emerald-400 transition-colors ${isCollapsed ? 'mx-auto' : ''}`}>
             <Menu size={16} />
           </button>
        </div>

        {/* Navigation */}
        <div className="flex-1 py-4">
          <nav className="space-y-1">
            <button 
              onClick={() => setCurrentView('dashboard')}
              title="Dashboard"
              className={`w-full flex items-center gap-3 py-2.5 transition-colors ${
                currentView === 'dashboard' ? 'bg-[#111a12] text-gray-100 border-r-2 border-emerald-500' : 'text-gray-400 hover:text-gray-200 hover:bg-gray-900/50'
              } ${isCollapsed ? 'justify-center px-0' : 'px-6'}`}>
              <LayoutDashboard size={18} className={currentView === 'dashboard' ? 'text-emerald-500' : ''} />
              {!isCollapsed && <span className={`text-sm ${currentView === 'dashboard' ? 'font-semibold' : ''}`}>Dashboard</span>}
            </button>
            <button 
              onClick={() => setCurrentView('reports')}
              title="Reports"
              className={`w-full flex items-center gap-3 py-2.5 transition-colors ${
                currentView === 'reports' ? 'bg-[#111a12] text-gray-100 border-r-2 border-emerald-500' : 'text-gray-400 hover:text-gray-200 hover:bg-gray-900/50'
              } ${isCollapsed ? 'justify-center px-0' : 'px-6'}`}>
              <FileText size={18} className={currentView === 'reports' ? 'text-emerald-500' : ''} />
              {!isCollapsed && <span className={`text-sm ${currentView === 'reports' ? 'font-semibold' : ''}`}>Reports</span>}
            </button>
            <button 
              onClick={() => setIsEngineSettingsOpen(true)}
              title="Engine Settings"
              className={`w-full flex items-center gap-3 py-2.5 text-gray-400 hover:text-gray-200 hover:bg-gray-900/50 transition-colors ${isCollapsed ? 'justify-center px-0' : 'px-6'}`}>
              <Settings size={18} />
              {!isCollapsed && <span className="text-sm">Engine Settings</span>}
            </button>
          </nav>
        </div>

        {/* Status Indicators */}
        <div className={`border-t border-gray-900 bg-[#0a0f0a] ${isCollapsed ? 'p-2 flex justify-center' : 'p-4'}`}>
          {!isCollapsed ? (
            <>
              <div className="flex items-center justify-between text-xs mb-3">
                <span className="text-gray-500">Telemetry Engine</span>
                <span className="text-emerald-500 font-bold flex items-center gap-1.5 bg-emerald-500/10 px-2 py-0.5 rounded">
                  <Server size={10} /> LOCAL IDLE
                </span>
              </div>
              <div className="h-0.5 w-full bg-gray-800 overflow-hidden">
                <div className="h-full bg-emerald-500 w-1/4"></div>
              </div>
            </>
          ) : (
            <div title="Engine Status: LOCAL IDLE" className="my-2">
               <Server size={18} className="text-emerald-500" />
            </div>
          )}
        </div>
      </div>

      {/* Overlays */}
      <EngineSettingsModal 
        isOpen={isEngineSettingsOpen} 
        onClose={() => setIsEngineSettingsOpen(false)} 
      />
    </>
  );
}
