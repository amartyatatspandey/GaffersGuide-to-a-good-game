"use client";
import React from 'react';
import { Database, Search, FolderOpen, Calendar, ArrowRight } from 'lucide-react';

const MOCK_REPORTS = [
  { id: 1, title: 'MCI_vs_ARS_WK5.mp4', date: 'Oct 14, 2026', duration: '94:12', flags: ['Midfield Disconnect', 'High Press Exploited'], score: '2 - 1', status: 'analyzed' },
  { id: 2, title: 'LIV_vs_CHE_FINAL.mp4', date: 'Oct 02, 2026', duration: '120:00', flags: ['Defensive Block', 'Transition Speed'], score: '1 - 0', status: 'analyzed' },
  { id: 3, title: 'RM_vs_BAR_ELCL.mp4', date: 'Sep 28, 2026', duration: '96:45', flags: ['False 9 Impact', 'Overloads'], score: '3 - 2', status: 'analyzed' },
  { id: 4, title: 'BAY_vs_DOR_DERB.mp4', date: 'Sep 15, 2026', duration: '92:10', flags: ['Wing Play', 'Set Piece Vuln.'], score: '2 - 2', status: 'analyzed' },
];

export function ReportsArchive() {
  return (
    <div className="h-full w-full bg-[#0a0f0a] flex flex-col animate-fade-in font-sans p-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-200 tracking-tight flex items-center gap-3">
            <FolderOpen className="text-emerald-500" />
            Telemetry Archive
          </h1>
          <p className="text-sm text-gray-500 mt-2 font-mono">Historical match analysis and tactical debriefs.</p>
        </div>
        
        <div className="relative">
          <input 
            type="text" 
            placeholder="Search archives..." 
            className="bg-[#111a12] border border-gray-800 rounded-lg pl-10 pr-4 py-2 text-sm text-gray-300 focus:outline-none focus:border-emerald-500 transition-colors w-64"
          />
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-600" />
        </div>
      </div>

      {/* Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {MOCK_REPORTS.map((report) => (
          <div key={report.id} className="bg-[#111a12] border border-gray-900 hover:border-emerald-500/50 rounded-xl p-5 cursor-pointer transition-all hover:bg-[#152018] group flex flex-col justify-between min-h-[220px]">
            <div>
              <div className="flex justify-between items-start mb-4">
                <div className="p-2 bg-emerald-500/10 rounded-lg">
                  <Database size={20} className="text-emerald-500" />
                </div>
                <span className="text-xs font-bold text-gray-600 font-mono flex items-center gap-1">
                  <Calendar size={12}/> {report.date}
                </span>
              </div>
              <h3 className="font-bold text-gray-300 text-sm mb-1 truncate" title={report.title}>{report.title}</h3>
              <p className="text-xs font-mono text-gray-500 mb-4">{report.duration} • FT: {report.score}</p>
              
              <div className="flex flex-wrap gap-2">
                {report.flags.map(flag => (
                  <span key={flag} className="px-2 py-0.5 bg-gray-900 border border-gray-800 rounded text-[10px] uppercase tracking-wider text-amber-500/80 font-bold">
                    {flag}
                  </span>
                ))}
              </div>
            </div>

            <div className="mt-6 flex items-center justify-between border-t border-gray-900 pt-4">
               <span className="text-xs font-bold text-emerald-500 uppercase tracking-widest">Analyzed</span>
               <ArrowRight size={16} className="text-gray-600 group-hover:text-emerald-400 transition-colors transform group-hover:translate-x-1" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
