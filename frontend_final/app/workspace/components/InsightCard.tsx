"use client";
import React, { useState } from 'react';
import { useStreamingText } from '@/hooks/useStreamingText';
import { TacticalGrid } from './TacticalGrid';
import { ChevronDown, ChevronUp } from 'lucide-react';

export interface KeywordConfig {
  text: string;
  color: 'emerald' | 'amber' | 'cyan';
  role: string;
}

interface TeamInsight {
  payload: string;
  keywords: KeywordConfig[];
}

interface InsightCardProps {
  title: string;
  minute: string;
  blueTeam: TeamInsight;
  redTeam: TeamInsight;
}

export function InsightCard({ title, minute, blueTeam, redTeam }: InsightCardProps) {
  const [activeTeam, setActiveTeam] = useState<'blue' | 'red'>('blue');
  const currentData = activeTeam === 'blue' ? blueTeam : redTeam;
  const { displayedText } = useStreamingText(currentData.payload, 20);
  const [activeRole, setActiveRole] = useState<string | null>(null);
  const [isGridOpen, setIsGridOpen] = useState(false);

  const handleRoleClick = (role: string) => {
    setActiveRole(role);
    setIsGridOpen(true);
  };

  const renderInterpolatedText = (text: string) => {
    if (currentData.keywords.length === 0) return <span>{text}</span>;
    const regexText = currentData.keywords.map(k => k.text).join('|');
    const regex = new RegExp(`(${regexText})`, 'gi');
    const parts = text.split(regex);

    return parts.map((part, i) => {
      const kw = currentData.keywords.find(k => k.text.toLowerCase() === part.toLowerCase());
      if (kw) {
        // Build color scheme
        const colorClasses = 
          kw.color === 'emerald' ? 'text-emerald-400 bg-emerald-900/30 shadow-[0_0_8px_rgba(52,211,153,0.4)] hover:bg-emerald-800' :
          kw.color === 'amber'   ? 'text-amber-400 bg-amber-900/30 shadow-[0_0_8px_rgba(251,191,36,0.4)] hover:bg-amber-800' :
          kw.color === 'cyan'    ? 'text-cyan-400 bg-cyan-900/30 shadow-[0_0_8px_rgba(34,211,238,0.4)] hover:bg-cyan-800' : '';

        return (
          <span 
            key={i} 
            onClick={() => handleRoleClick(kw.role)}
            className={`font-bold px-1.5 py-0.5 rounded cursor-pointer transition-all ${colorClasses}`}
          >
            {part}
          </span>
        );
      }
      return <span key={i}>{part}</span>;
    });
  };

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-xl p-6 shadow-lg mb-4 max-w-2xl relative overflow-hidden font-sans transition-all duration-500">
      {/* Header */}
      <div className="flex justify-between items-start mb-3">
        <h3 className="text-xl font-bold text-gray-200 flex items-center gap-2">
          <span>🚨</span> {title}
        </h3>
      </div>

      {/* Team Toggle */}
      <div className="flex mb-5 bg-[#0a0f0a] border border-gray-700/50 p-1 rounded-lg w-full max-w-[280px]">
        <button 
          onClick={() => setActiveTeam('blue')}
          className={`flex-1 py-1.5 text-xs font-bold uppercase tracking-wider rounded transition-all ${
            activeTeam === 'blue' ? 'bg-blue-600/20 text-blue-400 shadow-[0_0_10px_rgba(59,130,246,0.2)]' : 'text-gray-500 hover:text-gray-300'
          }`}
        >
          Team Blue
        </button>
        <button 
          onClick={() => setActiveTeam('red')}
          className={`flex-1 py-1.5 text-xs font-bold uppercase tracking-wider rounded transition-all ${
            activeTeam === 'red' ? 'bg-red-600/20 text-red-500 shadow-[0_0_10px_rgba(239,68,68,0.2)]' : 'text-gray-500 hover:text-gray-300'
          }`}
        >
          Team Red
        </button>
      </div>
      
      {/* Interactive Tactical Role Badges */}
      <div className="flex flex-wrap gap-2 mb-4">
        {Array.from(new Set(currentData.keywords.map(k => k.role))).map((uniqueRole, idx) => (
           <button 
             key={idx}
             onClick={() => handleRoleClick(uniqueRole)}
             className="px-3 py-1 bg-emerald-900/50 text-emerald-300 border border-emerald-800 rounded-full text-xs font-bold tracking-wide hover:bg-emerald-800 transition-colors cursor-pointer ring-2 ring-transparent hover:ring-emerald-400 shadow-[0_0_10px_rgba(16,185,129,0.2)] uppercase">
             {uniqueRole}
           </button>
        ))}
      </div>
      
      {/* Philosophical Anchor */}
      <blockquote className="border-l-4 border-gray-600 pl-3 mb-4 italic text-sm text-gray-400">
        "The intention is to overload the midfield to gain numerical superiority."<br/>
        <span className="font-semibold non-italic">- Pep Guardiola / Positional Play</span>
      </blockquote>
      
      {/* Actionable Instruction */}
      <p className="text-gray-200 leading-relaxed text-sm transition-all duration-300">
        {renderInterpolatedText(displayedText)}
        <span className="inline-block w-2 bg-emerald-500 animate-pulse h-4 align-middle ml-1 shadow-[0_0_5px_rgba(16,185,129,0.8)]"></span>
      </p>

      {/* Grid Expansion Toggle */}
      <div className="mt-6 flex justify-center border-t border-gray-700/50 pt-3">
         <button 
           onClick={() => setIsGridOpen(!isGridOpen)} 
           className="flex items-center gap-2 text-xs font-bold text-gray-500 hover:text-emerald-400 uppercase tracking-widest font-mono transition-colors"
         >
           {isGridOpen ? <ChevronUp size={14}/> : <ChevronDown size={14}/>}
           {isGridOpen ? 'Collapse Spatial Grid' : 'Expand Spatial Grid'}
         </button>
      </div>

      {/* Embedded Tactical Grid */}
      {isGridOpen && (
        <div id="tactical-grid-overlay" className="mt-4 h-[300px] w-full bg-gray-900 border border-gray-700 rounded-lg relative overflow-hidden transition-all duration-500 flex items-center justify-center p-2">
           <TacticalGrid activeRole={activeRole} />
        </div>
      )}
    </div>
  );
}
