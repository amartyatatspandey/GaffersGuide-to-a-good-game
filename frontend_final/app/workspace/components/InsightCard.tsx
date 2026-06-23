"use client";
import React, { useState } from 'react';
import { useStreamingText } from '@/hooks/useStreamingText';
import { Zap, TrendingUp, Shield, Activity, Film, Play, User, Clock } from 'lucide-react';
import { resolvePlayerLabel, resolveTeamLabel, type JobPlayerMappings } from '@/lib/playerMappingUtils';

const EVENT_NAME_LOOKUP: Record<string, string> = {
  // Movement
  "MOV_001": "High-Speed Run",
  "MOV_002": "Sprint",
  "MOV_003": "Recovery Run",
  "MOV_004": "Overlap Run",
  "MOV_005": "Underlap Run",
  "MOV_006": "Third-Man Run",
  "MOV_007": "Diagonal Run",
  // Positional
  "POS_001": "Wide Positioning",
  "POS_002": "Half-Space Occupation",
  "POS_003": "Between-Lines",
  "POS_004": "Deep Positioning",
  "POS_005": "Advanced Positioning",
  "POS_006": "Pressing Trap Position",
  // Threat
  "THR_001": "Dangerous Run",
  "THR_002": "Final-Third Entry",
  "THR_003": "Box Entry",
  "THR_004": "Transition Involvement",
  "THR_005": "Dangerous Reception",
  "THR_006": "Channel Exploitation",
  "THR_007": "Isolated Defender Exploit",
  // Shape
  "SHP_001": "High Press Moment",
  "SHP_002": "Mid Block",
  "SHP_003": "Low Block",
  "SHP_004": "Compact Shape",
  "SHP_005": "Stretched Shape",
  "SHP_006": "Overload Zone",
  "SHP_007": "Pressing Trap Triggered",
  "SHP_008": "Counter-Attack Launch",
  // Transition
  "TRN_001": "Defensive Transition",
  "TRN_002": "Offensive Transition",
  "TRN_003": "Press Success",
  "TRN_004": "Press Failure",
  "TRN_005": "Counter-Attack Sequence",
};

interface TeamInsight {
  payload: string;
  keywords: any[];
}

interface InsightCardProps {
  title: string;
  minute: string;
  blueTeam: TeamInsight;
  redTeam: TeamInsight;
  metrics?: any;
  evidenceClips?: any[];
  threatContext?: any;
  eventCountSummary?: Record<string, number> | null;
  onPlayClip?: (startTimeS: number) => void;
  useAltNames?: boolean;
  dictionary?: Record<string, string>;
  savedMappings?: JobPlayerMappings | null;
}

export function InsightCard({
  title,
  minute,
  blueTeam,
  redTeam,
  metrics,
  evidenceClips = [],
  threatContext,
  eventCountSummary,
  onPlayClip,
  useAltNames = false,
  dictionary = {},
  savedMappings = null,
}: InsightCardProps) {
  const [activeTeam, setActiveTeam] = useState<'blue' | 'red'>('blue');
  const currentData = activeTeam === 'blue' ? blueTeam : redTeam;
  const { displayedText } = useStreamingText(currentData.payload, 20);

  // blue = team_1, red = team_0  (matches backend naming)
  const teamMetrics = activeTeam === 'blue' ? metrics?.team_1 : metrics?.team_0;

  const compactnessLabel = teamMetrics?.compactness != null
    ? (teamMetrics.compactness > 70 ? "High" : teamMetrics.compactness > 40 ? "Med" : "Low")
    : "—";
  const transitionLabel = teamMetrics?.transition_speed != null
    ? (teamMetrics.transition_speed > 70 ? "Fast" : teamMetrics.transition_speed > 40 ? "Norm" : "Slow")
    : "—";
  const tacticalPower = teamMetrics?.tactical_power?.toFixed(1)
    ?? (activeTeam === 'blue' ? metrics?.team_blue_score?.toFixed(1) : metrics?.team_red_score?.toFixed(1))
    ?? "—";
  const winProb = metrics?.win_probability
    ? (activeTeam === 'blue' ? metrics.win_probability.team_blue : metrics.win_probability.team_red)
    : teamMetrics?.win_prob
    ?? "—";

  const hasTelemetry = (evidenceClips && evidenceClips.length > 0) || 
                       (threatContext?.top_threats && threatContext.top_threats.length > 0) ||
                       (eventCountSummary && Object.keys(eventCountSummary).length > 0);

  return (
    <div className="w-full flex flex-col gap-6 font-sans">
      
      {/* Metrics Bar */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-black/30 border border-gray-800 p-3 rounded-lg flex flex-col gap-1">
          <span className="text-[9px] font-mono uppercase text-gray-500 tracking-widest">Tactical Power</span>
          <div className="flex items-center justify-between">
            <span className="text-sm font-bold text-emerald-400">
              {tacticalPower}
            </span>
            <Activity size={12} className="text-emerald-500/50" />
          </div>
        </div>
        <div className="bg-black/30 border border-gray-800 p-3 rounded-lg flex flex-col gap-1">
          <span className="text-[9px] font-mono uppercase text-gray-500 tracking-widest">Win Prob</span>
          <div className="flex items-center justify-between">
            <span className="text-sm font-bold text-cyan-400">
              {winProb}{winProb !== "—" ? "%" : ""}
            </span>
            <TrendingUp size={12} className="text-cyan-500/50" />
          </div>
        </div>
        <div className="bg-black/30 border border-gray-800 p-3 rounded-lg flex flex-col gap-1">
          <span className="text-[9px] font-mono uppercase text-gray-500 tracking-widest">Compactness</span>
          <div className="flex items-center justify-between">
            <span className="text-sm font-bold text-amber-400">
              {compactnessLabel}
            </span>
            <Shield size={12} className="text-amber-500/50" />
          </div>
        </div>
        <div className="bg-black/30 border border-gray-800 p-3 rounded-lg flex flex-col gap-1">
          <span className="text-[9px] font-mono uppercase text-gray-500 tracking-widest">Trans. Speed</span>
          <div className="flex items-center justify-between">
            <span className="text-sm font-bold text-white">
              {transitionLabel}
            </span>
            <Zap size={12} className="text-emerald-500/50" />
          </div>
        </div>
      </div>

      {/* Main Grid: Split info on left, telemetry on right if available */}
      <div className={`grid grid-cols-1 ${hasTelemetry ? 'lg:grid-cols-[1.2fr_0.8fr]' : ''} gap-8`}>
        
        {/* Left Side: Coach Instructions */}
        <div className="flex flex-col gap-4">
          <div className="flex items-center justify-between border-b border-gray-800 pb-2">
            <h3 className="text-lg font-bold text-gray-100 flex items-center gap-2">
              <span className="text-red-500 font-mono">[{minute}&apos;]</span>
              {title}
            </h3>
            <div className="flex bg-gray-900 rounded-md p-0.5 border border-gray-800">
              <button 
                onClick={() => setActiveTeam('blue')}
                className={`px-3 py-1 text-[10px] font-bold uppercase rounded ${activeTeam === 'blue' ? 'bg-blue-600/30 text-blue-400' : 'text-gray-500'}`}
              >
                {useAltNames && dictionary["team_1"] ? dictionary["team_1"] : (savedMappings?.team_b_name ?? "Blue")}
              </button>
              <button 
                onClick={() => setActiveTeam('red')}
                className={`px-3 py-1 text-[10px] font-bold uppercase rounded ${activeTeam === 'red' ? 'bg-red-600/30 text-red-400' : 'text-gray-500'}`}
              >
                {useAltNames && dictionary["team_0"] ? dictionary["team_0"] : (savedMappings?.team_a_name ?? "Red")}
              </button>
            </div>
          </div>

          <div className="bg-[#111a12]/30 border border-emerald-500/10 rounded-xl p-5 relative overflow-hidden flex-1">
             <div className="absolute top-0 left-0 w-1 h-full bg-emerald-500/40" />
             <p className="text-gray-300 leading-relaxed text-sm font-medium">
               {displayedText}
               <span className="inline-block w-1.5 h-4 bg-emerald-500 ml-1 animate-pulse" />
             </p>
          </div>
        </div>

        {/* Right Side: Telemetry / Visual Evidence */}
        {hasTelemetry && (
          <div className="flex flex-col gap-4 bg-gray-950/40 border border-gray-900 rounded-2xl p-5 backdrop-blur-sm">
            
            {/* 1. Evidence Clips */}
            {evidenceClips && evidenceClips.length > 0 && (
              <div className="space-y-2.5">
                <h4 className="text-[10px] font-bold font-mono text-gray-400 uppercase tracking-widest flex items-center gap-2">
                  <Film size={12} className="text-emerald-500" />
                  Visual Evidence Clips
                </h4>
                <div className="space-y-2">
                  {evidenceClips.map((clip, idx) => (
                    <button
                      key={idx}
                      onClick={() => onPlayClip && onPlayClip(clip.start_time_s)}
                      className="w-full text-left p-3 bg-black/40 hover:bg-[#111a12]/15 border border-gray-900 hover:border-emerald-500/30 rounded-xl transition-all flex items-center justify-between group"
                    >
                      <div className="flex-1 min-w-0 pr-2">
                        <div className="text-xs font-semibold text-gray-300 truncate group-hover:text-emerald-400 transition-colors">
                          {clip.label || "Evidence Moment"}
                        </div>
                        <div className="flex items-center gap-2 mt-1">
                          <span className="text-[9px] font-mono text-gray-600 flex items-center gap-1">
                            <Clock size={8} />
                            {Math.floor(clip.start_time_s / 60)}:{(Math.floor(clip.start_time_s % 60)).toString().padStart(2, '0')}
                          </span>
                          <span className="text-[9px] font-mono text-emerald-500/50 bg-emerald-500/5 px-1.5 py-0.5 rounded">
                            {Math.round(clip.confidence_pct || clip.relevance_score * 100)}% Match
                          </span>
                        </div>
                      </div>
                      <div className="h-6 w-6 rounded-full bg-emerald-500/10 flex items-center justify-center text-emerald-500 group-hover:bg-emerald-500/20 group-hover:scale-105 transition-all">
                        <Play size={10} fill="currentColor" />
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* 2. Top Threats */}
            {threatContext?.top_threats && threatContext.top_threats.length > 0 && (
              <div className="space-y-2.5 pt-2 border-t border-gray-900">
                <h4 className="text-[10px] font-bold font-mono text-gray-400 uppercase tracking-widest flex items-center gap-2">
                  <User size={12} className="text-emerald-500" />
                  Key Opponent Threats
                </h4>
                <div className="space-y-2">
                  {threatContext.top_threats.map((threat: any, idx: number) => (
                    <div key={idx} className="p-3 bg-black/20 border border-gray-900/80 rounded-xl space-y-1.5">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div className={`h-5 w-5 rounded-full flex items-center justify-center text-[10px] font-bold ${threat.team_id === 'team_0' ? 'bg-red-500/20 text-red-400' : 'bg-blue-500/20 text-blue-400'}`}>
                            {savedMappings?.mappings?.[String(threat.player_id)]
                              ? `#${savedMappings.mappings[String(threat.player_id)].number}`
                              : `P${threat.player_id}`}
                          </div>
                          <span className="text-[10px] font-bold text-gray-400 uppercase tracking-tight">
                            {resolvePlayerLabel(threat.player_id, { savedMappings, useAltNames, dictionary })}
                            {' '}({resolveTeamLabel(threat.team_id, { savedMappings, useAltNames, dictionary, short: true })})
                          </span>
                        </div>
                        <span className="text-[11px] font-mono font-bold text-emerald-400">
                          Threat: {threat.threat_score.toFixed(1)}
                        </span>
                      </div>
                      <div className="h-1 bg-gray-900 rounded-full overflow-hidden">
                        <div 
                          className={`h-full ${threat.team_id === 'team_0' ? 'bg-red-500' : 'bg-blue-500'}`} 
                          style={{ width: `${threat.threat_score}%` }} 
                        />
                      </div>
                      {threat.explanation && (
                        <p className="text-[9px] text-gray-500 leading-normal font-sans pt-0.5">
                          {threat.explanation}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* 3. Event Frequencies */}
            {eventCountSummary && Object.keys(eventCountSummary).length > 0 && (
              <div className="space-y-2.5 pt-2 border-t border-gray-900">
                <h4 className="text-[10px] font-bold font-mono text-gray-400 uppercase tracking-widest flex items-center gap-2">
                  <Activity size={12} className="text-emerald-500" />
                  Ontology Triggers
                </h4>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(eventCountSummary).map(([code, count]) => (
                    <div 
                      key={code}
                      className="px-2.5 py-1 bg-black/40 border border-gray-900 hover:border-emerald-500/20 rounded-lg flex items-center gap-2 transition-colors cursor-help"
                      title={`${code}: ${EVENT_NAME_LOOKUP[code] || 'Unknown Event'}`}
                    >
                      <span className="text-[9px] font-mono font-semibold text-gray-400">
                        {(() => {
                          const orig = EVENT_NAME_LOOKUP[code] || code;
                          return useAltNames && dictionary[orig] ? dictionary[orig] : orig;
                        })()}
                      </span>
                      <span className="text-[10px] font-mono font-bold text-emerald-400 bg-emerald-500/10 px-1 py-0.25 rounded">
                        {count}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

          </div>
        )}
      </div>

      <div className="mt-auto pt-4 border-t border-gray-900 flex justify-between items-center">
         <span className="text-[9px] font-mono text-gray-600 uppercase tracking-widest italic">Tactical Engine Analysis active · v1.2</span>
         <span className="text-[9px] font-mono text-emerald-500/50 uppercase tracking-widest">Gaffer&apos;s Guide Standard</span>
      </div>
    </div>
  );
}
