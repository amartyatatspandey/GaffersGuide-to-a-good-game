"use client";
import React, { useState, useEffect, useMemo } from 'react';
import { Users, User, Play, Bot, AlertTriangle, Loader2, BarChart2, ShieldAlert, Zap, Clock, Film } from 'lucide-react';
import { getJobEvents } from '@/lib/api/jobs';
import { getApiBaseUrl, getAuthHeaders } from '@/lib/apiBase';

interface PlayerReportsProps {
  job: any;
  useAltNames: boolean;
  dictionary: Record<string, string>;
  onPlayClip: (startTimeS: number) => void;
  onAskAI: (prompt: string) => void;
}

const EVENT_NAME_LOOKUP: Record<string, string> = {
  "THR_001": "Dangerous Run",
  "THR_002": "Final-Third Entry",
  "THR_003": "Box Entry",
  "THR_004": "Transition Involvement",
  "THR_005": "Dangerous Reception",
  "THR_006": "Channel Exploitation",
  "THR_007": "Isolated Defender Exploit",
};

export function PlayerReports({
  job,
  useAltNames,
  dictionary,
  onPlayClip,
  onAskAI,
}: PlayerReportsProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<any>(null);
  const [expandedPlayer, setExpandedPlayer] = useState<string | null>(null);
  const [playerClips, setPlayerClips] = useState<Record<string, any[]>>({});
  const [loadingClips, setLoadingClips] = useState<Record<string, boolean>>({});

  useEffect(() => {
    if (!job?.jobId) {
      setLoading(false);
      return;
    }
    loadEventsData();
  }, [job]);

  async function loadEventsData() {
    try {
      setLoading(true);
      setError(null);
      const res = await getJobEvents(job.jobId);
      setData(res);
    } catch (err) {
      console.error(err);
      setError("Failed to load player reports. Ensure the analysis is complete.");
    } finally {
      setLoading(false);
    }
  }

  const getTeamLabel = (teamId: string) => {
    if (useAltNames && dictionary[teamId]) return dictionary[teamId];
    return teamId === "team_0" ? "Red Team" : "Blue Team";
  };

  const getPlayerLabel = (playerId: number) => {
    const key = `P${playerId}`;
    if (useAltNames && dictionary[key]) return dictionary[key];
    return `Player ${playerId}`;
  };

  const fetchClipsForPlayer = async (playerId: number) => {
    const key = `player-${playerId}`;
    if (playerClips[key] || loadingClips[key]) return;

    setLoadingClips(prev => ({ ...prev, [key]: true }));
    try {
      const base = getApiBaseUrl();
      const response = await fetch(`${base}/api/v1/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...getAuthHeaders()
        },
        body: JSON.stringify({
          message: `show the clips for player ${playerId}`,
          job_id: job.jobId,
          llm_engine: "local"
        }),
      });

      if (response.ok) {
        const chatData = await response.json();
        const clips = chatData.evidence?.clips || [];
        setPlayerClips(prev => ({ ...prev, [key]: clips }));
      }
    } catch (err) {
      console.error("Failed to load clips for player:", err);
    } finally {
      setLoadingClips(prev => ({ ...prev, [key]: false }));
    }
  };

  const handleExpandToggle = (playerKey: string, playerId: number) => {
    if (expandedPlayer === playerKey) {
      setExpandedPlayer(null);
    } else {
      setExpandedPlayer(playerKey);
      fetchClipsForPlayer(playerId);
    }
  };

  const threatColor = (score: number) => {
    if (score >= 70) return { bar: "bg-red-500", text: "text-red-400 font-bold", bg: "bg-red-500/10 border-red-500/20" };
    if (score >= 40) return { bar: "bg-amber-500", text: "text-amber-400 font-bold", bg: "bg-amber-500/10 border-amber-500/20" };
    return { bar: "bg-emerald-500", text: "text-emerald-400 font-medium", bg: "bg-emerald-500/10 border-emerald-500/20" };
  };

  const sortedProfiles = useMemo(() => {
    if (!data?.threat_profiles) return { team0: [], team1: [] };
    const team0 = data.threat_profiles.filter((p: any) => p.team_id === 'team_0');
    const team1 = data.threat_profiles.filter((p: any) => p.team_id === 'team_1');
    return {
      team0: [...team0].sort((a, b) => a.threat_rank - b.threat_rank),
      team1: [...team1].sort((a, b) => a.threat_rank - b.threat_rank),
    };
  }, [data]);

  if (!job) {
    return (
      <div className="h-full w-full bg-[#0a0f0a] flex flex-col items-center justify-center p-8 text-center">
        <Users className="text-gray-800 mb-4 animate-pulse" size={64} />
        <h2 className="text-lg font-bold text-gray-400 font-mono uppercase tracking-widest">No Active Match loaded</h2>
        <p className="text-sm text-gray-600 max-w-sm mt-2 font-mono">Select a completed analysis job or upload raw footage in the dashboard to generate player reports.</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="h-full w-full bg-[#0a0f0a] flex flex-col items-center justify-center gap-4">
        <Loader2 className="animate-spin text-emerald-500" size={48} />
        <p className="text-sm font-mono text-gray-500 uppercase tracking-widest animate-pulse">Compiling Tactical Threat Matrices...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full w-full bg-[#0a0f0a] flex flex-col items-center justify-center p-8 text-center">
        <AlertTriangle className="text-amber-500/70 mb-4" size={48} />
        <h2 className="text-sm font-bold text-amber-500 font-mono uppercase tracking-widest">Analysis telemetry Unavailable</h2>
        <p className="text-xs text-gray-500 max-w-md mt-2 font-sans">{error}</p>
        <button 
          onClick={loadEventsData}
          className="mt-6 bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/20 font-mono text-xs font-bold py-2 px-6 rounded-lg transition-all"
        >
          Retry Fetch
        </button>
      </div>
    );
  }

  return (
    <div className="h-full w-full bg-[#0a0f0a] flex flex-col animate-fade-in font-sans p-8 overflow-y-auto custom-scrollbar">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8 border-b border-gray-900 pb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-200 tracking-tight flex items-center gap-3">
            <Users className="text-emerald-500" />
            Individual Player Tactical Reports
          </h1>
          <p className="text-sm text-gray-500 mt-2 font-mono">
            Radar-tracked player threat metrics and automatic ontology event counts.
          </p>
        </div>
        <div className="flex items-center gap-6 bg-[#111a12]/30 border border-gray-900 px-6 py-3 rounded-xl font-mono text-xs">
          <div className="flex flex-col gap-0.5">
            <span className="text-gray-600 uppercase text-[9px] tracking-widest">Active Job</span>
            <span className="text-emerald-400 font-bold">{job?.file?.name || 'Historical Match'}</span>
          </div>
          <div className="h-8 w-px bg-gray-800" />
          <div className="flex flex-col gap-0.5">
            <span className="text-gray-600 uppercase text-[9px] tracking-widest">Ranks Sourced</span>
            <span className="text-gray-400 font-bold">{data?.event_stats?.players_with_events ?? 0} Players</span>
          </div>
        </div>
      </div>

      {/* Grid of Teams */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-8 items-start">
        {/* Team 0 Column (Red Team) */}
        <div className="bg-[#111a12]/10 border border-gray-900/80 rounded-2xl p-6 shadow-xl flex flex-col gap-4">
          <div className="flex justify-between items-center border-b border-gray-900/60 pb-3">
            <h2 className="text-sm font-bold text-red-400 font-mono uppercase tracking-widest flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-red-500" />
              {getTeamLabel('team_0')}
            </h2>
            <span className="text-[10px] font-mono text-gray-600 uppercase">Opposition Attacking Threat</span>
          </div>

          <div className="space-y-4">
            {sortedProfiles.team0.length === 0 ? (
              <div className="py-12 text-center text-xs text-gray-600 font-mono italic">No threat metrics logged for this team.</div>
            ) : (
              sortedProfiles.team0.map((player: any) => {
                const colors = threatColor(player.threat_score);
                const playerKey = `team0-${player.player_id}`;
                const isExpanded = expandedPlayer === playerKey;
                return (
                  <div 
                    key={player.player_id}
                    className={`border rounded-xl transition-all ${
                      isExpanded 
                        ? 'bg-[#152018]/40 border-emerald-500/35 shadow-lg shadow-emerald-950/5' 
                        : 'bg-black/20 border-gray-900/60 hover:bg-[#111a12]/10 hover:border-gray-800'
                    }`}
                  >
                    {/* Player Summary Header */}
                    <div 
                      onClick={() => handleExpandToggle(playerKey, player.player_id)}
                      className="p-5 flex flex-col md:flex-row md:items-center justify-between gap-4 cursor-pointer select-none"
                    >
                      <div className="flex items-center gap-3">
                        <div className="p-2 bg-gray-900 rounded-lg">
                          <User size={18} className="text-gray-400" />
                        </div>
                        <div>
                          <div className="font-bold text-gray-200 text-sm">{getPlayerLabel(player.player_id)}</div>
                          <div className="text-[10px] font-mono text-gray-600 uppercase tracking-tight mt-0.5">Threat Rank: #{player.threat_rank}</div>
                        </div>
                      </div>

                      <div className="flex flex-col md:items-end gap-1.5 w-full md:w-48">
                        <div className="flex justify-between w-full text-[10px] font-mono">
                          <span className="text-gray-600">Threat Score</span>
                          <span className={colors.text}>{player.threat_score.toFixed(1)}/100</span>
                        </div>
                        <div className="w-full h-1.5 bg-gray-950 rounded-full overflow-hidden">
                          <div className={`h-full ${colors.bar}`} style={{ width: `${player.threat_score}%` }} />
                        </div>
                      </div>
                    </div>

                    {/* Detailed Collapsible Area */}
                    {isExpanded && (
                      <div className="px-5 pb-5 pt-3 border-t border-gray-900/60 space-y-5 animate-fade-in text-xs font-sans">
                        {/* Summary breakdown text */}
                        {player.explanation && (
                          <div className="space-y-1.5">
                            <div className="text-[9px] font-bold font-mono text-gray-500 uppercase tracking-widest flex items-center gap-1.5">
                              <ShieldAlert size={12} className="text-emerald-500" />
                              Tactical Intelligence Verdict
                            </div>
                            <div className="bg-black/35 rounded-lg p-4 text-gray-400 font-sans leading-relaxed whitespace-pre-line border border-gray-900/40">
                              {player.explanation}
                            </div>
                          </div>
                        )}

                        {/* Frequency breakdown of ontology triggers */}
                        {player.event_counts && Object.keys(player.event_counts).length > 0 && (
                          <div className="space-y-2">
                            <div className="text-[9px] font-bold font-mono text-gray-500 uppercase tracking-widest flex items-center gap-1.5">
                              <BarChart2 size={12} className="text-emerald-500" />
                              Ontology Triggers Breakdown
                            </div>
                            <div className="flex flex-wrap gap-2">
                              {Object.entries(player.event_counts).map(([code, count]) => (
                                <div key={code} className="px-3 py-1.5 bg-black/40 border border-gray-900 hover:border-emerald-500/20 rounded-lg flex items-center gap-2">
                                  <span className="font-mono text-[10px] text-gray-400">{EVENT_NAME_LOOKUP[code] || code}</span>
                                  <span className="font-mono font-bold text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded text-[10px]">{count as number}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Supporting clips queried from chat */}
                        <div className="space-y-2 pt-2 border-t border-gray-950">
                          <div className="text-[9px] font-bold font-mono text-gray-500 uppercase tracking-widest flex items-center gap-1.5">
                            <Film size={12} className="text-emerald-500" />
                            Visual Evidence (AI Retrieved Clips)
                          </div>
                          {loadingClips[playerKey] ? (
                            <div className="flex items-center gap-2 text-gray-500 italic py-2">
                              <Loader2 size={12} className="animate-spin text-emerald-500" />
                              <span>Searching event catalog...</span>
                            </div>
                          ) : !playerClips[playerKey] || playerClips[playerKey].length === 0 ? (
                            <div className="text-gray-600 italic font-mono py-1">No supported moments registered in index.</div>
                          ) : (
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                              {playerClips[playerKey].map((clip: any, cIdx: number) => (
                                <button
                                  key={cIdx}
                                  onClick={() => onPlayClip(clip.start_time_s)}
                                  className="w-full text-left p-3 rounded-lg bg-black/40 border border-gray-900 hover:border-emerald-500/30 hover:bg-[#111a12]/20 transition-all flex items-center justify-between group"
                                >
                                  <div className="flex-1 min-w-0 pr-2">
                                    <div className="text-[11px] font-bold text-gray-200 truncate group-hover:text-emerald-400 transition-colors">
                                      {clip.label || "Key Moment"}
                                    </div>
                                    <div className="flex items-center gap-2 mt-1.5 text-[9px] font-mono text-gray-600">
                                      <span className="flex items-center gap-1"><Clock size={10} /> {Math.floor(clip.start_time_s / 60)}:{(Math.floor(clip.start_time_s % 60)).toString().padStart(2, '0')}</span>
                                      <span className="bg-emerald-500/5 text-emerald-500/50 px-1 py-0.25 rounded">{Math.round(clip.confidence_pct || clip.relevance_score * 100)}% Match</span>
                                    </div>
                                  </div>
                                  <div className="h-6 w-6 rounded-full bg-emerald-500/10 flex items-center justify-center text-emerald-500 group-hover:bg-emerald-500/20 group-hover:scale-105 transition-all shrink-0">
                                    <Play size={8} fill="currentColor" />
                                  </div>
                                </button>
                              ))}
                            </div>
                          )}
                        </div>

                        {/* Ask AI Trigger */}
                        <div className="flex justify-end pt-2 border-t border-gray-950">
                          <button
                            onClick={() => onAskAI(`Explain how we should defend against ${getPlayerLabel(player.player_id)} (P${player.player_id}) given their threat rank of #${player.threat_rank} and tactical profile.`)}
                            className="bg-emerald-500 text-black hover:bg-emerald-400 font-bold px-4 py-2 rounded-lg text-xs transition-all flex items-center gap-1.5"
                          >
                            <Bot size={14} /> Ask AI about this Player
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Team 1 Column (Blue Team) */}
        <div className="bg-[#111a12]/10 border border-gray-900/80 rounded-2xl p-6 shadow-xl flex flex-col gap-4">
          <div className="flex justify-between items-center border-b border-gray-900/60 pb-3">
            <h2 className="text-sm font-bold text-blue-400 font-mono uppercase tracking-widest flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-blue-500" />
              {getTeamLabel('team_1')}
            </h2>
            <span className="text-[10px] font-mono text-gray-600 uppercase">Opposition Attacking Threat</span>
          </div>

          <div className="space-y-4">
            {sortedProfiles.team1.length === 0 ? (
              <div className="py-12 text-center text-xs text-gray-600 font-mono italic">No threat metrics logged for this team.</div>
            ) : (
              sortedProfiles.team1.map((player: any) => {
                const colors = threatColor(player.threat_score);
                const playerKey = `team1-${player.player_id}`;
                const isExpanded = expandedPlayer === playerKey;
                return (
                  <div 
                    key={player.player_id}
                    className={`border rounded-xl transition-all ${
                      isExpanded 
                        ? 'bg-[#152018]/40 border-emerald-500/35 shadow-lg shadow-emerald-950/5' 
                        : 'bg-black/20 border-gray-900/60 hover:bg-[#111a12]/10 hover:border-gray-800'
                    }`}
                  >
                    {/* Player Summary Header */}
                    <div 
                      onClick={() => handleExpandToggle(playerKey, player.player_id)}
                      className="p-5 flex flex-col md:flex-row md:items-center justify-between gap-4 cursor-pointer select-none"
                    >
                      <div className="flex items-center gap-3">
                        <div className="p-2 bg-gray-900 rounded-lg">
                          <User size={18} className="text-gray-400" />
                        </div>
                        <div>
                          <div className="font-bold text-gray-200 text-sm">{getPlayerLabel(player.player_id)}</div>
                          <div className="text-[10px] font-mono text-gray-600 uppercase tracking-tight mt-0.5">Threat Rank: #{player.threat_rank}</div>
                        </div>
                      </div>

                      <div className="flex flex-col md:items-end gap-1.5 w-full md:w-48">
                        <div className="flex justify-between w-full text-[10px] font-mono">
                          <span className="text-gray-600">Threat Score</span>
                          <span className={colors.text}>{player.threat_score.toFixed(1)}/100</span>
                        </div>
                        <div className="w-full h-1.5 bg-gray-950 rounded-full overflow-hidden">
                          <div className={`h-full ${colors.bar}`} style={{ width: `${player.threat_score}%` }} />
                        </div>
                      </div>
                    </div>

                    {/* Detailed Collapsible Area */}
                    {isExpanded && (
                      <div className="px-5 pb-5 pt-3 border-t border-gray-900/60 space-y-5 animate-fade-in text-xs font-sans">
                        {/* Summary breakdown text */}
                        {player.explanation && (
                          <div className="space-y-1.5">
                            <div className="text-[9px] font-bold font-mono text-gray-500 uppercase tracking-widest flex items-center gap-1.5">
                              <ShieldAlert size={12} className="text-emerald-500" />
                              Tactical Intelligence Verdict
                            </div>
                            <div className="bg-black/35 rounded-lg p-4 text-gray-400 font-sans leading-relaxed whitespace-pre-line border border-gray-900/40">
                              {player.explanation}
                            </div>
                          </div>
                        )}

                        {/* Frequency breakdown of ontology triggers */}
                        {player.event_counts && Object.keys(player.event_counts).length > 0 && (
                          <div className="space-y-2">
                            <div className="text-[9px] font-bold font-mono text-gray-500 uppercase tracking-widest flex items-center gap-1.5">
                              <BarChart2 size={12} className="text-emerald-500" />
                              Ontology Triggers Breakdown
                            </div>
                            <div className="flex flex-wrap gap-2">
                              {Object.entries(player.event_counts).map(([code, count]) => (
                                <div key={code} className="px-3 py-1.5 bg-black/40 border border-gray-900 hover:border-emerald-500/20 rounded-lg flex items-center gap-2">
                                  <span className="font-mono text-[10px] text-gray-400">{EVENT_NAME_LOOKUP[code] || code}</span>
                                  <span className="font-mono font-bold text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded text-[10px]">{count as number}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Supporting clips queried from chat */}
                        <div className="space-y-2 pt-2 border-t border-gray-950">
                          <div className="text-[9px] font-bold font-mono text-gray-500 uppercase tracking-widest flex items-center gap-1.5">
                            <Film size={12} className="text-emerald-500" />
                            Visual Evidence (AI Retrieved Clips)
                          </div>
                          {loadingClips[playerKey] ? (
                            <div className="flex items-center gap-2 text-gray-500 italic py-2">
                              <Loader2 size={12} className="animate-spin text-emerald-500" />
                              <span>Searching event catalog...</span>
                            </div>
                          ) : !playerClips[playerKey] || playerClips[playerKey].length === 0 ? (
                            <div className="text-gray-600 italic font-mono py-1">No supported moments registered in index.</div>
                          ) : (
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                              {playerClips[playerKey].map((clip: any, cIdx: number) => (
                                <button
                                  key={cIdx}
                                  onClick={() => onPlayClip(clip.start_time_s)}
                                  className="w-full text-left p-3 rounded-lg bg-black/40 border border-gray-900 hover:border-emerald-500/30 hover:bg-[#111a12]/20 transition-all flex items-center justify-between group"
                                >
                                  <div className="flex-1 min-w-0 pr-2">
                                    <div className="text-[11px] font-bold text-gray-200 truncate group-hover:text-emerald-400 transition-colors">
                                      {clip.label || "Key Moment"}
                                    </div>
                                    <div className="flex items-center gap-2 mt-1.5 text-[9px] font-mono text-gray-600">
                                      <span className="flex items-center gap-1"><Clock size={10} /> {Math.floor(clip.start_time_s / 60)}:{(Math.floor(clip.start_time_s % 60)).toString().padStart(2, '0')}</span>
                                      <span className="bg-emerald-500/5 text-emerald-500/50 px-1 py-0.25 rounded">{Math.round(clip.confidence_pct || clip.relevance_score * 100)}% Match</span>
                                    </div>
                                  </div>
                                  <div className="h-6 w-6 rounded-full bg-emerald-500/10 flex items-center justify-center text-emerald-500 group-hover:bg-emerald-500/20 group-hover:scale-105 transition-all shrink-0">
                                    <Play size={8} fill="currentColor" />
                                  </div>
                                </button>
                              ))}
                            </div>
                          )}
                        </div>

                        {/* Ask AI Trigger */}
                        <div className="flex justify-end pt-2 border-t border-gray-950">
                          <button
                            onClick={() => onAskAI(`Explain how we should defend against ${getPlayerLabel(player.player_id)} (P${player.player_id}) given their threat rank of #${player.threat_rank} and tactical profile.`)}
                            className="bg-emerald-500 text-black hover:bg-emerald-400 font-bold px-4 py-2 rounded-lg text-xs transition-all flex items-center gap-1.5"
                          >
                            <Bot size={14} /> Ask AI about this Player
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
