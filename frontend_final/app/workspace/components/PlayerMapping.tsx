"use client";
import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
  Users, User, Shield, CheckCircle, RefreshCw,
  Save, AlertTriangle, ArrowRightLeft, Database,
  Zap, Activity, MapPin, Clock, TrendingUp, Sparkles,
  ChevronDown, ChevronUp, Check, X
} from 'lucide-react';

// ── Types ──────────────────────────────────────────────────────────────────

interface MatchSetupPlayer {
  id: string;
  name: string;
  number: number;
  position: 'GK' | 'DF' | 'MF' | 'FW';
  isStarting: boolean;
}

interface TeamSetup {
  name: string;
  formation: string;
  players: MatchSetupPlayer[];
}

interface MatchSetupData {
  match_id: string;
  fixture_name: string;
  competition: string;
  video_filename: string;
  video_size: string;
  video_duration: string;
  team_a: TeamSetup;
  team_b: TeamSetup;
  created_at: string;
}

interface LineupPlayer {
  id: string;
  name: string;
  number: number;
  position: string;
  team: 'A' | 'B';
  teamName: string;
  isStarting: boolean;
}

interface TrackedPlayerStats {
  id: number;
  teamId: string | null;
  activeFrames: number;
  distanceM: number;
  avgSpeedKmh: number;
  maxSpeedKmh: number;
  avgX: number | null;
  avgY: number | null;
  minutesTracked: number;
}

interface SuggestedMapping {
  lineupPlayerId: string;
  playerName: string;
  playerNumber: number;
  position: string;
  teamName: string;
  confidence: number; // 0–100
  reason: string;
}

interface PlayerMappingProps {
  job: {
    jobId: string;
    tracking?: {
      frames: any[];
    } | null;
  } | null;
}

// ── Constants ──────────────────────────────────────────────────────────────
const FPS = 25.0;
const MAX_DISPLACEMENT_PER_FRAME_M = 4.2;
const MAX_SPEED_KMH_CAP = 50.0;
const MAX_REALISTIC_SPRINT_KMH = 38.0;

const POSITION_CLUSTER_LABELS: Record<string, string> = {
  GK: 'Goalkeeper',
  DF: 'Defender',
  MF: 'Midfielder',
  FW: 'Forward',
};

const POSITION_AREA_THRESHOLDS = {
  GK: { xMin: 0, xMax: 20, yMin: 0, yMax: 100 },
  DF: { xMin: 10, xMax: 42, yMin: 0, yMax: 100 },
  MF: { xMin: 30, xMax: 70, yMin: 0, yMax: 100 },
  FW: { xMin: 58, xMax: 100, yMin: 0, yMax: 100 },
};

// ── Pitch heatmap mini-canvas ──────────────────────────────────────────────
function MiniHeatmap({ positions, teamId }: { positions: [number, number][]; teamId: string | null }) {
  const color = teamId === 'team_0' ? '#3b82f6' : '#ef4444';

  const w = 70, h = 45;
  const circles = positions.slice(0, 200);

  return (
    <svg width={w} height={h} className="rounded overflow-hidden" style={{ background: '#061a0d' }}>
      {/* Pitch outline */}
      <rect x={1} y={1} width={w - 2} height={h - 2} fill="none" stroke="#1a3321" strokeWidth={0.8} rx={2} />
      {/* Centre line */}
      <line x1={w / 2} y1={1} x2={w / 2} y2={h - 1} stroke="#1a3321" strokeWidth={0.5} />
      {/* Centre circle */}
      <circle cx={w / 2} cy={h / 2} r={7} fill="none" stroke="#1a3321" strokeWidth={0.5} />
      {/* Position dots */}
      {circles.map(([x, y], i) => (
        <circle
          key={i}
          cx={(x / 105) * w}
          cy={(y / 68) * h}
          r={1.8}
          fill={color}
          opacity={0.35}
        />
      ))}
    </svg>
  );
}

// ── Position inference from average X ─────────────────────────────────────
function inferPosition(avgX: number | null, teamId: string | null): 'GK' | 'DF' | 'MF' | 'FW' {
  if (avgX === null) return 'MF';

  // Normalise to 0–100 scale. If team_1 (right-to-left), mirror
  const normX = teamId === 'team_1' ? 105 - avgX : avgX;
  const pct = (normX / 105) * 100;

  if (pct < 18) return 'GK';
  if (pct < 42) return 'DF';
  if (pct < 65) return 'MF';
  return 'FW';
}

// ── Auto-suggest mappings ─────────────────────────────────────────────────
function computeSuggestions(
  stats: TrackedPlayerStats,
  lineupPlayers: LineupPlayer[],
  usedLineupIds: Set<string>
): SuggestedMapping | null {
  if (lineupPlayers.length === 0) return null;

  const inferredPos = inferPosition(stats.avgX, stats.teamId);
  // Prefer same team side
  const sideFilter = stats.teamId === 'team_0' ? 'A' : 'B';

  // Score candidates
  const scored = lineupPlayers
    .filter(p => !usedLineupIds.has(p.id))
    .map(p => {
      let score = 0;

      // Same team side: strong signal
      if (p.team === sideFilter) score += 40;

      // Position match
      if (p.position === inferredPos) score += 35;
      else if (
        (p.position === 'DF' && inferredPos === 'GK') ||
        (p.position === 'GK' && inferredPos === 'DF')
      ) score += 10;
      else if (
        (p.position === 'MF' && inferredPos === 'FW') ||
        (p.position === 'FW' && inferredPos === 'MF')
      ) score += 15;

      // Starter bonus
      if (p.isStarting) score += 10;

      // Activity bonus — higher minutes = more confident
      score += Math.min(10, stats.minutesTracked / 3);

      return { p, score };
    })
    .sort((a, b) => b.score - a.score);

  if (scored.length === 0) return null;

  const best = scored[0];
  // Cap confidence to sensible max (90 when perfect match)
  const confidence = Math.round(Math.min(92, best.score));

  return {
    lineupPlayerId: best.p.id,
    playerName: best.p.name,
    playerNumber: best.p.number,
    position: best.p.position,
    teamName: best.p.teamName,
    confidence,
    reason: `${POSITION_CLUSTER_LABELS[inferredPos]} cluster · ${best.p.team === sideFilter ? 'Same team side' : 'Cross-team'} · ${stats.minutesTracked.toFixed(0)} min tracked`,
  };
}

// ── Main Component ─────────────────────────────────────────────────────────
export function PlayerMapping({ job }: PlayerMappingProps) {
  const [matchSetups, setMatchSetups] = useState<MatchSetupData[]>([]);
  const [selectedSetupId, setSelectedSetupId] = useState<string>('');
  const [mappings, setMappings] = useState<Record<string, string>>({});
  const [saveSuccess, setSaveSuccess] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<number | null>(null);

  // ── Load Match Setups & Mappings ─────────────────────────────────────
  useEffect(() => {
    if (typeof window === 'undefined') return;

    const setupsStr = localStorage.getItem('gaffer-match-setups');
    if (setupsStr) {
      try {
        const setups: MatchSetupData[] = JSON.parse(setupsStr);
        setMatchSetups(setups);
        if (setups.length > 0) setSelectedSetupId(setups[0].match_id);
      } catch (e) { console.error('Failed to parse setups', e); }
    }

    if (job?.jobId) {
      const mappingsStr = localStorage.getItem('gaffer-player-mappings');
      if (mappingsStr) {
        try {
          const allMappings = JSON.parse(mappingsStr);
          const jobMapping = allMappings[job.jobId];
          if (jobMapping?.mappings) {
            const rawMappings: Record<string, string> = {};
            Object.entries(jobMapping.mappings).forEach(([pid, player]: [string, any]) => {
              rawMappings[pid] = player.id;
            });
            setMappings(rawMappings);
            if (jobMapping.setup_id) setSelectedSetupId(jobMapping.setup_id);
          }
        } catch (e) { console.error('Failed to parse mappings', e); }
      }
    }
  }, [job?.jobId]);

  // ── Compute rich tracked player stats ──────────────────────────────────
  const trackedPlayers = useMemo((): TrackedPlayerStats[] => {
    if (!job?.tracking?.frames) return [];

    type PData = {
      id: number;
      teamId: string | null;
      activeFrames: number;
      distanceCovered: number;
      speeds: number[];
      lastFrameIdx: number | null;
      lastX: number | null;
      lastY: number | null;
      xSum: number;
      ySum: number;
      coordCount: number;
      positions: [number, number][];
    };

    const playersMap: Record<number, PData> = {};

    for (const frame of job.tracking.frames) {
      const fIdx = frame.frame_idx ?? 0;
      for (const p of (frame.players || [])) {
        const pid = p.id;
        if (pid === undefined || pid === null) continue;

        if (!playersMap[pid]) {
          playersMap[pid] = {
            id: pid,
            teamId: p.team_id || null,
            activeFrames: 0,
            distanceCovered: 0,
            speeds: [],
            lastFrameIdx: null,
            lastX: null,
            lastY: null,
            xSum: 0,
            ySum: 0,
            coordCount: 0,
            positions: [],
          };
        }

        const pd = playersMap[pid];
        if (!pd.teamId && p.team_id) pd.teamId = p.team_id;

        if (p.x_pitch != null && p.y_pitch != null) {
          pd.activeFrames += 1;
          pd.xSum += p.x_pitch;
          pd.ySum += p.y_pitch;
          pd.coordCount += 1;

          // Subsample positions for heatmap (every 15 frames)
          if (pd.activeFrames % 15 === 0) {
            pd.positions.push([p.x_pitch, p.y_pitch]);
          }

          if (p.speed_kmh != null) {
            const clamped = Math.min(p.speed_kmh, MAX_SPEED_KMH_CAP);
            if (clamped > 0) pd.speeds.push(clamped);
          }

          if (pd.lastX !== null && pd.lastY !== null && pd.lastFrameIdx !== null) {
            const gap = fIdx - pd.lastFrameIdx;
            if (gap > 0 && gap < FPS * 2) {
              const dist = Math.hypot(p.x_pitch - pd.lastX, p.y_pitch - pd.lastY);
              if (dist <= MAX_DISPLACEMENT_PER_FRAME_M * gap) {
                pd.distanceCovered += dist;
              }
            }
          }
          pd.lastFrameIdx = fIdx;
          pd.lastX = p.x_pitch;
          pd.lastY = p.y_pitch;
        }
      }
    }

    const allPids = Object.keys(playersMap).map(Number);
    const t0 = allPids.filter(pid => playersMap[pid].teamId === 'team_0');
    const t1 = allPids.filter(pid => playersMap[pid].teamId === 'team_1');
    t0.sort((a, b) => playersMap[b].activeFrames - playersMap[a].activeFrames);
    t1.sort((a, b) => playersMap[b].activeFrames - playersMap[a].activeFrames);

    return [...t0.slice(0, 13), ...t1.slice(0, 13)].map(pid => {
      const pd = playersMap[pid];
      const validSpeeds = pd.speeds.filter(s => s > 0 && s <= MAX_REALISTIC_SPRINT_KMH);
      const avgSpeedKmh = validSpeeds.length > 0
        ? validSpeeds.reduce((a, b) => a + b, 0) / validSpeeds.length
        : 0;
      const maxSpeedKmh = validSpeeds.length > 0 ? Math.max(...validSpeeds) : 0;
      const minutesTracked = (pd.activeFrames / FPS) / 60;

      return {
        id: pid,
        teamId: pd.teamId,
        activeFrames: pd.activeFrames,
        distanceM: pd.distanceCovered,
        avgSpeedKmh,
        maxSpeedKmh,
        avgX: pd.coordCount > 0 ? pd.xSum / pd.coordCount : null,
        avgY: pd.coordCount > 0 ? pd.ySum / pd.coordCount : null,
        minutesTracked,
        positions: pd.positions,
      } as TrackedPlayerStats & { positions: [number, number][] };
    });
  }, [job?.tracking?.frames]);

  // ── Lineup players ──────────────────────────────────────────────────────
  const currentSetup = useMemo(() =>
    matchSetups.find(s => s.match_id === selectedSetupId) || null,
    [matchSetups, selectedSetupId]
  );

  const lineupPlayers = useMemo((): LineupPlayer[] => {
    if (!currentSetup) return [];
    const list: LineupPlayer[] = [];
    currentSetup.team_a.players.forEach(p => list.push({ ...p, team: 'A', teamName: currentSetup.team_a.name }));
    currentSetup.team_b.players.forEach(p => list.push({ ...p, team: 'B', teamName: currentSetup.team_b.name }));
    return list;
  }, [currentSetup]);

  // ── Auto-suggest (per player, excluding already-used IDs) ─────────────
  const suggestions = useMemo(() => {
    const usedIds = new Set(Object.values(mappings).filter(Boolean));
    const result: Record<number, SuggestedMapping | null> = {};
    for (const tp of trackedPlayers) {
      const usedForThis = new Set([...usedIds]);
      if (mappings[String(tp.id)]) usedForThis.delete(mappings[String(tp.id)]);
      result[tp.id] = computeSuggestions(tp as any, lineupPlayers, usedForThis);
    }
    return result;
  }, [trackedPlayers, lineupPlayers, mappings]);

  // ── Handlers ───────────────────────────────────────────────────────────
  const handleMapChange = useCallback((trackedId: number, lineupPlayerId: string) => {
    setMappings(prev => ({ ...prev, [String(trackedId)]: lineupPlayerId }));
  }, []);

  const handleAcceptSuggestion = useCallback((trackedId: number) => {
    const sug = suggestions[trackedId];
    if (sug) {
      setMappings(prev => ({ ...prev, [String(trackedId)]: sug.lineupPlayerId }));
    }
  }, [suggestions]);

  const handleAcceptAll = useCallback(() => {
    const newMappings = { ...mappings };
    const usedIds = new Set(Object.values(newMappings).filter(Boolean));
    for (const tp of trackedPlayers) {
      if (newMappings[String(tp.id)]) continue; // skip already mapped
      const sug = computeSuggestions(tp as any, lineupPlayers, usedIds);
      if (sug) {
        newMappings[String(tp.id)] = sug.lineupPlayerId;
        usedIds.add(sug.lineupPlayerId);
      }
    }
    setMappings(newMappings);
  }, [mappings, trackedPlayers, lineupPlayers]);

  const handleSaveMappings = useCallback(() => {
    if (!job?.jobId || !currentSetup) return;
    const finalMappings: Record<string, any> = {};
    Object.entries(mappings).forEach(([pid, lineupPlayerId]) => {
      if (!lineupPlayerId) return;
      const pd = lineupPlayers.find(p => p.id === lineupPlayerId);
      if (pd) {
        finalMappings[pid] = { id: pd.id, name: pd.name, number: pd.number, position: pd.position, team: pd.team };
      }
    });
    const payload = {
      setup_id: currentSetup.match_id,
      team_a_name: currentSetup.team_a.name,
      team_b_name: currentSetup.team_b.name,
      mappings: finalMappings,
    };
    try {
      const all = JSON.parse(localStorage.getItem('gaffer-player-mappings') || '{}');
      all[job.jobId] = payload;
      localStorage.setItem('gaffer-player-mappings', JSON.stringify(all));
      setSaveSuccess('Mappings saved successfully!');
      setTimeout(() => setSaveSuccess(null), 3000);
    } catch (e) {
      console.error(e);
      setSaveSuccess('Failed to save mappings.');
    }
  }, [job?.jobId, currentSetup, mappings, lineupPlayers]);

  const handleClearMappings = useCallback(() => {
    setMappings({});
    if (job?.jobId) {
      try {
        const all = JSON.parse(localStorage.getItem('gaffer-player-mappings') || '{}');
        delete all[job.jobId];
        localStorage.setItem('gaffer-player-mappings', JSON.stringify(all));
      } catch (e) { console.error(e); }
    }
  }, [job?.jobId]);

  const mappedCount = Object.values(mappings).filter(Boolean).length;
  const totalTracked = trackedPlayers.length;

  // ── No job guard ────────────────────────────────────────────────────────
  if (!job) {
    return (
      <div className="h-full w-full bg-[#050805] flex flex-col items-center justify-center p-8 text-center">
        <AlertTriangle className="text-amber-500 mb-4" size={48} />
        <h2 className="text-lg font-bold text-gray-300">No active job loaded</h2>
        <p className="text-xs text-gray-600 font-mono mt-1 max-w-sm">
          Please upload a video or open a historical match from the Match Centre first.
        </p>
      </div>
    );
  }

  // ── Group tracked players by team ──────────────────────────────────────
  const team0Players = (trackedPlayers as (TrackedPlayerStats & { positions: [number, number][] })[]).filter(tp => tp.teamId === 'team_0');
  const team1Players = (trackedPlayers as (TrackedPlayerStats & { positions: [number, number][] })[]).filter(tp => tp.teamId === 'team_1');

  const teamAName = currentSetup?.team_a.name ?? 'Red Team';
  const teamBName = currentSetup?.team_b.name ?? 'Blue Team';

  // ── Render ──────────────────────────────────────────────────────────────
  return (
    <div className="h-full w-full bg-[#050805] flex flex-col font-sans overflow-hidden">

      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div className="px-8 pt-7 pb-5 border-b border-gray-900/60 bg-gradient-to-b from-[#0a120a] to-[#050805] shrink-0">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
              <ArrowRightLeft size={18} className="text-emerald-500" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-100 tracking-tight">Player Identity Mapping</h1>
              <p className="text-xs text-gray-500 font-mono mt-0.5">Map CV tracker seeds → real lineup profiles</p>
            </div>
          </div>

          <div className="flex items-center gap-3 flex-wrap">
            {/* Progress pill */}
            <div className="flex items-center gap-2 bg-black/40 border border-gray-900 px-3 py-2 rounded-xl">
              <div className="h-1.5 w-24 bg-gray-900 rounded-full overflow-hidden">
                <div
                  className="h-full bg-emerald-500 rounded-full transition-all"
                  style={{ width: `${totalTracked > 0 ? (mappedCount / totalTracked) * 100 : 0}%` }}
                />
              </div>
              <span className="text-[10px] font-mono text-gray-400">
                {mappedCount}/{totalTracked} mapped
              </span>
            </div>

            {/* Lineup source */}
            <div className="flex items-center gap-2 bg-black/40 border border-gray-900 px-3 py-2 rounded-xl">
              <span className="text-[10px] font-mono text-gray-600 uppercase shrink-0">Lineup:</span>
              {matchSetups.length === 0 ? (
                <span className="text-[10px] text-amber-500 font-mono">No setups found</span>
              ) : (
                <select
                  id="setup-source-select"
                  value={selectedSetupId}
                  onChange={e => setSelectedSetupId(e.target.value)}
                  className="bg-[#050805] border border-gray-800 text-gray-300 text-xs font-mono rounded px-2 py-0.5 focus:outline-none focus:border-emerald-500/50 cursor-pointer"
                >
                  {matchSetups.map(s => (
                    <option key={s.match_id} value={s.match_id}>{s.fixture_name}</option>
                  ))}
                </select>
              )}
            </div>

            {/* Auto-suggest all */}
            {lineupPlayers.length > 0 && (
              <button
                onClick={handleAcceptAll}
                className="flex items-center gap-1.5 px-3 py-2 rounded-xl border border-emerald-500/30 bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 text-[10px] font-mono font-bold uppercase tracking-wide transition-all"
              >
                <Sparkles size={11} />
                Auto-Map All
              </button>
            )}
          </div>
        </div>
      </div>

      {/* ── Body ───────────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto px-8 py-6 space-y-10">
        {trackedPlayers.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-center">
            <Database className="text-gray-800 mb-4 animate-pulse" size={48} />
            <p className="text-gray-500 font-mono text-xs">No tracker seeds resolved in telemetry frames.</p>
          </div>
        ) : (
          <>
            {[
              { label: teamAName, players: team0Players, teamId: 'team_0', accentClass: 'text-blue-400', borderClass: 'border-blue-500/20', bgClass: 'bg-blue-500/5' },
              { label: teamBName, players: team1Players, teamId: 'team_1', accentClass: 'text-red-400', borderClass: 'border-red-500/20', bgClass: 'bg-red-500/5' },
            ].map(({ label, players, teamId, accentClass, borderClass, bgClass }) => (
              players.length > 0 && (
                <div key={teamId}>
                  {/* Team Section Header */}
                  <div className={`flex items-center gap-3 mb-4`}>
                    <div className={`h-px flex-1 ${teamId === 'team_0' ? 'bg-blue-500/20' : 'bg-red-500/20'}`} />
                    <div className={`flex items-center gap-2 px-3 py-1 rounded-full border ${borderClass} ${bgClass}`}>
                      <Shield size={10} className={accentClass} />
                      <span className={`text-[10px] font-mono font-bold uppercase tracking-widest ${accentClass}`}>
                        {label}
                      </span>
                      <span className="text-[9px] text-gray-600 font-mono">
                        · {players.length} players
                      </span>
                    </div>
                    <div className={`h-px flex-1 ${teamId === 'team_0' ? 'bg-blue-500/20' : 'bg-red-500/20'}`} />
                  </div>

                  {/* Lineup Candidates sidebar (right column) */}
                  <div className="grid grid-cols-1 xl:grid-cols-[1fr_260px] gap-4">

                    {/* Left: Tracked players stack */}
                    <div className="space-y-3">
                      {players.map(tp => {
                        const tpFull = tp as TrackedPlayerStats & { positions: [number, number][] };
                        const mappedLineupId = mappings[String(tp.id)] || '';
                        const selectedPlayer = lineupPlayers.find(p => p.id === mappedLineupId);
                        const suggestion = suggestions[tp.id];
                        const inferredPos = inferPosition(tp.avgX, tp.teamId);
                        const isExpanded = expandedId === tp.id;
                        const isMapped = !!mappedLineupId;

                        return (
                          <div
                            key={tp.id}
                            id={`mapping-row-${tp.id}`}
                            className={`rounded-2xl border transition-all ${isMapped
                              ? 'border-emerald-500/20 bg-[#0a120a]/80'
                              : 'border-gray-900 bg-[#0a0f0a]'
                              }`}
                          >
                            {/* ── Main Row ─────────────────────────────── */}
                            <div className="flex items-start gap-4 p-4">

                              {/* Player avatar / ID */}
                              <div className={`shrink-0 h-11 w-11 rounded-xl flex flex-col items-center justify-center font-mono font-bold text-sm border ${isMapped
                                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                                : 'bg-gray-900/50 border-gray-800 text-gray-400'
                                }`}>
                                <span className="text-[10px] text-gray-600 leading-none">ID</span>
                                <span>{tp.id}</span>
                              </div>

                              {/* Stats */}
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 flex-wrap mb-2">
                                  <span className="text-sm font-bold text-gray-100">Player {tp.id}</span>
                                  <span className={`text-[9px] font-mono px-2 py-0.5 rounded-full border ${borderClass} ${bgClass} ${accentClass}`}>
                                    {POSITION_CLUSTER_LABELS[inferredPos]}
                                  </span>
                                  {isMapped && (
                                    <span className="flex items-center gap-1 text-[9px] text-emerald-400 font-mono bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded-full">
                                      <Check size={8} />
                                      Mapped
                                    </span>
                                  )}
                                </div>

                                {/* Metric chips */}
                                <div className="flex flex-wrap gap-x-4 gap-y-1">
                                  <div className="flex items-center gap-1 text-[10px] text-gray-500 font-mono">
                                    <Clock size={9} className="text-gray-600" />
                                    <span className="text-gray-300 font-bold">{tp.minutesTracked.toFixed(0)}'</span> tracked
                                  </div>
                                  <div className="flex items-center gap-1 text-[10px] text-gray-500 font-mono">
                                    <Activity size={9} className="text-gray-600" />
                                    <span className="text-gray-300 font-bold">{(tp.distanceM / 1000).toFixed(2)} km</span>
                                  </div>
                                  <div className="flex items-center gap-1 text-[10px] text-gray-500 font-mono">
                                    <TrendingUp size={9} className="text-gray-600" />
                                    <span className="text-gray-300 font-bold">{tp.avgSpeedKmh.toFixed(1)}</span> km/h avg
                                  </div>
                                  <div className="flex items-center gap-1 text-[10px] text-gray-500 font-mono">
                                    <Zap size={9} className="text-emerald-700" />
                                    <span className="text-gray-300 font-bold">{tp.maxSpeedKmh.toFixed(1)}</span> km/h max
                                  </div>
                                  <div className="flex items-center gap-1 text-[10px] text-gray-500 font-mono">
                                    <MapPin size={9} className="text-gray-600" />
                                    {tp.avgX != null ? (
                                      <span className="text-gray-300 font-bold">{tp.avgX.toFixed(0)}m</span>
                                    ) : <span className="text-gray-600">—</span>}
                                    {' '}avg x
                                  </div>
                                  <div className="flex items-center gap-1 text-[10px] text-gray-500 font-mono">
                                    <Database size={9} className="text-gray-600" />
                                    <span className="text-gray-300 font-bold">{tp.activeFrames.toLocaleString()}</span> frames
                                  </div>
                                </div>
                              </div>

                              {/* Heatmap */}
                              <div className="shrink-0 hidden sm:block">
                                <MiniHeatmap positions={tpFull.positions} teamId={tp.teamId} />
                              </div>

                              {/* Expand toggle */}
                              <button
                                onClick={() => setExpandedId(isExpanded ? null : tp.id)}
                                className="shrink-0 text-gray-700 hover:text-gray-400 transition-colors p-1"
                              >
                                {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                              </button>
                            </div>

                            {/* ── Expanded: Suggestion + Manual Assign ── */}
                            {isExpanded && (
                              <div className="px-4 pb-4 border-t border-gray-900/60 pt-4 space-y-4">

                                {/* Auto-suggest pill */}
                                {suggestion && !isMapped && (
                                  <div className="flex items-center justify-between gap-3 bg-emerald-500/5 border border-emerald-500/15 rounded-xl px-4 py-3">
                                    <div className="flex items-start gap-3 min-w-0">
                                      <Sparkles size={14} className="text-emerald-500 shrink-0 mt-0.5" />
                                      <div>
                                        <div className="flex items-center gap-2 mb-0.5">
                                          <span className="text-xs font-bold text-emerald-300">
                                            #{suggestion.playerNumber} {suggestion.playerName}
                                          </span>
                                          <span className="text-[9px] font-mono text-gray-500">({suggestion.position})</span>
                                          <span className="text-[9px] font-mono text-gray-600">· {suggestion.teamName}</span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                          <div className="h-1 w-20 bg-gray-900 rounded-full overflow-hidden">
                                            <div
                                              className="h-full bg-emerald-500 rounded-full"
                                              style={{ width: `${suggestion.confidence}%` }}
                                            />
                                          </div>
                                          <span className="text-[9px] font-mono text-emerald-500">{suggestion.confidence}%</span>
                                          <span className="text-[9px] font-mono text-gray-600 hidden md:block">{suggestion.reason}</span>
                                        </div>
                                      </div>
                                    </div>
                                    <button
                                      onClick={() => handleAcceptSuggestion(tp.id)}
                                      className="shrink-0 flex items-center gap-1.5 px-3 py-1.5 bg-emerald-500 hover:bg-emerald-400 text-black text-[10px] font-bold rounded-lg transition-colors"
                                    >
                                      <Check size={10} />
                                      Accept
                                    </button>
                                  </div>
                                )}

                                {/* If already mapped, show current assignment with option to clear */}
                                {isMapped && selectedPlayer && (
                                  <div className="flex items-center justify-between gap-3 bg-emerald-500/5 border border-emerald-500/15 rounded-xl px-4 py-3">
                                    <div className="flex items-center gap-3">
                                      <CheckCircle size={14} className="text-emerald-500 shrink-0" />
                                      <div>
                                        <span className="text-xs font-bold text-emerald-300">
                                          #{selectedPlayer.number} {selectedPlayer.name}
                                        </span>
                                        <span className="text-[9px] font-mono text-gray-500 ml-2">({selectedPlayer.position}) · {selectedPlayer.teamName}</span>
                                      </div>
                                    </div>
                                    <button
                                      onClick={() => handleMapChange(tp.id, '')}
                                      className="shrink-0 flex items-center gap-1 text-[9px] text-gray-600 hover:text-red-400 font-mono transition-colors px-2 py-1 rounded"
                                    >
                                      <X size={9} /> Unassign
                                    </button>
                                  </div>
                                )}

                                {/* Manual override */}
                                <div className="flex flex-col gap-1.5">
                                  <span className="text-[9px] font-mono text-gray-600 uppercase tracking-widest">Manual assignment</span>
                                  <div className="flex items-center gap-2">
                                    <select
                                      id={`map-select-${tp.id}`}
                                      value={mappedLineupId}
                                      onChange={e => handleMapChange(tp.id, e.target.value)}
                                      className="flex-1 max-w-xs bg-black border border-gray-800 hover:border-emerald-500/20 text-gray-300 text-xs font-mono rounded-xl px-3 py-2 focus:outline-none focus:border-emerald-500/50 cursor-pointer"
                                    >
                                      <option value="">-- Unassigned --</option>
                                      {currentSetup && (
                                        <>
                                          <optgroup label={currentSetup.team_a.name}>
                                            {lineupPlayers.filter(p => p.team === 'A').sort((a, b) => a.number - b.number).map(p => (
                                              <option key={p.id} value={p.id}>
                                                #{p.number} {p.name} ({p.position}){p.isStarting ? ' ★' : ''}
                                              </option>
                                            ))}
                                          </optgroup>
                                          <optgroup label={currentSetup.team_b.name}>
                                            {lineupPlayers.filter(p => p.team === 'B').sort((a, b) => a.number - b.number).map(p => (
                                              <option key={p.id} value={p.id}>
                                                #{p.number} {p.name} ({p.position}){p.isStarting ? ' ★' : ''}
                                              </option>
                                            ))}
                                          </optgroup>
                                        </>
                                      )}
                                    </select>
                                    {selectedPlayer && (
                                      <span className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-[9px] font-mono font-medium border ${selectedPlayer.team === 'A'
                                        ? 'bg-blue-500/10 text-blue-400 border-blue-500/20'
                                        : 'bg-red-500/10 text-red-400 border-red-500/20'
                                        }`}>
                                        {selectedPlayer.teamName.slice(0, 14)}
                                      </span>
                                    )}
                                  </div>
                                </div>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>

                    {/* Right: Lineup candidates reference panel */}
                    <div className="hidden xl:block">
                      <div className={`rounded-2xl border ${borderClass} ${bgClass} p-4 sticky top-4`}>
                        <div className={`text-[9px] font-mono font-bold uppercase tracking-widest ${accentClass} mb-3 flex items-center gap-1.5`}>
                          <Users size={9} />
                          {label} Roster
                        </div>
                        <div className="space-y-1">
                          {lineupPlayers.filter(p => p.team === (teamId === 'team_0' ? 'A' : 'B')).sort((a, b) => a.number - b.number).map(p => {
                            const alreadyMapped = Object.values(mappings).includes(p.id);
                            return (
                              <div
                                key={p.id}
                                className={`flex items-center justify-between gap-2 px-2 py-1.5 rounded-lg ${alreadyMapped ? 'opacity-40' : ''}`}
                              >
                                <div className="flex items-center gap-2 min-w-0">
                                  <span className={`text-[10px] font-mono font-bold w-6 text-right ${accentClass}`}>
                                    #{p.number}
                                  </span>
                                  <span className="text-[10px] text-gray-300 font-medium truncate">{p.name}</span>
                                </div>
                                <div className="flex items-center gap-1 shrink-0">
                                  <span className="text-[8px] font-mono text-gray-600">{p.position}</span>
                                  {alreadyMapped && <Check size={8} className="text-emerald-500" />}
                                  {p.isStarting && !alreadyMapped && (
                                    <span className="text-[7px] text-yellow-600">★</span>
                                  )}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    </div>

                  </div>
                </div>
              )
            ))}
          </>
        )}
      </div>

      {/* ── Footer Controls ─────────────────────────────────────────────── */}
      <div className="shrink-0 px-8 py-4 border-t border-gray-900 bg-[#050805] flex justify-between items-center gap-4">
        <button
          onClick={handleClearMappings}
          className="px-4 py-2 border border-gray-900 rounded-xl text-xs font-mono text-gray-500 hover:text-red-400 hover:border-red-500/20 transition-all font-bold"
        >
          Reset All
        </button>

        <div className="flex items-center gap-4">
          {saveSuccess && (
            <span className="text-emerald-400 text-xs font-mono flex items-center gap-1.5">
              <CheckCircle size={12} /> {saveSuccess}
            </span>
          )}
          <button
            id="save-mappings-btn"
            onClick={handleSaveMappings}
            className="px-6 py-2.5 bg-emerald-500 hover:bg-emerald-400 text-black font-bold rounded-xl text-xs transition-colors tracking-wide flex items-center gap-1.5"
          >
            <Save size={13} /> Save Assignments
          </button>
        </div>
      </div>

    </div>
  );
}
