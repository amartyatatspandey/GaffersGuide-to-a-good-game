"use client";
import React, { useState, useEffect, useMemo } from 'react';
import { 
  Users, User, Play, Bot, AlertTriangle, Loader2, BarChart2, 
  ShieldAlert, Zap, Clock, Film, ChevronLeft, Search, ArrowUpDown, 
  Gauge, TrendingUp, Compass, Award 
} from 'lucide-react';
import { getJobEvents } from '@/lib/api/jobs';
import { getApiBaseUrl, getAuthHeaders } from '@/lib/apiBase';
import { PlayerDetailView } from './PlayerDetailView';

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

// ── Physical constants for football speed validation ──────────────────────
// A football pitch is 105m × 68m. At 25 FPS, the maximum realistic
// displacement per frame for a sprinting player (~38 km/h ≈ 10.6 m/s)
// is ~0.42m. We use 4.2m as a generous threshold to account for
// interpolation noise while still rejecting obvious teleportation.
const MAX_DISPLACEMENT_PER_FRAME_M = 4.2;   // reject jumps > this per frame gap=1
const MAX_SPEED_KMH_CAP = 50.0;              // absolute hard cap
const MAX_REALISTIC_SPRINT_KMH = 38.0;       // Usain Bolt on grass
const SPEED_OUTLIER_PERCENTILE = 0.95;        // use 95th percentile for "max speed"

// Heuristic to calculate physical performance metrics from tracking frames
const computePhysicalMetrics = (frames: any[]) => {
  const playersMap: Record<number, {
    id: number;
    teamId: string | null;
    activeFrames: number;
    distanceCovered: number;
    speeds: number[];
    lastFrameIdx: number | null;
    lastX: number | null;
    lastY: number | null;
    xPitchSum: number;
    yPitchSum: number;
    xPitchCount: number;
  }> = {};

  const fps = 25.0;

  for (const frame of frames || []) {
    const fIdx = frame.frame_idx;
    const players = frame.players || [];
    for (const p of players) {
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
          xPitchSum: 0,
          yPitchSum: 0,
          xPitchCount: 0
        };
      }

      const pData = playersMap[pid];
      
      // Update team_id if it was null previously and is now resolved
      if (!pData.teamId && p.team_id) {
        pData.teamId = p.team_id;
      }

      if (p.x_pitch !== undefined && p.y_pitch !== undefined && p.x_pitch !== null && p.y_pitch !== null) {
        pData.activeFrames += 1;
        pData.xPitchSum += p.x_pitch;
        pData.yPitchSum += p.y_pitch;
        pData.xPitchCount += 1;

        if (p.speed_kmh !== undefined && p.speed_kmh !== null) {
          // Cap backend speed values — they can also contain teleportation artifacts
          const clampedSpeed = Math.min(p.speed_kmh, MAX_SPEED_KMH_CAP);
          if (clampedSpeed > 0) {
            pData.speeds.push(clampedSpeed);
          }
        }

        // Calculate distance step with teleportation rejection
        if (pData.lastFrameIdx !== null && pData.lastX !== null && pData.lastY !== null) {
          const gap = fIdx - pData.lastFrameIdx;
          if (gap > 0 && gap < fps * 2) { // less than 2 seconds gap
            const dist = Math.hypot(p.x_pitch - pData.lastX, p.y_pitch - pData.lastY);
            // Reject teleportation: max plausible displacement scales with gap
            const maxAllowedDist = MAX_DISPLACEMENT_PER_FRAME_M * gap;
            if (dist <= maxAllowedDist) {
              pData.distanceCovered += dist;
            }
            // else: skip this segment — tracker likely teleported
          }
        }
        pData.lastFrameIdx = fIdx;
        pData.lastX = p.x_pitch;
        pData.lastY = p.y_pitch;
      }
    }
  }

  // Calculate sprints
  const sprintThreshold = 24.0;
  const resetThreshold = 19.0;

  const result: Record<number, {
    id: number;
    teamId: string | null;
    distanceCovered: number; // in meters
    avgSpeed: number; // in km/h
    maxSpeed: number; // in km/h
    sprintCount: number;
    activeFrames: number;
    speeds: number[];
    avgX: number | undefined;
    avgY: number | undefined;
  }> = {};

  for (const pidStr of Object.keys(playersMap)) {
    const pid = Number(pidStr);
    const pData = playersMap[pid];
    
    // Filter speed array: remove values above realistic sprint threshold
    const validSpeeds = pData.speeds.filter(s => s > 0 && s <= MAX_REALISTIC_SPRINT_KMH);
    
    // Compute max speed using 95th percentile (robust to remaining noise)
    let maxSpeed = 0.0;
    if (validSpeeds.length > 0) {
      const sorted = [...validSpeeds].sort((a, b) => a - b);
      const p95Idx = Math.min(sorted.length - 1, Math.floor(sorted.length * SPEED_OUTLIER_PERCENTILE));
      maxSpeed = sorted[p95Idx];
    }
    
    // Compute average speed: (distance / duration) * 3.6
    const durationSeconds = pData.activeFrames / fps;
    let avgSpeed = durationSeconds > 0 ? (pData.distanceCovered / durationSeconds) * 3.6 : 0.0;
    // Hard cap average speed — even with filtering, aggregated distance can spike
    avgSpeed = Math.min(avgSpeed, MAX_REALISTIC_SPRINT_KMH);

    // Compute sprints
    let sprintCount = 0;
    let isSprinting = false;
    for (const speed of pData.speeds) {
      if (speed >= sprintThreshold && !isSprinting) {
        isSprinting = true;
        sprintCount++;
      } else if (speed < resetThreshold) {
        isSprinting = false;
      }
    }

    result[pid] = {
      id: pid,
      teamId: pData.teamId,
      distanceCovered: Math.round(pData.distanceCovered * 10) / 10,
      avgSpeed: Math.round(Math.min(avgSpeed, MAX_SPEED_KMH_CAP) * 10) / 10,
      maxSpeed: Math.round(Math.min(maxSpeed, MAX_SPEED_KMH_CAP) * 10) / 10,
      sprintCount,
      activeFrames: pData.activeFrames,
      speeds: validSpeeds,  // use filtered speeds downstream
      avgX: pData.xPitchCount > 0 ? pData.xPitchSum / pData.xPitchCount : undefined,
      avgY: pData.xPitchCount > 0 ? pData.yPitchSum / pData.xPitchCount : undefined
    };
  }

  return result;
};

// Heuristic to classify player's tactical position based on pitch locations
const getPlayerPosition = (playerId: number, frames: any[]) => {
  const pFrames = (frames || []).flatMap(f => f.players || []).filter((pl: any) => pl.id === playerId);
  if (pFrames.length === 0) return "Midfielder";
  
  const xs = pFrames.map((pl: any) => pl.x_pitch).filter(x => x !== null && x !== undefined);
  if (xs.length === 0) return "Midfielder";
  
  const avgX = xs.reduce((a, b) => a + b, 0) / xs.length;
  const varianceX = xs.reduce((sum, val) => sum + Math.pow(val - avgX, 2), 0) / xs.length;
  const stdDevX = Math.sqrt(varianceX);
  
  // Goalkeepers stay very close to their goal line with extremely low variance
  if (stdDevX < 6.0 && (avgX < 12.0 || avgX > 93.0)) {
    return "Goalkeeper";
  }
  
  const teamId = pFrames.find((pl: any) => pl.team_id)?.team_id || null;
  
  if (teamId === 'team_0') {
    if (avgX < 40) return "Defender";
    if (avgX > 68) return "Forward";
    return "Midfielder";
  } else if (teamId === 'team_1') {
    if (avgX > 65) return "Defender";
    if (avgX < 37) return "Forward";
    return "Midfielder";
  } else {
    if (avgX < 35 || avgX > 70) return "Defender / Forward";
    return "Midfielder";
  }
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
  
  // Roster view states
  const [selectedPlayerId, setSelectedPlayerId] = useState<number | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [teamFilter, setTeamFilter] = useState<'all' | 'team_0' | 'team_1' | 'unclassified'>('all');
  const [sortBy, setSortBy] = useState<'id' | 'distanceCovered' | 'avgSpeed' | 'maxSpeed' | 'sprintCount' | 'threatScore' | 'threatRank'>('threatScore');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  // Clips queried from AI evidence system
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

  // Read mappings from localStorage
  const savedMappings = useMemo(() => {
    if (typeof window === 'undefined' || !job?.jobId) return null;
    const stored = localStorage.getItem('gaffer-player-mappings');
    if (!stored) return null;
    try {
      const all = JSON.parse(stored);
      return all[job.jobId] || null;
    } catch {
      return null;
    }
  }, [job?.jobId]);

  const getTeamLabel = (teamId: string | null) => {
    if (!teamId || teamId === "unknown") return "Unclassified";
    if (savedMappings) {
      if (teamId === 'team_0' && savedMappings.team_a_name) return savedMappings.team_a_name;
      if (teamId === 'team_1' && savedMappings.team_b_name) return savedMappings.team_b_name;
    }
    if (useAltNames && dictionary[teamId]) return dictionary[teamId];
    return teamId === "team_0" ? "Red Team" : "Blue Team";
  };

  const getPlayerLabel = (playerId: number) => {
    if (savedMappings && savedMappings.mappings?.[String(playerId)]) {
      return savedMappings.mappings[String(playerId)].name;
    }
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

  const handlePlayerClick = (playerId: number) => {
    setSelectedPlayerId(playerId);
    fetchClipsForPlayer(playerId);
  };

  // Compute physical performance metrics
  const physicalMetrics = useMemo(() => {
    if (!job?.tracking?.frames) return {};
    return computePhysicalMetrics(job.tracking.frames);
  }, [job?.tracking?.frames]);

  // Aggregate tracking & threat details into a unified profile per player
  const playerData = useMemo(() => {
    const THREAT_WEIGHTS: Record<string, number> = {
      "THR_001": 2.5,
      "THR_002": 1.2,
      "THR_003": 1.8,
      "THR_004": 0.8,
      "THR_005": 1.5,
      "THR_006": 2.0,
      "THR_007": 1.6,
    };

    const DOMINANT_TEMPLATES: Record<string, string> = {
      "THR_001": "Player {pid} repeatedly finds space behind the defensive line through timing and movement.",
      "THR_003": "Player {pid} consistently reaches the box, indicating a structured attacking run pattern.",
      "THR_006": "Player {pid} is exploiting the channel between the centre-back and full-back.",
      "THR_007": "Player {pid} is winning 1v1 duels in wide areas, creating crossing and cutting opportunities.",
      "THR_002": "Player {pid} frequently penetrates the final third, maintaining constant forward pressure.",
      "THR_005": "Player {pid} receives in dangerous areas and creates problems when in possession.",
      "THR_004": "Player {pid} is highly involved in transitions, participating in most attacking sequences.",
    };

    const pluralizeEventName = (code: string) => {
      const lookups: Record<string, string> = {
        "THR_001": "Dangerous Runs",
        "THR_002": "Final-Third Entries",
        "THR_003": "Box Entries",
        "THR_004": "Transition Involvements",
        "THR_005": "Dangerous Receptions",
        "THR_006": "Channel Exploitations",
        "THR_007": "Isolated Defender Exploits",
      };
      return lookups[code] || EVENT_NAME_LOOKUP[code] || code;
    };

    const generateExplanation = (playerId: number, teamLabel: string, threatScore: number, eventCounts: Record<string, number>, primaryThreatTypes: string[]) => {
      const verdict = `Player ${playerId} (${teamLabel}) threat score: ${threatScore.toFixed(1)}/100.`;
      const bullets = primaryThreatTypes.map(et => {
        const count = eventCounts[et] || 0;
        const name = pluralizeEventName(et);
        return `  • ${count}× ${name}`;
      });
      const evidence = bullets.length > 0 ? `Primary contributions:\n${bullets.join('\n')}` : "";
      
      const dominant = primaryThreatTypes[0];
      const pattern = dominant ? (DOMINANT_TEMPLATES[dominant]?.replace("{pid}", String(playerId)) || "") : "";
      
      return [verdict, evidence, pattern].filter(Boolean).join('\n\n');
    };

    const POSITION_ORDER: Record<string, number> = {
      "Goalkeeper": 1,
      "Defender": 2,
      "Midfielder": 3,
      "Forward": 4,
      "Defender / Forward": 5
    };

    const playersList: any[] = [];
    const threatProfiles = data?.threat_profiles || [];
    
    const threatPids = threatProfiles.map((p: any) => p.player_id);
    const physicalPids = Object.keys(physicalMetrics).map(Number);
    const allPids = Array.from(new Set([...threatPids, ...physicalPids]));

    // 1. Resolve team for each player ID
    const resolvedTeams: Record<number, string> = {};
    for (const pid of allPids) {
      const physical = physicalMetrics[pid];
      const threatProfile = threatProfiles.find((p: any) => p.player_id === pid);
      let teamId = physical?.teamId || threatProfile?.team_id || null;
      if (teamId === "unknown" || !teamId) {
        teamId = null;
      }
      if (teamId) {
        resolvedTeams[pid] = teamId;
      }
    }

    // 2. Filter out referee, outliers, and keep team_0/team_1
    const team0Players = allPids.filter(pid => resolvedTeams[pid] === 'team_0');
    const team1Players = allPids.filter(pid => resolvedTeams[pid] === 'team_1');

    // 3. Sort by active frame count to identify seeds
    team0Players.sort((a, b) => (physicalMetrics[b]?.activeFrames || 0) - (physicalMetrics[a]?.activeFrames || 0));
    team1Players.sort((a, b) => (physicalMetrics[b]?.activeFrames || 0) - (physicalMetrics[a]?.activeFrames || 0));

    // Choose top K = 13 seeds per team (allowing maximum 26 participants)
    const team0Seeds = new Set(team0Players.slice(0, Math.min(13, team0Players.length)));
    const team1Seeds = new Set(team1Players.slice(0, Math.min(13, team1Players.length)));
    const allSeeds = new Set([...team0Seeds, ...team1Seeds]);

    // 4. Create merge map (maps tracker ID to target seed)
    const mergeMap: Record<number, number> = {};
    allSeeds.forEach(pid => {
      mergeMap[pid] = pid;
    });

    for (const pid of allPids) {
      if (allSeeds.has(pid)) continue;
      const team = resolvedTeams[pid];
      if (team !== 'team_0' && team !== 'team_1') continue;

      const seeds = team === 'team_0' ? team0Seeds : team1Seeds;
      if (seeds.size === 0) continue;

      const pPhys = physicalMetrics[pid];
      const avgX = pPhys?.avgX;
      const avgY = pPhys?.avgY;

      let bestSeed = Array.from(seeds)[0];
      if (avgX !== undefined && avgY !== undefined) {
        let minD = Infinity;
        for (const seed of seeds) {
          const sPhys = physicalMetrics[seed];
          const sAvgX = sPhys?.avgX;
          const sAvgY = sPhys?.avgY;
          if (sAvgX !== undefined && sAvgY !== undefined) {
            const dist = Math.hypot(avgX - sAvgX, avgY - sAvgY);
            if (dist < minD) {
              minD = dist;
              bestSeed = seed;
            }
          }
        }
      } else {
        let maxThreat = -1;
        for (const seed of seeds) {
          const tProfile = threatProfiles.find((p: any) => p.player_id === seed);
          const score = tProfile?.threat_score || 0;
          if (score > maxThreat) {
            maxThreat = score;
            bestSeed = seed;
          }
        }
      }
      mergeMap[pid] = bestSeed;
    }

    // 5. Aggregate metrics for seeds
    const aggregated: Record<number, any> = {};
    allSeeds.forEach(seedPid => {
      const team = resolvedTeams[seedPid];
      const seedPhys = physicalMetrics[seedPid] || {
        id: seedPid,
        teamId: team,
        distanceCovered: 0,
        avgSpeed: 0,
        maxSpeed: 0,
        sprintCount: 0,
        activeFrames: 0,
        speeds: []
      };

      const seedThreat = threatProfiles.find((p: any) => p.player_id === seedPid) || {
        player_id: seedPid,
        team_id: team,
        event_counts: {},
        event_ids: {},
        threat_score: 0,
        threat_rank: 999,
        primary_threat_types: [],
        explanation: ""
      };

      aggregated[seedPid] = {
        id: seedPid,
        teamId: team,
        distanceCovered: seedPhys.distanceCovered,
        sprintCount: seedPhys.sprintCount,
        activeFrames: seedPhys.activeFrames,
        speeds: [...(seedPhys.speeds || [])],
        eventCounts: { ...(seedThreat.event_counts || {}) },
        eventIds: { ...(seedThreat.event_ids || {}) },
        threatScore: seedThreat.threat_score || 0,
        threatRank: seedThreat.threat_rank || 999,
        primaryThreatTypes: [...(seedThreat.primary_threat_types || [])],
        explanations: seedThreat.explanation ? [seedThreat.explanation] : []
      };
    });

    for (const pid of allPids) {
      if (allSeeds.has(pid)) continue;
      const targetSeed = mergeMap[pid];
      if (!targetSeed) continue;

      const pPhys = physicalMetrics[pid];
      const pThreat = threatProfiles.find((p: any) => p.player_id === pid);

      const agg = aggregated[targetSeed];
      if (!agg) continue;

      if (pPhys) {
        agg.distanceCovered += pPhys.distanceCovered;
        agg.sprintCount += pPhys.sprintCount;
        agg.activeFrames += pPhys.activeFrames;
        if (pPhys.speeds) {
          agg.speeds.push(...pPhys.speeds);
        }
      }

      if (pThreat) {
        if (pThreat.event_counts) {
          Object.entries(pThreat.event_counts).forEach(([code, count]) => {
            agg.eventCounts[code] = (agg.eventCounts[code] || 0) + (count as number);
          });
        }
        if (pThreat.event_ids) {
          Object.entries(pThreat.event_ids).forEach(([code, ids]) => {
            agg.eventIds[code] = [...(agg.eventIds[code] || []), ...(ids as string[])];
          });
        }
        agg.threatScore = Math.max(agg.threatScore, pThreat.threat_score || 0);
        if (pThreat.explanation) {
          agg.explanations.push(pThreat.explanation);
        }
      }
    }

    // 6. Finalise profiles and assign positions / details
    const seedProfiles = Array.from(allSeeds).map(seedPid => {
      const agg = aggregated[seedPid];
      const mapping = savedMappings?.mappings?.[String(seedPid)];
      
      let position = job?.tracking?.frames ? getPlayerPosition(seedPid, job.tracking.frames) : "Midfielder";
      let jerseyNumber = undefined;
      let assignedTeamId = agg.teamId;

      if (mapping) {
        jerseyNumber = mapping.number;
        position = mapping.position === 'GK' ? 'Goalkeeper' :
                   mapping.position === 'DF' ? 'Defender' :
                   mapping.position === 'MF' ? 'Midfielder' :
                   mapping.position === 'FW' ? 'Forward' : mapping.position;
        // Keep the teamId aligned with mapping selection
        assignedTeamId = mapping.team === 'A' ? 'team_0' : 'team_1';
      }

      // Use percentile-based max speed to resist outlier noise from merged tracker IDs
      let maxSpeed = 0.0;
      const aggValidSpeeds = agg.speeds.filter((s: number) => s > 0 && s <= MAX_REALISTIC_SPRINT_KMH);
      if (aggValidSpeeds.length > 0) {
        const sorted = [...aggValidSpeeds].sort((a: number, b: number) => a - b);
        const p95Idx = Math.min(sorted.length - 1, Math.floor(sorted.length * SPEED_OUTLIER_PERCENTILE));
        maxSpeed = sorted[p95Idx];
      }
      const durationSeconds = agg.activeFrames / 25.0;
      let avgSpeed = durationSeconds > 0 ? (agg.distanceCovered / durationSeconds) * 3.6 : 0.0;
      avgSpeed = Math.min(avgSpeed, MAX_REALISTIC_SPRINT_KMH);
      
      // Calculate top 3 primary threat types based on combined event counts weighted contribution
      const contributions = Object.entries(agg.eventCounts).map(([et, count]) => ({
        et,
        score: (count as number) * (THREAT_WEIGHTS[et] || 0)
      }));
      contributions.sort((a, b) => b.score - a.score);
      const primaryThreatTypes = contributions.slice(0, 3).map(c => c.et);

      const teamLabel = assignedTeamId === 'team_0' ? (savedMappings?.team_a_name || 'Red Team') : (savedMappings?.team_b_name || 'Blue Team');
      const explanation = generateExplanation(seedPid, teamLabel, agg.threatScore, agg.eventCounts, primaryThreatTypes);

      return {
        id: seedPid,
        teamId: assignedTeamId,
        position,
        distanceCovered: Math.round(agg.distanceCovered * 10) / 10,
        avgSpeed: Math.round(Math.min(avgSpeed, MAX_SPEED_KMH_CAP) * 10) / 10,
        maxSpeed: Math.round(Math.min(maxSpeed, MAX_SPEED_KMH_CAP) * 10) / 10,
        sprintCount: agg.sprintCount,
        minutesPlayed: Math.round(durationSeconds / 60),
        threatScore: agg.threatScore,
        threatRank: 999, // Assigned below
        primaryThreatTypes,
        explanation,
        eventCounts: agg.eventCounts,
        eventIds: agg.eventIds,
        jerseyNumber
      };
    });

    // 7. Sort by threatScore within each team to compute threatRank
    const team0SeedsList = seedProfiles.filter(p => p.teamId === 'team_0');
    team0SeedsList.sort((a, b) => b.threatScore - a.threatScore);
    team0SeedsList.forEach((p, idx) => {
      p.threatRank = idx + 1;
    });

    const team1SeedsList = seedProfiles.filter(p => p.teamId === 'team_1');
    team1SeedsList.sort((a, b) => b.threatScore - a.threatScore);
    team1SeedsList.forEach((p, idx) => {
      p.threatRank = idx + 1;
    });

    // 8. Assign Jersey Numbers sequentially by position
    const assignJerseyNumbersForTeam = (teamList: any[]) => {
      const unmapped = teamList.filter(p => p.jerseyNumber === undefined);
      
      unmapped.sort((a, b) => {
        const orderA = POSITION_ORDER[a.position] || 99;
        const orderB = POSITION_ORDER[b.position] || 99;
        if (orderA !== orderB) return orderA - orderB;
        return b.minutesPlayed - a.minutesPlayed;
      });

      let dfCount = 0;
      let mfCount = 0;
      let fwCount = 0;

      const dfNumbers = [2, 3, 4, 5, 12, 13, 15];
      const mfNumbers = [6, 8, 10, 14, 16, 20, 21];
      const fwNumbers = [7, 9, 11, 17, 18, 19, 22];

      unmapped.forEach(p => {
        if (p.position === "Goalkeeper") {
          p.jerseyNumber = 1;
        } else if (p.position === "Defender") {
          p.jerseyNumber = dfNumbers[dfCount % dfNumbers.length];
          dfCount++;
        } else if (p.position === "Midfielder") {
          p.jerseyNumber = mfNumbers[mfCount % mfNumbers.length];
          mfCount++;
        } else if (p.position === "Forward") {
          p.jerseyNumber = fwNumbers[fwCount % fwNumbers.length];
          fwCount++;
        } else {
          p.jerseyNumber = 23 + (dfCount + mfCount + fwCount);
        }
      });
    };

    assignJerseyNumbersForTeam(team0SeedsList);
    assignJerseyNumbersForTeam(team1SeedsList);

    return { players: [...team0SeedsList, ...team1SeedsList], mergeMap };
  }, [data?.threat_profiles, physicalMetrics, job?.tracking?.frames, savedMappings]);

  const allPlayers = playerData.players;
  const playerMergeMap = playerData.mergeMap;

  // Filter and sort roster
  const filteredAndSortedPlayers = useMemo(() => {
    return allPlayers
      .filter(p => {
        const nameMatch = getPlayerLabel(p.id).toLowerCase().includes(searchQuery.toLowerCase()) || 
                          p.id.toString().includes(searchQuery) ||
                          p.position.toLowerCase().includes(searchQuery.toLowerCase());
        
        if (teamFilter === 'all') return nameMatch;
        if (teamFilter === 'unclassified') return nameMatch && !p.teamId;
        return nameMatch && p.teamId === teamFilter;
      })
      .sort((a, b) => {
        let valA: any = a[sortBy];
        let valB: any = b[sortBy];
        
        if (sortBy === 'threatRank') {
          // threat rank is better if lower, reverse sorting direction
          valA = a.threatRank === 999 ? 9999 : a.threatRank;
          valB = b.threatRank === 999 ? 9999 : b.threatRank;
          return sortOrder === 'asc' ? valB - valA : valA - valB;
        }

        if (typeof valA === 'string') {
          return sortOrder === 'asc' 
            ? valA.localeCompare(valB) 
            : valB.localeCompare(valA);
        }
        
        return sortOrder === 'asc' ? valA - valB : valB - valA;
      });
  }, [allPlayers, searchQuery, teamFilter, sortBy, sortOrder]);

  const handleSort = (field: typeof sortBy) => {
    if (sortBy === field) {
      setSortOrder(prev => prev === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setSortOrder('desc');
    }
  };

  const threatColor = (score: number) => {
    if (score >= 70) return { text: "text-red-400 font-bold", border: "border-red-500/30", bg: "bg-red-500/10", bar: "bg-red-500" };
    if (score >= 40) return { text: "text-amber-400 font-bold", border: "border-amber-500/30", bg: "bg-amber-500/10", bar: "bg-amber-500" };
    return { text: "text-emerald-400 font-medium", border: "border-emerald-500/30", bg: "bg-emerald-500/10", bar: "bg-emerald-500" };
  };

  if (!job) {
    return (
      <div className="h-full w-full bg-[#0a0f0a] flex flex-col items-center justify-center p-8 text-center">
        <Users className="text-gray-800 mb-4 animate-pulse" size={64} />
        <h2 className="text-lg font-bold text-gray-400 font-mono uppercase tracking-widest">No Active Match Loaded</h2>
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
        <h2 className="text-sm font-bold text-amber-500 font-mono uppercase tracking-widest">Analysis Telemetry Unavailable</h2>
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

  // Find currently selected player details
  const selectedPlayer = allPlayers.find(p => p.id === selectedPlayerId);

  return (
    <div className="h-full w-full bg-[#050805] flex flex-col font-sans p-8 overflow-y-auto custom-scrollbar">
      
      {/* Roster List View */}
      {!selectedPlayer ? (
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8 border-b border-gray-900 pb-6">
            <div>
              <h1 className="text-2xl font-bold text-gray-200 tracking-tight flex items-center gap-3">
                <Users className="text-emerald-500" />
                Player Performance &amp; Threat Reports
              </h1>
              <p className="text-sm text-gray-500 mt-2 font-mono">
                Sourced from optical-flow tracking vectors and rule-based tactical ontology.
              </p>
            </div>
            <div className="flex items-center gap-6 bg-[#111a12]/30 border border-gray-900 px-6 py-3 rounded-xl font-mono text-xs">
              <div className="flex flex-col gap-0.5">
                <span className="text-gray-600 uppercase text-[9px] tracking-widest">Active Match</span>
                <span className="text-emerald-400 font-bold">{job?.file?.name || 'Historical Match'}</span>
              </div>
              <div className="h-8 w-px bg-gray-800" />
              <div className="flex flex-col gap-0.5">
                <span className="text-gray-600 uppercase text-[9px] tracking-widest">Total Rostered</span>
                <span className="text-gray-400 font-bold">{allPlayers.length} Players</span>
              </div>
            </div>
          </div>

          {/* Search, Filters, and Roster Controls */}
          <div className="flex flex-col lg:flex-row gap-4 mb-6">
            <div className="relative flex-1">
              <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 text-gray-600 h-4 w-4" />
              <input
                id="player-search"
                type="text"
                placeholder="Search players by ID, alias, or position..."
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                className="w-full bg-black/40 border border-gray-900 rounded-xl py-2.5 pl-10 pr-4 text-xs text-gray-200 focus:outline-none focus:border-emerald-500/50 transition-colors font-mono"
              />
            </div>
            
            <div className="flex flex-wrap items-center gap-4">
              {/* Team Filter */}
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-mono uppercase text-gray-600">Filter Team:</span>
                <select
                  id="team-filter"
                  value={teamFilter}
                  onChange={e => setTeamFilter(e.target.value as any)}
                  className="bg-black/60 border border-gray-900 rounded-xl px-3 py-2 text-xs font-mono text-gray-400 focus:outline-none focus:border-emerald-500/50"
                >
                  <option value="all">All Teams</option>
                  <option value="team_0">{getTeamLabel('team_0')}</option>
                  <option value="team_1">{getTeamLabel('team_1')}</option>
                  <option value="unclassified">Unclassified</option>
                </select>
              </div>
            </div>
          </div>

          {/* Main Roster Table */}
          <div className="flex-1 bg-black/20 border border-gray-900/60 rounded-2xl overflow-hidden shadow-xl">
            <div className="overflow-x-auto h-full">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-gray-900 bg-black/40 text-[10px] font-mono text-gray-500 uppercase tracking-wider select-none">
                    <th 
                      onClick={() => handleSort('id')} 
                      className="py-4 px-6 cursor-pointer hover:text-emerald-400 transition-colors"
                    >
                      <div className="flex items-center gap-1.5">
                        Player <ArrowUpDown size={12} />
                      </div>
                    </th>
                    <th className="py-4 px-6">Team</th>
                    <th className="py-4 px-6">Position</th>
                    <th 
                      onClick={() => handleSort('minutesPlayed' as any)} 
                      className="py-4 px-6 cursor-pointer hover:text-emerald-400 transition-colors"
                    >
                      <div className="flex items-center gap-1.5">
                        Mins <ArrowUpDown size={12} />
                      </div>
                    </th>
                    <th 
                      onClick={() => handleSort('distanceCovered')} 
                      className="py-4 px-6 cursor-pointer hover:text-emerald-400 transition-colors"
                    >
                      <div className="flex items-center gap-1.5">
                        Distance <ArrowUpDown size={12} />
                      </div>
                    </th>
                    <th 
                      onClick={() => handleSort('avgSpeed')} 
                      className="py-4 px-6 cursor-pointer hover:text-emerald-400 transition-colors"
                    >
                      <div className="flex items-center gap-1.5">
                        Avg Speed <ArrowUpDown size={12} />
                      </div>
                    </th>
                    <th 
                      onClick={() => handleSort('maxSpeed')} 
                      className="py-4 px-6 cursor-pointer hover:text-emerald-400 transition-colors"
                    >
                      <div className="flex items-center gap-1.5">
                        Max Speed <ArrowUpDown size={12} />
                      </div>
                    </th>
                    <th 
                      onClick={() => handleSort('sprintCount')} 
                      className="py-4 px-6 cursor-pointer hover:text-emerald-400 transition-colors"
                    >
                      <div className="flex items-center gap-1.5">
                        Sprints <ArrowUpDown size={12} />
                      </div>
                    </th>
                    <th 
                      onClick={() => handleSort('threatScore')} 
                      className="py-4 px-6 cursor-pointer hover:text-emerald-400 transition-colors"
                    >
                      <div className="flex items-center gap-1.5">
                        Threat Score <ArrowUpDown size={12} />
                      </div>
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-900/40 text-xs">
                  {filteredAndSortedPlayers.length === 0 ? (
                    <tr>
                      <td colSpan={9} className="py-12 text-center text-gray-600 font-mono italic">
                        No players matching search criteria found.
                      </td>
                    </tr>
                  ) : (
                    filteredAndSortedPlayers.map(p => {
                      const colors = threatColor(p.threatScore);
                      return (
                        <tr 
                          key={p.id}
                          id={`player-row-${p.id}`}
                          onClick={() => handlePlayerClick(p.id)}
                          className="hover:bg-[#111a12]/20 cursor-pointer transition-colors border-gray-900"
                        >
                          {/* Player ID / Name */}
                          <td className="py-4 px-6 font-bold text-gray-200">
                            <div className="flex items-center gap-3">
                              <div className="h-7 w-7 rounded-lg bg-gray-900 border border-gray-800 flex items-center justify-center text-gray-400 text-xs font-mono font-bold">
                                {p.jerseyNumber !== undefined ? `#${p.jerseyNumber}` : <User size={14} />}
                              </div>
                              <span>{getPlayerLabel(p.id)}</span>
                            </div>
                          </td>
                          
                          {/* Team */}
                          <td className="py-4 px-6">
                            <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-mono font-medium ${
                              p.teamId === 'team_0' 
                                ? 'bg-red-500/10 text-red-400 border border-red-500/20' 
                                : p.teamId === 'team_1' 
                                ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20' 
                                : 'bg-gray-800 text-gray-400'
                            }`}>
                              <span className={`h-1.5 w-1.5 rounded-full ${
                                p.teamId === 'team_0' ? 'bg-red-500' : p.teamId === 'team_1' ? 'bg-blue-500' : 'bg-gray-500'
                              }`} />
                              {getTeamLabel(p.teamId)}
                            </span>
                          </td>
                          
                          {/* Position */}
                          <td className="py-4 px-6 text-gray-400 font-mono">
                            {p.position}
                          </td>

                          {/* Minutes Played */}
                          <td className="py-4 px-6 font-mono text-gray-300">
                            {p.minutesPlayed !== undefined ? `${p.minutesPlayed} min` : '--'}
                          </td>
                          
                          {/* Distance Covered */}
                          <td className="py-4 px-6 font-mono text-gray-300">
                            {p.distanceCovered ? `${p.distanceCovered.toFixed(1)} m` : '--'}
                          </td>
                          
                          {/* Avg Speed */}
                          <td className="py-4 px-6 font-mono text-gray-300">
                            {p.avgSpeed ? `${p.avgSpeed.toFixed(1)} km/h` : '--'}
                          </td>
                          
                          {/* Max Speed */}
                          <td className="py-4 px-6 font-mono text-gray-300">
                            {p.maxSpeed ? `${p.maxSpeed.toFixed(1)} km/h` : '--'}
                          </td>
                          
                          {/* Sprints */}
                          <td className="py-4 px-6 font-mono text-gray-300">
                            <span className={p.sprintCount > 5 ? 'text-amber-400 font-bold' : ''}>
                              {p.sprintCount}
                            </span>
                          </td>
                          
                          {/* Threat Score */}
                          <td className="py-4 px-6 font-mono">
                            <span className={`px-2 py-0.5 rounded text-[10px] ${colors.bg} ${colors.text} border ${colors.border}`}>
                              {p.threatScore ? `${p.threatScore.toFixed(0)}/100` : '0/100'}
                            </span>
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      ) : (
        <PlayerDetailView
          player={selectedPlayer}
          job={job}
          mergeMap={playerMergeMap}
          allPlayers={allPlayers}
          getPlayerLabel={getPlayerLabel}
          getTeamLabel={getTeamLabel}
          onBack={() => setSelectedPlayerId(null)}
          onAskAI={onAskAI}
          onPlayClip={onPlayClip}
          playerClips={playerClips}
          loadingClips={loadingClips}
        />
      )}
    </div>
  );
}
