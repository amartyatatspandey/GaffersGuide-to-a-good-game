"use client";
import React, { useMemo, useRef, useEffect, useState } from 'react';
import {
  ChevronLeft, Bot, User, Compass, Award, TrendingUp, Clock,
  Zap, Gauge, BarChart2, ShieldAlert, Film, Play, Loader2,
  MapPin, Activity, Timer, Flame
} from 'lucide-react';

// ── Types ────────────────────────────────────────────────────────────────
interface PlayerDetailViewProps {
  player: any;
  job: any;
  mergeMap: Record<number, number>;
  allPlayers: any[];
  getPlayerLabel: (id: number) => string;
  getTeamLabel: (teamId: string | null) => string;
  onBack: () => void;
  onAskAI: (prompt: string) => void;
  onPlayClip: (startTimeS: number) => void;
  playerClips: Record<string, any[]>;
  loadingClips: Record<string, boolean>;
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

// ── Pitch dimensions (meters) ────────────────────────────────────────────
const PITCH_LENGTH = 105;
const PITCH_WIDTH = 68;

// ── Color helpers ────────────────────────────────────────────────────────
function heatColor(intensity: number): string {
  // 0 = transparent, 0.3 = blue, 0.5 = green, 0.7 = yellow, 1.0 = red
  const t = Math.max(0, Math.min(1, intensity));
  if (t < 0.25) {
    const s = t / 0.25;
    return `rgba(0, ${Math.round(60 + s * 140)}, ${Math.round(200 - s * 100)}, ${0.1 + s * 0.3})`;
  }
  if (t < 0.5) {
    const s = (t - 0.25) / 0.25;
    return `rgba(${Math.round(s * 200)}, ${Math.round(200 - s * 40)}, ${Math.round(100 - s * 80)}, ${0.4 + s * 0.2})`;
  }
  if (t < 0.75) {
    const s = (t - 0.5) / 0.25;
    return `rgba(${Math.round(200 + s * 55)}, ${Math.round(160 - s * 100)}, ${Math.round(20 - s * 20)}, ${0.6 + s * 0.15})`;
  }
  const s = (t - 0.75) / 0.25;
  return `rgba(255, ${Math.round(60 - s * 60)}, 0, ${0.75 + s * 0.25})`;
}

function densityColor(intensity: number): string {
  const t = Math.max(0, Math.min(1, intensity));
  const r = Math.round(16 + t * 52);
  const g = Math.round(185 + t * 70);
  const b = Math.round(129 - t * 50);
  return `rgba(${r}, ${g}, ${b}, ${0.05 + t * 0.7})`;
}

// ── Extract all frame-level position/speed data for one aggregated player ─
function extractPlayerFrameData(
  frames: any[],
  seedId: number,
  mergeMap: Record<number, number>,
  fps: number = 25
): {
  positions: { x: number; y: number; frame: number; time: number }[];
  speeds: { speed: number; frame: number; time: number }[];
  activityBins: { time: number; count: number }[];
} {
  // Build a set of all tracker IDs that map to this seed
  const trackerIds = new Set<number>();
  trackerIds.add(seedId);
  for (const [tid, sid] of Object.entries(mergeMap)) {
    if (Number(sid) === seedId) trackerIds.add(Number(tid));
  }

  const positions: { x: number; y: number; frame: number; time: number }[] = [];
  const speeds: { speed: number; frame: number; time: number }[] = [];

  for (const frame of frames || []) {
    const fIdx = frame.frame_idx;
    const time = fIdx / fps;
    for (const p of frame.players || []) {
      if (!trackerIds.has(p.id)) continue;
      if (p.x_pitch != null && p.y_pitch != null) {
        positions.push({ x: p.x_pitch, y: p.y_pitch, frame: fIdx, time });
      }
      if (p.speed_kmh != null && p.speed_kmh >= 0) {
        speeds.push({ speed: Math.min(p.speed_kmh, 50), frame: fIdx, time });
      }
    }
  }

  // Activity bins: group frames into 30-second buckets
  const binSize = 30; // seconds
  const activityMap: Record<number, number> = {};
  for (const pos of positions) {
    const bin = Math.floor(pos.time / binSize);
    activityMap[bin] = (activityMap[bin] || 0) + 1;
  }
  const activityBins = Object.entries(activityMap)
    .map(([bin, count]) => ({ time: Number(bin) * binSize, count: count as number }))
    .sort((a, b) => a.time - b.time);

  return { positions, speeds, activityBins };
}

// ── Match Rating heuristic (0–10 scale) ──────────────────────────────────
function computeMatchRating(player: any, allPlayers: any[]): number {
  // Weighted composite: threat score (40%), distance (20%), sprints (15%), mins (15%), speed (10%)
  const maxDist = Math.max(...allPlayers.map(p => p.distanceCovered || 0), 1);
  const maxSprints = Math.max(...allPlayers.map(p => p.sprintCount || 0), 1);
  const maxMins = Math.max(...allPlayers.map(p => p.minutesPlayed || 0), 1);

  const threatNorm = (player.threatScore || 0) / 100;
  const distNorm = (player.distanceCovered || 0) / maxDist;
  const sprintNorm = (player.sprintCount || 0) / maxSprints;
  const minsNorm = (player.minutesPlayed || 0) / maxMins;
  const speedNorm = Math.min((player.avgSpeed || 0) / 15, 1);

  const raw = threatNorm * 0.4 + distNorm * 0.2 + sprintNorm * 0.15 + minsNorm * 0.15 + speedNorm * 0.1;
  return Math.round(Math.max(4, Math.min(10, 4 + raw * 6)) * 10) / 10;
}

// ── Heatmap Canvas Component ─────────────────────────────────────────────
function PitchHeatmap({ positions, title }: { positions: { x: number; y: number }[]; title: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const CANVAS_W = 525;
  const CANVAS_H = 340;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear
    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);

    // Draw pitch background
    ctx.fillStyle = '#0d1f0d';
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

    // Pitch markings
    const scaleX = CANVAS_W / PITCH_LENGTH;
    const scaleY = CANVAS_H / PITCH_WIDTH;

    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.lineWidth = 1;

    // Pitch outline
    ctx.strokeRect(4, 4, CANVAS_W - 8, CANVAS_H - 8);

    // Center line
    ctx.beginPath();
    ctx.moveTo(CANVAS_W / 2, 4);
    ctx.lineTo(CANVAS_W / 2, CANVAS_H - 4);
    ctx.stroke();

    // Center circle
    ctx.beginPath();
    ctx.arc(CANVAS_W / 2, CANVAS_H / 2, 9.15 * scaleX, 0, Math.PI * 2);
    ctx.stroke();

    // Penalty areas
    const penW = 16.5 * scaleX;
    const penH = 40.3 * scaleY;
    const penY = (CANVAS_H - penH) / 2;
    ctx.strokeRect(4, penY, penW, penH); // Left
    ctx.strokeRect(CANVAS_W - 4 - penW, penY, penW, penH); // Right

    // Goal areas
    const goalW = 5.5 * scaleX;
    const goalH = 18.3 * scaleY;
    const goalY = (CANVAS_H - goalH) / 2;
    ctx.strokeRect(4, goalY, goalW, goalH);
    ctx.strokeRect(CANVAS_W - 4 - goalW, goalY, goalW, goalH);

    if (positions.length === 0) return;

    // Build heatmap grid
    const gridW = 42;
    const gridH = 27;
    const grid: number[][] = Array.from({ length: gridH }, () => new Array(gridW).fill(0));

    for (const pos of positions) {
      const gx = Math.floor(Math.max(0, Math.min(PITCH_LENGTH - 0.01, pos.x)) / PITCH_LENGTH * gridW);
      const gy = Math.floor(Math.max(0, Math.min(PITCH_WIDTH - 0.01, pos.y)) / PITCH_WIDTH * gridH);
      grid[gy][gx] += 1;
    }

    // Gaussian blur (3x3 kernel)
    const blurred: number[][] = Array.from({ length: gridH }, () => new Array(gridW).fill(0));
    const kernel = [
      [0.0625, 0.125, 0.0625],
      [0.125, 0.25, 0.125],
      [0.0625, 0.125, 0.0625]
    ];
    for (let y = 0; y < gridH; y++) {
      for (let x = 0; x < gridW; x++) {
        let sum = 0;
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            const ny = Math.max(0, Math.min(gridH - 1, y + ky));
            const nx = Math.max(0, Math.min(gridW - 1, x + kx));
            sum += grid[ny][nx] * kernel[ky + 1][kx + 1];
          }
        }
        blurred[y][x] = sum;
      }
    }

    // Find max for normalization
    let maxVal = 0;
    for (const row of blurred) for (const v of row) if (v > maxVal) maxVal = v;
    if (maxVal === 0) return;

    // Render
    const cellW = CANVAS_W / gridW;
    const cellH = CANVAS_H / gridH;
    for (let y = 0; y < gridH; y++) {
      for (let x = 0; x < gridW; x++) {
        const intensity = blurred[y][x] / maxVal;
        if (intensity > 0.02) {
          ctx.fillStyle = heatColor(intensity);
          ctx.fillRect(x * cellW, y * cellH, cellW + 0.5, cellH + 0.5);
        }
      }
    }
  }, [positions]);

  return (
    <div className="flex flex-col gap-2">
      <div className="text-[9px] font-bold font-mono text-gray-500 uppercase tracking-widest flex items-center gap-1.5">
        <Flame size={12} className="text-orange-500" />
        {title}
      </div>
      <div className="bg-[#0d1f0d] rounded-xl overflow-hidden border border-gray-900/60 shadow-inner">
        <canvas
          ref={canvasRef}
          width={CANVAS_W}
          height={CANVAS_H}
          className="w-full h-auto"
          style={{ imageRendering: 'auto' }}
        />
      </div>
      <div className="flex items-center justify-between text-[9px] font-mono text-gray-600 px-1">
        <span>Low presence</span>
        <div className="flex gap-0.5">
          {[0.1, 0.3, 0.5, 0.7, 0.9].map(v => (
            <div key={v} className="w-6 h-2 rounded-sm" style={{ backgroundColor: heatColor(v) }} />
          ))}
        </div>
        <span>High presence</span>
      </div>
    </div>
  );
}

// ── Position Density Map Component ───────────────────────────────────────
function PositionDensityMap({ positions, title }: { positions: { x: number; y: number }[]; title: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const CANVAS_W = 525;
  const CANVAS_H = 340;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);

    // Background
    ctx.fillStyle = '#050a05';
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

    // Pitch markings (subtle)
    const scaleX = CANVAS_W / PITCH_LENGTH;
    const scaleY = CANVAS_H / PITCH_WIDTH;
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 0.5;
    ctx.strokeRect(4, 4, CANVAS_W - 8, CANVAS_H - 8);
    ctx.beginPath();
    ctx.moveTo(CANVAS_W / 2, 4);
    ctx.lineTo(CANVAS_W / 2, CANVAS_H - 4);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(CANVAS_W / 2, CANVAS_H / 2, 9.15 * scaleX, 0, Math.PI * 2);
    ctx.stroke();

    if (positions.length === 0) return;

    // Draw density as overlapping translucent dots
    // Down-sample if too many positions
    const maxDots = 2000;
    const step = Math.max(1, Math.floor(positions.length / maxDots));
    const sampled = positions.filter((_, i) => i % step === 0);

    for (const pos of sampled) {
      const px = (pos.x / PITCH_LENGTH) * CANVAS_W;
      const py = (pos.y / PITCH_WIDTH) * CANVAS_H;

      const gradient = ctx.createRadialGradient(px, py, 0, px, py, 12);
      gradient.addColorStop(0, 'rgba(16, 185, 129, 0.15)');
      gradient.addColorStop(0.5, 'rgba(16, 185, 129, 0.05)');
      gradient.addColorStop(1, 'rgba(16, 185, 129, 0)');

      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(px, py, 12, 0, Math.PI * 2);
      ctx.fill();
    }

    // Draw average position marker
    const avgX = positions.reduce((s, p) => s + p.x, 0) / positions.length;
    const avgY = positions.reduce((s, p) => s + p.y, 0) / positions.length;
    const avgPx = (avgX / PITCH_LENGTH) * CANVAS_W;
    const avgPy = (avgY / PITCH_WIDTH) * CANVAS_H;

    // Crosshair
    ctx.strokeStyle = 'rgba(251, 191, 36, 0.7)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(avgPx - 15, avgPy);
    ctx.lineTo(avgPx + 15, avgPy);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(avgPx, avgPy - 15);
    ctx.lineTo(avgPx, avgPy + 15);
    ctx.stroke();
    ctx.setLineDash([]);

    // Average position dot
    ctx.fillStyle = 'rgba(251, 191, 36, 0.9)';
    ctx.beginPath();
    ctx.arc(avgPx, avgPy, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = 'rgba(251, 191, 36, 0.5)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(avgPx, avgPy, 8, 0, Math.PI * 2);
    ctx.stroke();

    // Label
    ctx.fillStyle = 'rgba(251, 191, 36, 0.8)';
    ctx.font = '9px monospace';
    ctx.fillText(`AVG (${avgX.toFixed(0)}m, ${avgY.toFixed(0)}m)`, avgPx + 12, avgPy - 4);
  }, [positions]);

  return (
    <div className="flex flex-col gap-2">
      <div className="text-[9px] font-bold font-mono text-gray-500 uppercase tracking-widest flex items-center gap-1.5">
        <MapPin size={12} className="text-emerald-500" />
        {title}
      </div>
      <div className="bg-[#050a05] rounded-xl overflow-hidden border border-gray-900/60 shadow-inner">
        <canvas
          ref={canvasRef}
          width={CANVAS_W}
          height={CANVAS_H}
          className="w-full h-auto"
          style={{ imageRendering: 'auto' }}
        />
      </div>
    </div>
  );
}

// ── Speed Timeline SVG Component ─────────────────────────────────────────
function SpeedTimeline({ speeds, title }: { speeds: { speed: number; time: number }[]; title: string }) {
  const SVG_W = 600;
  const SVG_H = 140;
  const PAD_L = 40;
  const PAD_R = 12;
  const PAD_T = 12;
  const PAD_B = 28;

  const chartW = SVG_W - PAD_L - PAD_R;
  const chartH = SVG_H - PAD_T - PAD_B;

  const data = useMemo(() => {
    if (speeds.length === 0) return { bins: [], maxSpeed: 0, maxTime: 0 };
    const maxTime = Math.max(...speeds.map(s => s.time));
    const maxSpeed = Math.max(...speeds.map(s => s.speed), 1);
    const binSize = 5; // 5-second bins
    const numBins = Math.ceil(maxTime / binSize) + 1;
    const bins: { time: number; avgSpeed: number; maxSpeed: number }[] = [];

    for (let i = 0; i < numBins; i++) {
      const t0 = i * binSize;
      const t1 = t0 + binSize;
      const inBin = speeds.filter(s => s.time >= t0 && s.time < t1);
      if (inBin.length > 0) {
        bins.push({
          time: t0 + binSize / 2,
          avgSpeed: inBin.reduce((s, v) => s + v.speed, 0) / inBin.length,
          maxSpeed: Math.max(...inBin.map(v => v.speed)),
        });
      }
    }
    return { bins, maxSpeed: Math.min(maxSpeed, 40), maxTime };
  }, [speeds]);

  if (data.bins.length === 0) {
    return (
      <div className="flex flex-col gap-2">
        <div className="text-[9px] font-bold font-mono text-gray-500 uppercase tracking-widest flex items-center gap-1.5">
          <Gauge size={12} className="text-cyan-500" />
          {title}
        </div>
        <div className="bg-black/30 border border-gray-900/50 rounded-xl p-8 text-center text-gray-600 text-xs font-mono">
          No speed data available.
        </div>
      </div>
    );
  }

  const toX = (t: number) => PAD_L + (t / data.maxTime) * chartW;
  const toY = (s: number) => PAD_T + chartH - (Math.min(s, data.maxSpeed) / data.maxSpeed) * chartH;

  // Build SVG path for avg speed line
  const avgPath = data.bins.map((b, i) => `${i === 0 ? 'M' : 'L'} ${toX(b.time).toFixed(1)} ${toY(b.avgSpeed).toFixed(1)}`).join(' ');
  // Area fill path
  const areaPath = `${avgPath} L ${toX(data.bins[data.bins.length - 1].time).toFixed(1)} ${(PAD_T + chartH).toFixed(1)} L ${toX(data.bins[0].time).toFixed(1)} ${(PAD_T + chartH).toFixed(1)} Z`;

  // Sprint zones
  const sprintZones: { x1: number; x2: number }[] = [];
  let inSprint = false;
  let sprintStart = 0;
  for (const b of data.bins) {
    if (b.maxSpeed >= 24 && !inSprint) {
      inSprint = true;
      sprintStart = b.time;
    } else if (b.maxSpeed < 19 && inSprint) {
      inSprint = false;
      sprintZones.push({ x1: toX(sprintStart), x2: toX(b.time) });
    }
  }
  if (inSprint) sprintZones.push({ x1: toX(sprintStart), x2: toX(data.bins[data.bins.length - 1].time) });

  // Y-axis labels
  const yTicks = [0, 10, 20, 30, 40].filter(v => v <= data.maxSpeed);

  return (
    <div className="flex flex-col gap-2">
      <div className="text-[9px] font-bold font-mono text-gray-500 uppercase tracking-widest flex items-center gap-1.5">
        <Gauge size={12} className="text-cyan-500" />
        {title}
      </div>
      <div className="bg-black/30 border border-gray-900/50 rounded-xl p-3 shadow-inner overflow-hidden">
        <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`} className="w-full h-auto">
          {/* Grid lines */}
          {yTicks.map(v => (
            <g key={v}>
              <line x1={PAD_L} y1={toY(v)} x2={SVG_W - PAD_R} y2={toY(v)} stroke="rgba(255,255,255,0.05)" strokeWidth={0.5} />
              <text x={PAD_L - 4} y={toY(v) + 3} fill="rgba(255,255,255,0.25)" fontSize="8" fontFamily="monospace" textAnchor="end">
                {v}
              </text>
            </g>
          ))}

          {/* Sprint zones */}
          {sprintZones.map((z, i) => (
            <rect key={i} x={z.x1} y={PAD_T} width={z.x2 - z.x1} height={chartH} fill="rgba(251,191,36,0.06)" />
          ))}

          {/* Area fill */}
          <path d={areaPath} fill="url(#speedGradient)" />

          {/* Line */}
          <path d={avgPath} fill="none" stroke="rgba(6,182,212,0.8)" strokeWidth={1.5} />

          {/* Max speed dots */}
          {data.bins.filter(b => b.maxSpeed >= 24).map((b, i) => (
            <circle key={i} cx={toX(b.time)} cy={toY(b.maxSpeed)} r={2.5} fill="rgba(251,191,36,0.8)" />
          ))}

          {/* Gradient definition */}
          <defs>
            <linearGradient id="speedGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="rgba(6,182,212,0.25)" />
              <stop offset="100%" stopColor="rgba(6,182,212,0)" />
            </linearGradient>
          </defs>

          {/* X axis */}
          <line x1={PAD_L} y1={PAD_T + chartH} x2={SVG_W - PAD_R} y2={PAD_T + chartH} stroke="rgba(255,255,255,0.08)" strokeWidth={0.5} />

          {/* Time labels */}
          {[0, 0.25, 0.5, 0.75, 1].map(frac => {
            const t = frac * data.maxTime;
            const min = Math.floor(t / 60);
            const sec = Math.floor(t % 60);
            return (
              <text key={frac} x={toX(t)} y={SVG_H - 4} fill="rgba(255,255,255,0.25)" fontSize="8" fontFamily="monospace" textAnchor="middle">
                {min}:{sec.toString().padStart(2, '0')}
              </text>
            );
          })}

          {/* Y axis label */}
          <text x={4} y={PAD_T + chartH / 2} fill="rgba(255,255,255,0.2)" fontSize="7" fontFamily="monospace" textAnchor="middle" transform={`rotate(-90, 8, ${PAD_T + chartH / 2})`}>
            km/h
          </text>
        </svg>
      </div>
      <div className="flex items-center gap-4 text-[9px] font-mono text-gray-600 px-1">
        <span className="flex items-center gap-1">
          <div className="w-3 h-0.5 bg-cyan-500 rounded" /> Avg Speed
        </span>
        <span className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-full bg-amber-500/80" /> Sprint Peak
        </span>
        <span className="flex items-center gap-1">
          <div className="w-4 h-2 bg-amber-500/10 rounded" /> Sprint Zone
        </span>
      </div>
    </div>
  );
}

// ── Activity Timeline Component ──────────────────────────────────────────
function ActivityTimeline({ activityBins, title, minutesPlayed }: {
  activityBins: { time: number; count: number }[];
  title: string;
  minutesPlayed: number;
}) {
  if (activityBins.length === 0) {
    return (
      <div className="flex flex-col gap-2">
        <div className="text-[9px] font-bold font-mono text-gray-500 uppercase tracking-widest flex items-center gap-1.5">
          <Activity size={12} className="text-violet-500" />
          {title}
        </div>
        <div className="bg-black/30 border border-gray-900/50 rounded-xl p-8 text-center text-gray-600 text-xs font-mono">
          No activity data available.
        </div>
      </div>
    );
  }

  const maxCount = Math.max(...activityBins.map(b => b.count), 1);
  const maxTime = Math.max(...activityBins.map(b => b.time + 30));

  return (
    <div className="flex flex-col gap-2">
      <div className="text-[9px] font-bold font-mono text-gray-500 uppercase tracking-widest flex items-center gap-1.5">
        <Activity size={12} className="text-violet-500" />
        {title}
      </div>
      <div className="bg-black/30 border border-gray-900/50 rounded-xl p-4 shadow-inner">
        <div className="flex items-end gap-[2px] h-20">
          {activityBins.map((bin, i) => {
            const height = (bin.count / maxCount) * 100;
            const isHigh = bin.count >= maxCount * 0.7;
            return (
              <div key={i} className="flex-1 relative group" title={`${Math.floor(bin.time / 60)}:${(Math.floor(bin.time % 60)).toString().padStart(2, '0')} — ${bin.count} frames`}>
                <div
                  className={`w-full rounded-t-sm transition-all duration-200 ${isHigh ? 'bg-violet-500' : 'bg-violet-500/40'} group-hover:bg-violet-400`}
                  style={{ height: `${Math.max(2, height)}%` }}
                />
              </div>
            );
          })}
        </div>
        <div className="flex justify-between text-[8px] font-mono text-gray-600 mt-2">
          <span>0:00</span>
          <span>{Math.floor(maxTime / 120)}:{(Math.floor((maxTime / 2) % 60)).toString().padStart(2, '0')}</span>
          <span>{Math.floor(maxTime / 60)}:{(Math.floor(maxTime % 60)).toString().padStart(2, '0')}</span>
        </div>
      </div>
      <div className="flex items-center gap-4 text-[9px] font-mono text-gray-600 px-1">
        <span className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-sm bg-violet-500" /> High activity
        </span>
        <span className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-sm bg-violet-500/40" /> Normal
        </span>
        <span>30s bins • {activityBins.length} segments</span>
      </div>
    </div>
  );
}

// ── Stat Card ────────────────────────────────────────────────────────────
function StatCard({ icon, label, value, subValue, barPct, barColor = 'bg-emerald-500', colSpan }: {
  icon: React.ReactNode;
  label: string;
  value: string;
  subValue?: string;
  barPct: number;
  barColor?: string;
  colSpan?: string;
}) {
  return (
    <div className={`bg-black/30 border border-gray-900/50 rounded-xl p-4 flex flex-col gap-1.5 shadow-inner ${colSpan || ''}`}>
      <span className="text-gray-600 text-[10px] font-mono uppercase tracking-tight flex items-center gap-1">
        {icon} {label}
      </span>
      <span className="text-gray-200 font-mono font-bold text-lg">{value}</span>
      {subValue && <span className="text-gray-600 text-[9px] font-mono">{subValue}</span>}
      <div className="w-full h-1 bg-gray-950 rounded-full overflow-hidden mt-1">
        <div className={`h-full ${barColor} transition-all duration-500`} style={{ width: `${Math.min(100, barPct)}%` }} />
      </div>
    </div>
  );
}

// ── Main Component ───────────────────────────────────────────────────────
export function PlayerDetailView({
  player,
  job,
  mergeMap,
  allPlayers,
  getPlayerLabel,
  getTeamLabel,
  onBack,
  onAskAI,
  onPlayClip,
  playerClips,
  loadingClips,
}: PlayerDetailViewProps) {
  const [activeTab, setActiveTab] = useState<'overview' | 'heatmap' | 'speed' | 'activity'>('overview');

  // Extract frame-level data for visualizations
  const frameData = useMemo(() => {
    if (!job?.tracking?.frames) return { positions: [], speeds: [], activityBins: [] };
    return extractPlayerFrameData(job.tracking.frames, player.id, mergeMap);
  }, [job?.tracking?.frames, player.id, mergeMap]);

  const matchRating = useMemo(() => computeMatchRating(player, allPlayers), [player, allPlayers]);

  const threatColor = (score: number) => {
    if (score >= 70) return { text: "text-red-400 font-bold", border: "border-red-500/30", bg: "bg-red-500/10", bar: "bg-red-500" };
    if (score >= 40) return { text: "text-amber-400 font-bold", border: "border-amber-500/30", bg: "bg-amber-500/10", bar: "bg-amber-500" };
    return { text: "text-emerald-400 font-medium", border: "border-emerald-500/30", bg: "bg-emerald-500/10", bar: "bg-emerald-500" };
  };

  const ratingColor = matchRating >= 8 ? 'text-emerald-400' : matchRating >= 6.5 ? 'text-amber-400' : 'text-gray-400';

  const tabs = [
    { id: 'overview' as const, label: 'Overview', icon: <BarChart2 size={12} /> },
    { id: 'heatmap' as const, label: 'Heatmap & Density', icon: <MapPin size={12} /> },
    { id: 'speed' as const, label: 'Speed Profile', icon: <Gauge size={12} /> },
    { id: 'activity' as const, label: 'Activity', icon: <Activity size={12} /> },
  ];

  return (
    <div className="flex flex-col h-full animate-fade-in">
      {/* Header Controls */}
      <div className="flex justify-between items-center mb-6">
        <button
          id="back-to-list"
          onClick={onBack}
          className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-emerald-400 font-mono transition-colors"
        >
          <ChevronLeft size={16} /> Back to Team Roster
        </button>
        <button
          id={`ask-ai-${player.id}`}
          onClick={() => onAskAI(`Explain how we should defend against ${getPlayerLabel(player.id)} (P${player.id}) given their threat score of ${player.threatScore.toFixed(0)}/100 and physical profile.`)}
          className="bg-emerald-500 hover:bg-emerald-400 text-black font-bold px-4 py-2 rounded-xl text-xs transition-all flex items-center gap-1.5 shadow-lg shadow-emerald-950/20"
        >
          <Bot size={14} /> Ask AI about this Player
        </button>
      </div>

      {/* Hero Profile Card */}
      <div className="bg-[#111a12]/10 border border-gray-900 rounded-2xl p-6 mb-6 flex flex-col md:flex-row md:items-center justify-between gap-6 shadow-xl relative overflow-hidden">
        <div className="absolute right-0 top-0 translate-x-12 -translate-y-12 h-48 w-48 rounded-full bg-emerald-500/5 blur-3xl" />
        <div className="flex items-center gap-4 relative">
          <div className="h-16 w-16 rounded-2xl bg-gray-900 border border-gray-800 flex items-center justify-center text-emerald-500 shadow-inner text-xl font-mono font-bold">
            {player.jerseyNumber !== undefined ? `#${player.jerseyNumber}` : <User size={32} />}
          </div>
          <div>
            <h2 className="text-xl font-bold text-gray-100">{getPlayerLabel(player.id)}</h2>
            <div className="flex flex-wrap items-center gap-3 mt-1.5 text-xs">
              <span className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded text-[10px] font-mono ${
                player.teamId === 'team_0'
                  ? 'bg-red-500/10 text-red-400 border border-red-500/20'
                  : player.teamId === 'team_1'
                  ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                  : 'bg-gray-800 text-gray-400'
              }`}>
                {getTeamLabel(player.teamId)}
              </span>
              <span className="text-gray-600 font-mono">|</span>
              <span className="text-gray-400 font-mono flex items-center gap-1">
                <Compass size={12} className="text-emerald-500/70" /> {player.position}
              </span>
              <span className="text-gray-600 font-mono">|</span>
              <span className="text-gray-400 font-mono flex items-center gap-1">
                <Clock size={12} className="text-emerald-500/70" /> {player.minutesPlayed} min
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {/* Match Rating */}
          <div className="flex items-center gap-3 bg-black/40 border border-gray-900 px-5 py-3 rounded-2xl">
            <div className="flex flex-col gap-0.5 items-center">
              <span className="text-gray-600 uppercase text-[9px] font-mono tracking-wider">Match Rating</span>
              <span className={`text-2xl font-mono font-black ${ratingColor}`}>{matchRating.toFixed(1)}</span>
            </div>
          </div>

          {/* Threat Rank */}
          {player.threatScore > 0 && (
            <div className="flex items-center gap-3 bg-black/40 border border-gray-900 px-5 py-3 rounded-2xl">
              <div className="flex flex-col gap-0.5">
                <span className="text-gray-600 uppercase text-[9px] font-mono tracking-wider">Threat Rank</span>
                <span className="text-gray-200 font-mono font-bold text-sm flex items-center gap-1">
                  <Award size={14} className="text-amber-500" /> #{player.threatRank}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Tab Bar */}
      <div className="flex items-center gap-1 mb-6 bg-black/20 border border-gray-900/60 rounded-xl p-1 w-fit">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-mono font-medium transition-all ${
              activeTab === tab.id
                ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/20'
                : 'text-gray-500 hover:text-gray-300 hover:bg-white/[0.02]'
            }`}
          >
            {tab.icon} {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="flex-1 min-h-0 overflow-y-auto custom-scrollbar">
        {activeTab === 'overview' && (
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 items-start">
            {/* Physical Performance */}
            <div className="xl:col-span-1 bg-[#111a12]/10 border border-gray-900 rounded-2xl p-6 shadow-xl flex flex-col gap-5">
              <div className="border-b border-gray-900/60 pb-3">
                <h3 className="text-xs font-bold text-emerald-400 font-mono uppercase tracking-widest flex items-center gap-2">
                  <TrendingUp size={14} /> Physical Performance
                </h3>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <StatCard
                  icon={<Clock size={10} className="text-emerald-500" />}
                  label="Minutes Played"
                  value={`${player.minutesPlayed} min`}
                  barPct={(player.minutesPlayed / 90) * 100}
                  colSpan="col-span-2"
                />
                <StatCard
                  icon={<TrendingUp size={10} className="text-emerald-500" />}
                  label="Distance Covered"
                  value={`${player.distanceCovered.toFixed(1)} m`}
                  barPct={(player.distanceCovered / 300) * 100}
                />
                <StatCard
                  icon={<Zap size={10} className="text-amber-500" />}
                  label="Sprint Count"
                  value={`${player.sprintCount} sprints`}
                  barPct={(player.sprintCount / 10) * 100}
                  barColor="bg-amber-500"
                />
                <StatCard
                  icon={<Gauge size={10} className="text-cyan-500" />}
                  label="Average Speed"
                  value={`${player.avgSpeed.toFixed(1)} km/h`}
                  barPct={(player.avgSpeed / 15) * 100}
                  barColor="bg-cyan-500"
                />
                <StatCard
                  icon={<Gauge size={10} className="text-red-400" />}
                  label="Maximum Speed"
                  value={`${player.maxSpeed.toFixed(1)} km/h`}
                  barPct={(player.maxSpeed / 38) * 100}
                  barColor="bg-red-400"
                />
              </div>
            </div>

            {/* Tactical Threat Profile */}
            <div className="xl:col-span-2 bg-[#111a12]/10 border border-gray-900 rounded-2xl p-6 shadow-xl flex flex-col gap-5">
              <div className="flex justify-between items-center border-b border-gray-900/60 pb-3">
                <h3 className="text-xs font-bold text-emerald-400 font-mono uppercase tracking-widest flex items-center gap-2">
                  <BarChart2 size={14} /> Tactical Threat Profile
                </h3>
                <span className="text-[10px] font-mono text-gray-600 uppercase">Normalized Threat Metric Sourcing</span>
              </div>

              <div className="flex flex-col md:flex-row gap-6 items-start md:items-center">
                <div className="flex items-center gap-4 bg-black/30 border border-gray-900/40 p-4 rounded-xl shadow-inner w-full md:w-56 shrink-0">
                  <div className="relative h-14 w-14 rounded-full border border-gray-800 flex items-center justify-center bg-gray-950 shrink-0">
                    <div className="text-[10px] font-mono font-bold text-gray-400">Score</div>
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-[11px] font-mono text-gray-500 uppercase">Normalized Rating</div>
                    <div className="text-base font-mono font-bold text-emerald-400">{player.threatScore.toFixed(0)}/100</div>
                    <div className="w-full h-1 bg-gray-950 rounded-full overflow-hidden mt-1">
                      <div className="h-full bg-emerald-500" style={{ width: `${player.threatScore}%` }} />
                    </div>
                  </div>
                </div>

                {player.primaryThreatTypes.length > 0 && (
                  <div className="flex-1 bg-black/30 border border-gray-900/40 p-4 rounded-xl shadow-inner w-full">
                    <div className="text-[9px] font-mono font-bold text-gray-500 uppercase tracking-widest flex items-center gap-1.5 mb-1.5">
                      <ShieldAlert size={12} className="text-emerald-500" /> Dominant Threat Mode
                    </div>
                    <div className="text-xs font-mono font-bold text-emerald-400 bg-emerald-500/5 border border-emerald-500/10 px-3 py-1.5 rounded-lg inline-block">
                      {EVENT_NAME_LOOKUP[player.primaryThreatTypes[0]] || player.primaryThreatTypes[0]}
                    </div>
                  </div>
                )}
              </div>

              <div className="space-y-2">
                <div className="text-[9px] font-bold font-mono text-gray-500 uppercase tracking-widest flex items-center gap-1.5">
                  <ShieldAlert size={12} className="text-emerald-500" /> Tactical Intelligence Verdict
                </div>
                <div className="bg-black/35 rounded-lg p-4 text-gray-400 font-sans leading-relaxed whitespace-pre-line border border-gray-900/40 text-xs">
                  {player.explanation}
                </div>
              </div>

              {Object.keys(player.eventCounts).length > 0 && (
                <div className="space-y-2">
                  <div className="text-[9px] font-bold font-mono text-gray-500 uppercase tracking-widest flex items-center gap-1.5">
                    <BarChart2 size={12} className="text-emerald-500" /> Ontology Triggers Breakdown
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(player.eventCounts).map(([code, count]) => (
                      <div key={code} className="px-3 py-1.5 bg-black/40 border border-gray-900 hover:border-emerald-500/20 rounded-lg flex items-center gap-2">
                        <span className="font-mono text-[10px] text-gray-400">{EVENT_NAME_LOOKUP[code] || code}</span>
                        <span className="font-mono font-bold text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded text-[10px]">{count as number}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'heatmap' && (
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            <div className="bg-[#111a12]/10 border border-gray-900 rounded-2xl p-6 shadow-xl">
              <PitchHeatmap positions={frameData.positions} title="Pitch Heatmap" />
            </div>
            <div className="bg-[#111a12]/10 border border-gray-900 rounded-2xl p-6 shadow-xl">
              <PositionDensityMap positions={frameData.positions} title="Position Density Map" />
            </div>
          </div>
        )}

        {activeTab === 'speed' && (
          <div className="bg-[#111a12]/10 border border-gray-900 rounded-2xl p-6 shadow-xl">
            <SpeedTimeline speeds={frameData.speeds} title="Speed Timeline" />
          </div>
        )}

        {activeTab === 'activity' && (
          <div className="bg-[#111a12]/10 border border-gray-900 rounded-2xl p-6 shadow-xl">
            <ActivityTimeline
              activityBins={frameData.activityBins}
              title="Activity Timeline"
              minutesPlayed={player.minutesPlayed}
            />
          </div>
        )}

        {/* Video Clips Section (always visible at bottom) */}
        <div className="bg-[#111a12]/10 border border-gray-900 rounded-2xl p-6 shadow-xl mt-6">
          <div className="border-b border-gray-900/60 pb-3 mb-4">
            <h3 className="text-xs font-bold text-emerald-400 font-mono uppercase tracking-widest flex items-center gap-2">
              <Film size={14} /> Visual Evidence (AI Sourced Video Clips)
            </h3>
          </div>

          {loadingClips[`player-${player.id}`] ? (
            <div className="flex items-center gap-2 text-gray-500 italic py-4">
              <Loader2 size={14} className="animate-spin text-emerald-500" />
              <span className="text-xs font-mono">Retrieving matching clips from index...</span>
            </div>
          ) : !playerClips[`player-${player.id}`] || playerClips[`player-${player.id}`].length === 0 ? (
            <div className="text-gray-600 italic font-mono py-2 text-xs">
              No supported video clips registered in matches index for this player.
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {playerClips[`player-${player.id}`].map((clip: any, clipIdx: number) => (
                <button
                  key={clipIdx}
                  id={`play-clip-${clipIdx}`}
                  onClick={() => onPlayClip(clip.start_time_s)}
                  className="w-full text-left p-4 rounded-xl bg-black/40 border border-gray-900 hover:border-emerald-500/30 hover:bg-[#111a12]/20 transition-all flex items-center justify-between group"
                >
                  <div className="flex-1 min-w-0 pr-2">
                    <div className="text-xs font-bold text-gray-200 truncate group-hover:text-emerald-400 transition-colors">
                      {clip.label || "Key Moment"}
                    </div>
                    <div className="flex items-center gap-3 mt-2 text-[10px] font-mono text-gray-600">
                      <span className="flex items-center gap-1">
                        <Clock size={10} />
                        {Math.floor(clip.start_time_s / 60)}:{(Math.floor(clip.start_time_s % 60)).toString().padStart(2, '0')}
                      </span>
                      <span className="bg-emerald-500/5 text-emerald-500/50 px-1.5 py-0.25 rounded">
                        {Math.round(clip.confidence_pct || clip.relevance_score * 100)}% Match
                      </span>
                    </div>
                  </div>
                  <div className="h-7 w-7 rounded-full bg-emerald-500/10 flex items-center justify-center text-emerald-500 group-hover:bg-emerald-500/20 group-hover:scale-105 transition-all shrink-0">
                    <Play size={10} fill="currentColor" />
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
