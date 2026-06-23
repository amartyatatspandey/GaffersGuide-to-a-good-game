"use client";
import React, { useState, useEffect, useMemo } from 'react';
import {
  Search, Calendar, Clock, Zap, Shield, TrendingUp,
  BarChart2, Play, Users, FileText, Loader2, Film,
  SlidersHorizontal, ChevronRight, Crosshair,
  AlertTriangle, Flame, Target, Trash2, Copy, Download, Award
} from 'lucide-react';
import { listMatches, getReport, deleteMatch } from '@/lib/api/jobs';
import {
  listLibrary,
  deleteFromLibrary,
  duplicateInLibrary,
  exportMatchAsJSON,
  upsertMatch,
  buildLibraryEntry,
  LibraryMatch
} from '@/lib/matchLibrary';

interface MatchesHubProps {
  onOpenMatch: (report: any) => void;
  onViewPlayers: (matchId: string) => void;
}

type SortOption = 'newest' | 'oldest' | 'tactical_power' | 'recent_analysis' | 'insights';
type StatusFilter = 'all' | 'completed' | 'processing' | 'pending' | 'error';

/**
 * Generates a sleek, high-fidelity tactical soccer pitch thumbnail
 * as a PNG data URL. Programmatic representation of telemetry field.
 */
function generatePitchThumbnail(title: string): string {
  if (typeof window === 'undefined') return '';
  try {
    const canvas = document.createElement('canvas');
    canvas.width = 300;
    canvas.height = 160;
    const ctx = canvas.getContext('2d');
    if (!ctx) return '';

    // Sleek dark background
    ctx.fillStyle = '#060a06';
    ctx.fillRect(0, 0, 300, 160);

    // Subtle pitch boundary and center markings
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.15)'; // Emerald line
    ctx.lineWidth = 1.5;

    // Pitch borders
    ctx.strokeRect(15, 10, 270, 140);

    // Center division line
    ctx.beginPath();
    ctx.moveTo(150, 10);
    ctx.lineTo(150, 150);
    ctx.stroke();

    // Center circle
    ctx.beginPath();
    ctx.arc(150, 80, 24, 0, Math.PI * 2);
    ctx.stroke();

    // Penalty area Left
    ctx.strokeRect(15, 35, 40, 90);
    // Penalty area Right
    ctx.strokeRect(245, 35, 40, 90);

    // Goal areas
    ctx.strokeRect(15, 55, 15, 50);
    ctx.strokeRect(270, 55, 15, 50);

    // Draw stylized tactical dots (formations)
    // Red Team - Left Side
    ctx.fillStyle = '#ef4444'; // Red
    const redPositions = [
      { x: 30, y: 80 }, // GK
      { x: 60, y: 35 }, { x: 60, y: 65 }, { x: 60, y: 95 }, { x: 60, y: 125 }, // Defenders
      { x: 100, y: 50 }, { x: 100, y: 110 }, { x: 120, y: 80 }, // Midfielders
      { x: 140, y: 40 }, { x: 140, y: 120 }, { x: 145, y: 80 } // Forwards
    ];
    redPositions.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
      ctx.fill();
      // Glow
      ctx.fillStyle = 'rgba(239, 68, 68, 0.2)';
      ctx.beginPath();
      ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#ef4444';
    });

    // Blue Team - Right Side
    ctx.fillStyle = '#06b6d4'; // Cyan/Blue
    const bluePositions = [
      { x: 270, y: 80 }, // GK
      { x: 240, y: 35 }, { x: 240, y: 65 }, { x: 240, y: 95 }, { x: 240, y: 125 }, // Defenders
      { x: 200, y: 50 }, { x: 200, y: 110 }, { x: 180, y: 80 }, // Midfielders
      { x: 160, y: 40 }, { x: 160, y: 120 }, { x: 155, y: 80 } // Forwards
    ];
    bluePositions.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
      ctx.fill();
      // Glow
      ctx.fillStyle = 'rgba(6, 182, 212, 0.2)';
      ctx.beginPath();
      ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#06b6d4';
    });

    // Draw a golden match ball
    ctx.fillStyle = '#f59e0b';
    ctx.beginPath();
    ctx.arc(142, 74, 2.5, 0, Math.PI * 2);
    ctx.fill();

    return canvas.toDataURL('image/png');
  } catch (e) {
    console.error("Failed to generate pitch thumbnail:", e);
    return '';
  }
}

export function MatchesHub({ onOpenMatch, onViewPlayers }: MatchesHubProps) {
  const [matches, setMatches] = useState<LibraryMatch[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<SortOption>('newest');
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all');
  const [competitionFilter, setCompetitionFilter] = useState<string>('All');
  
  const [openingMatch, setOpeningMatch] = useState<string | null>(null);
  const [actionInProgress, setActionInProgress] = useState<string | null>(null);
  const [hoveredCard, setHoveredCard] = useState<string | null>(null);

  useEffect(() => {
    loadMatches();
  }, []);

  async function loadMatches() {
    setLoading(true);
    try {
      // 1. Get backend matches
      let apiMatches: any[] = [];
      try {
        apiMatches = await listMatches('', 'newest');
      } catch (err) {
        console.warn("Failed to fetch backend matches:", err);
      }

      // 2. Get local library matches
      const localMatches = listLibrary();

      // 3. Merge matches
      const mergedMap = new Map<string, LibraryMatch>();

      // Load API matches first
      apiMatches.forEach(am => {
        const tpRed = am.tactical_power_red || 0;
        const tpBlue = am.tactical_power_blue || 0;
        const tpMax = Math.max(tpRed, tpBlue);

        const title = am.name || `Match ${am.id?.slice(0, 8)}`;
        const vsParts = title.split(/\s+vs\s+/i);
        const teamA = vsParts[0]?.trim() || 'Team A';
        const teamB = vsParts[1]?.trim() || 'Team B';

        const rawStatus = am.status || 'completed';
        let mappedStatus: 'completed' | 'processing' | 'pending' | 'error' = 'completed';
        if (rawStatus === 'processing') mappedStatus = 'processing';
        else if (rawStatus === 'pending') mappedStatus = 'pending';
        else if (rawStatus === 'failed' || rawStatus === 'error') mappedStatus = 'error';

        const libFormat: LibraryMatch = {
          id: am.id,
          title,
          teamA,
          teamB,
          competition: am.competition || 'Local Tournament',
          videoName: am.video_filename || '',
          duration: am.duration || '90:00',
          date: am.upload_date || new Date().toISOString(),
          analysisDate: am.analysis_date || new Date().toISOString(),
          status: mappedStatus,
          tacticalScore: Math.round(tpMax),
          tacticalPowerRed: tpRed,
          tacticalPowerBlue: tpBlue,
          compactness: Math.round(Math.max(am.compactness_red || 0, am.compactness_blue || 0)),
          transitionSpeed: Math.round(Math.max(am.transition_speed_red || 0, am.transition_speed_blue || 0)),
          insightCount: am.flaw_count || 0,
          thumbnail: null,
          coachAdvice: null,
          telemetrySummary: null
        };
        mergedMap.set(am.id, libFormat);
      });

      // Load local matches and overwrite backend copies if available
      localMatches.forEach(lm => {
        // Generate thumbnail if missing
        if (!lm.thumbnail) {
          lm.thumbnail = generatePitchThumbnail(lm.title);
          upsertMatch(lm);
        }
        mergedMap.set(lm.id, lm);
      });

      // Ensure every single match has a thumbnail
      const mergedList = Array.from(mergedMap.values()).map(m => {
        if (!m.thumbnail) {
          m.thumbnail = generatePitchThumbnail(m.title);
        }
        return m;
      });

      setMatches(mergedList);
    } catch (error) {
      console.error("Failed to load matches library:", error);
    } finally {
      setLoading(false);
    }
  }

  // Action Handlers
  async function handleOpenDashboard(match: LibraryMatch) {
    setOpeningMatch(match.id);
    try {
      if (match.coachAdvice) {
        // Prepare expected payload structure for handleOpenReport
        const reportPayload = {
          id: match.id,
          job_id: match.id,
          video_title: match.videoName || match.title,
          telemetry: match.telemetrySummary?.telemetry || match.coachAdvice?.telemetry || null,
          metadata: {
            saved_at: match.date
          },
          pipeline: match.coachAdvice?.pipeline || {
            duration: match.duration,
            team_0: {
              tactical_power: match.tacticalPowerRed,
              compactness: match.compactness,
              transition_speed: match.transitionSpeed
            },
            team_1: {
              tactical_power: match.tacticalPowerBlue
            }
          },
          advice_items: match.coachAdvice?.advice_items || []
        };
        onOpenMatch(reportPayload);
      } else {
        // Call backend API if we don't have it saved locally
        const fullReport = await getReport(match.id);
        onOpenMatch(fullReport);
      }
    } catch (error) {
      console.error("Failed to open match analysis:", error);
      alert("Failed to retrieve analysis report.");
    } finally {
      setOpeningMatch(null);
    }
  }

  async function handleDelete(matchId: string) {
    if (!window.confirm("Are you sure you want to delete this match analysis? This action is permanent.")) {
      return;
    }
    setActionInProgress(matchId);
    try {
      // 1. Delete from local storage
      deleteFromLibrary(matchId);

      // 2. Delete from backend if applicable
      try {
        await deleteMatch(matchId);
      } catch (err) {
        console.warn("Backend match deletion not supported or failed:", err);
      }

      await loadMatches();
    } catch (error) {
      console.error("Failed to delete match:", error);
      alert("Failed to delete the match.");
    } finally {
      setActionInProgress(null);
    }
  }

  async function handleDuplicate(matchId: string) {
    setActionInProgress(matchId);
    try {
      const copy = duplicateInLibrary(matchId);
      if (copy) {
        await loadMatches();
      } else {
        // If it's backend only, fetch report and then duplicate locally
        const fullReport = await getReport(matchId);
        const match = matches.find(m => m.id === matchId);
        if (match) {
          const entry = buildLibraryEntry(
            `${matchId}-copy-${Date.now()}`,
            `${match.title} (Copy)`,
            {
              generated_at: new Date().toISOString(),
              pipeline: fullReport.pipeline || {},
              advice_items: fullReport.advice_items || [],
              competition: match.competition || ''
            },
            fullReport.telemetry || null
          );
          entry.title = `${match.title} (Copy)`;
          upsertMatch(entry);
          await loadMatches();
        }
      }
    } catch (error) {
      console.error("Failed to duplicate match:", error);
      alert("Failed to duplicate the analysis.");
    } finally {
      setActionInProgress(null);
    }
  }

  async function handleExport(match: LibraryMatch) {
    try {
      if (match.coachAdvice) {
        exportMatchAsJSON(match);
      } else {
        // Fetch full report from backend first
        setActionInProgress(match.id);
        const fullReport = await getReport(match.id);
        const fullMatch = buildLibraryEntry(
          match.id,
          match.videoName || match.title,
          {
            generated_at: match.date,
            pipeline: fullReport.pipeline || {},
            advice_items: fullReport.advice_items || [],
            competition: match.competition || ''
          },
          fullReport.telemetry || null
        );
        exportMatchAsJSON(fullMatch);
      }
    } catch (err) {
      console.error("Failed to export match:", err);
      alert("Failed to export match analysis details.");
    } finally {
      setActionInProgress(null);
    }
  }

  // Extracted unique competitions for filtering
  const competitionsList = useMemo(() => {
    const set = new Set<string>();
    matches.forEach(m => {
      if (m.competition) set.add(m.competition);
    });
    return ['All', ...Array.from(set)];
  }, [matches]);

  // Compute status counts for filter chips
  const counts = useMemo(() => {
    const stats = { all: 0, completed: 0, processing: 0, pending: 0, error: 0 };
    matches.forEach(m => {
      stats.all++;
      const s = m.status;
      if (s === 'completed') stats.completed++;
      else if (s === 'processing') stats.processing++;
      else if (s === 'pending') stats.pending++;
      else if (s === 'error') stats.error++;
    });
    return stats;
  }, [matches]);

  // Filter and Sort Logic
  const filteredAndSorted = useMemo(() => {
    let filtered = matches;

    // Search query filter
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      filtered = filtered.filter(m =>
        m.title.toLowerCase().includes(q) ||
        m.teamA.toLowerCase().includes(q) ||
        m.teamB.toLowerCase().includes(q) ||
        m.competition.toLowerCase().includes(q) ||
        m.videoName.toLowerCase().includes(q)
      );
    }

    // Status filter
    if (statusFilter !== 'all') {
      filtered = filtered.filter(m => {
        if (statusFilter === 'completed') return m.status === 'completed';
        if (statusFilter === 'error') return m.status === 'error';
        return m.status === statusFilter;
      });
    }

    // Competition filter
    if (competitionFilter !== 'All') {
      filtered = filtered.filter(m => m.competition === competitionFilter);
    }

    // Sort options
    const sorted = [...filtered];
    switch (sortBy) {
      case 'oldest':
        sorted.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
        break;
      case 'tactical_power':
        sorted.sort((a, b) => (b.tacticalScore || 0) - (a.tacticalScore || 0));
        break;
      case 'recent_analysis':
        sorted.sort((a, b) => new Date(b.analysisDate).getTime() - new Date(a.analysisDate).getTime());
        break;
      case 'insights':
        sorted.sort((a, b) => (b.insightCount || 0) - (a.insightCount || 0));
        break;
      case 'newest':
      default:
        sorted.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
        break;
    }

    return sorted;
  }, [matches, searchQuery, statusFilter, competitionFilter, sortBy]);

  function formatDate(dateStr: string): string {
    if (!dateStr) return '--';
    try {
      const d = new Date(dateStr);
      return d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });
    } catch {
      return dateStr;
    }
  }

  function getStatusBadge(status: 'completed' | 'processing' | 'pending' | 'error') {
    switch (status) {
      case 'completed':
        return { label: 'Completed', bg: 'bg-emerald-500/10', text: 'text-emerald-400', border: 'border-emerald-500/20', dot: 'bg-emerald-500' };
      case 'processing':
        return { label: 'Processing', bg: 'bg-amber-500/10', text: 'text-amber-400', border: 'border-amber-500/20', dot: 'bg-amber-500 animate-pulse' };
      case 'pending':
        return { label: 'Pending', bg: 'bg-gray-800', text: 'text-gray-400', border: 'border-gray-700', dot: 'bg-gray-500' };
      case 'error':
        return { label: 'Error', bg: 'bg-red-500/10', text: 'text-red-400', border: 'border-red-500/20', dot: 'bg-red-500' };
      default:
        return { label: status || 'Unknown', bg: 'bg-gray-800', text: 'text-gray-400', border: 'border-gray-700', dot: 'bg-gray-500' };
    }
  }

  if (loading) {
    return (
      <div className="h-full w-full bg-[#050805] flex flex-col items-center justify-center gap-4">
        <Loader2 className="animate-spin text-emerald-500" size={48} />
        <p className="text-sm font-mono text-gray-500 uppercase tracking-widest animate-pulse">Loading Match Library...</p>
      </div>
    );
  }

  return (
    <div className="h-full w-full bg-[#050805] flex flex-col font-sans overflow-hidden">
      
      {/* ── Header ────────────────────────────────────────────────────── */}
      <div className="relative px-8 pt-8 pb-6 border-b border-gray-900/60 bg-gradient-to-b from-[#0a120a] to-[#050805]">
        <div className="absolute top-0 right-0 w-96 h-96 bg-emerald-500/[0.02] rounded-full blur-[100px] -translate-y-1/2 translate-x-1/4" />
        <div className="relative flex flex-col lg:flex-row justify-between items-start lg:items-end gap-4">
          <div>
            <h1 className="text-2xl font-bold text-gray-100 tracking-tight flex items-center gap-3">
              <div className="h-9 w-9 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
                <Film size={18} className="text-emerald-500" />
              </div>
              Match Library
            </h1>
            <p className="text-sm text-gray-500 mt-1.5 font-mono">
              Store and manage match setups, tactical analytics, and exported reports locally.
            </p>
          </div>

          {/* Stats Bar */}
          <div className="flex items-center gap-6 bg-black/30 border border-gray-900 px-5 py-2.5 rounded-xl">
            <div className="flex flex-col gap-0.5">
              <span className="text-[9px] font-mono text-gray-600 uppercase tracking-widest">Total library</span>
              <span className="text-gray-200 font-mono font-bold text-sm">{matches.length}</span>
            </div>
            <div className="h-6 w-px bg-gray-800" />
            <div className="flex flex-col gap-0.5">
              <span className="text-[9px] font-mono text-gray-600 uppercase tracking-widest">Completed analyses</span>
              <span className="text-emerald-400 font-mono font-bold text-sm">{counts.completed}</span>
            </div>
          </div>
        </div>
      </div>

      {/* ── Filters & Controls ─────────────────────────────────────────── */}
      <div className="px-8 py-4 flex flex-col gap-4 border-b border-gray-900/30">
        
        {/* Search & Selectors */}
        <div className="flex flex-col md:flex-row gap-3">
          <div className="relative flex-1 max-w-lg">
            <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 text-gray-600 h-4 w-4" />
            <input
              id="match-search"
              type="text"
              placeholder="Search by teams, competition, video title..."
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              className="w-full bg-black/40 border border-gray-900 rounded-xl py-2.5 pl-10 pr-4 text-xs text-gray-200 focus:outline-none focus:border-emerald-500/50 transition-colors font-mono placeholder:text-gray-700"
            />
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-2">
              <SlidersHorizontal size={14} className="text-gray-600" />
              <select
                id="match-sort"
                value={sortBy}
                onChange={e => setSortBy(e.target.value as SortOption)}
                className="bg-black/40 border border-gray-900 rounded-xl px-3 py-2.5 text-xs font-mono text-gray-400 focus:outline-none focus:border-emerald-500/50 cursor-pointer"
              >
                <option value="newest">Newest First</option>
                <option value="oldest">Oldest First</option>
                <option value="tactical_power">Highest Tactical Score</option>
                <option value="recent_analysis">Most Recent Analysis</option>
                <option value="insights">Most Insights</option>
              </select>
            </div>

            <select
              id="match-comp-filter"
              value={competitionFilter}
              onChange={e => setCompetitionFilter(e.target.value)}
              className="bg-black/40 border border-gray-900 rounded-xl px-3 py-2.5 text-xs font-mono text-gray-400 focus:outline-none focus:border-emerald-500/50 cursor-pointer"
            >
              <option value="All">All Competitions</option>
              {competitionsList.filter(c => c !== 'All').map(c => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Status Filter Chips */}
        <div className="flex flex-wrap gap-2 items-center">
          <span className="text-[10px] font-mono text-gray-600 uppercase tracking-wider mr-2">Filter status:</span>
          
          <button
            onClick={() => setStatusFilter('all')}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all border ${
              statusFilter === 'all'
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20 font-bold'
                : 'bg-black/20 text-gray-500 border-transparent hover:text-gray-300'
            }`}
          >
            All ({counts.all})
          </button>
          
          <button
            onClick={() => setStatusFilter('completed')}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all border ${
              statusFilter === 'completed'
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20 font-bold'
                : 'bg-black/20 text-gray-500 border-transparent hover:text-gray-300'
            }`}
          >
            Completed ({counts.completed})
          </button>

          <button
            onClick={() => setStatusFilter('processing')}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all border ${
              statusFilter === 'processing'
                ? 'bg-amber-500/10 text-amber-400 border-amber-500/20 font-bold'
                : 'bg-black/20 text-gray-500 border-transparent hover:text-gray-300'
            }`}
          >
            Processing ({counts.processing})
          </button>

          <button
            onClick={() => setStatusFilter('pending')}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all border ${
              statusFilter === 'pending'
                ? 'bg-gray-800 text-gray-400 border-gray-700 font-bold'
                : 'bg-black/20 text-gray-500 border-transparent hover:text-gray-300'
            }`}
          >
            Pending ({counts.pending})
          </button>

          <button
            onClick={() => setStatusFilter('error')}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all border ${
              statusFilter === 'error'
                ? 'bg-red-500/10 text-red-400 border-red-500/20 font-bold'
                : 'bg-black/20 text-gray-500 border-transparent hover:text-gray-300'
            }`}
          >
            Error ({counts.error})
          </button>
        </div>
      </div>

      {/* ── Matches Card Grid ─────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto custom-scrollbar px-8 py-6">
        {filteredAndSorted.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-72 text-center border border-dashed border-gray-900 rounded-3xl p-8">
            <Film size={40} className="text-gray-800 mb-4 animate-pulse" />
            <p className="text-gray-400 font-mono text-sm">No matches match search parameters</p>
            <p className="text-gray-600 font-mono text-xs mt-1">Try clearing filters or search queries</p>
            {(searchQuery || statusFilter !== 'all' || competitionFilter !== 'All') && (
              <button
                onClick={() => {
                  setSearchQuery('');
                  setStatusFilter('all');
                  setCompetitionFilter('All');
                }}
                className="mt-4 px-4 py-2 bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/20 rounded-xl text-emerald-400 text-xs font-mono transition-all"
              >
                Clear all filters
              </button>
            )}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            {filteredAndSorted.map(match => {
              const status = getStatusBadge(match.status);
              const isCompleted = match.status === 'completed';
              const isOpening = openingMatch === match.id;
              const isBusy = actionInProgress === match.id;
              const isHovered = hoveredCard === match.id;

              return (
                <div
                  key={match.id}
                  id={`match-card-${match.id}`}
                  onMouseEnter={() => setHoveredCard(match.id)}
                  onMouseLeave={() => setHoveredCard(null)}
                  className={`group relative bg-[#090d09] border rounded-2xl overflow-hidden transition-all duration-300 flex flex-col justify-between ${
                    isHovered
                      ? 'border-emerald-500/30 shadow-2xl shadow-emerald-950/20 scale-[1.01]'
                      : 'border-gray-900 shadow-xl'
                  }`}
                >
                  
                  {/* Thumbnail & Tactical Score Overlay */}
                  <div className="relative h-40 bg-black overflow-hidden border-b border-gray-950 flex-shrink-0">
                    {match.thumbnail ? (
                      <img
                        src={match.thumbnail}
                        alt="Match Radar Grid"
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-700 ease-out opacity-80"
                      />
                    ) : (
                      <div className="w-full h-full bg-[#050805] flex items-center justify-center">
                        <Film className="text-gray-800" size={36} />
                      </div>
                    )}
                    
                    {/* Score Ring Overlay */}
                    {isCompleted && (
                      <div className="absolute top-3 right-3 bg-black/75 backdrop-blur-md border border-emerald-500/30 px-2.5 py-1.5 rounded-xl text-emerald-400 font-mono font-bold text-xs flex items-center gap-1.5 shadow-lg">
                        <Award size={13} className="text-emerald-500" />
                        <span>{match.tacticalScore} Score</span>
                      </div>
                    )}

                    {/* Status Badge */}
                    <div className="absolute top-3 left-3">
                      <span className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl text-[10px] font-mono font-bold backdrop-blur-md border ${status.bg} ${status.text} ${status.border} shadow-lg`}>
                        <span className={`h-1.5 w-1.5 rounded-full ${status.dot}`} />
                        {status.label.toUpperCase()}
                      </span>
                    </div>

                    {/* Video Name Overlay */}
                    {match.videoName && (
                      <div className="absolute bottom-2 left-3 bg-black/60 backdrop-blur-sm px-2 py-0.5 rounded text-[8px] font-mono text-gray-500 max-w-[85%] truncate">
                        {match.videoName}
                      </div>
                    )}
                  </div>

                  {/* Body Content */}
                  <div className="p-5 flex-1 flex flex-col justify-between">
                    <div>
                      {/* Competition Title */}
                      <div className="text-[9px] font-mono font-bold text-emerald-500 uppercase tracking-widest mb-1 flex items-center gap-1">
                        <Target size={9} />
                        {match.competition || 'Local Tournament'}
                      </div>

                      {/* Match Heading */}
                      <h3 className="text-base font-bold text-gray-100 tracking-tight leading-snug group-hover:text-emerald-400 transition-colors duration-300">
                        {match.title}
                      </h3>

                      {/* Versus Teams Details */}
                      <div className="text-xs text-gray-400 font-medium mt-1">
                        {match.teamA} <span className="text-gray-600">vs</span> {match.teamB}
                      </div>

                      {/* Match Parameters Metarow */}
                      <div className="flex items-center gap-x-4 mt-3 text-[10px] font-mono text-gray-500 border-t border-gray-900/40 pt-3">
                        <span className="flex items-center gap-1"><Calendar size={11} /> {formatDate(match.date)}</span>
                        <span className="flex items-center gap-1"><Clock size={11} /> {match.duration}</span>
                      </div>
                    </div>

                    {/* Quick Stats Grid */}
                    {isCompleted && (
                      <div className="grid grid-cols-3 gap-2 mt-4 bg-black/40 border border-gray-900/50 rounded-xl p-2.5">
                        <div className="flex flex-col gap-0.5 text-center">
                          <span className="text-[8px] font-mono text-gray-600 uppercase tracking-wider">Compact</span>
                          <span className="text-xs font-mono font-bold text-gray-200">{match.compactness}%</span>
                        </div>
                        <div className="flex flex-col gap-0.5 text-center border-x border-gray-900">
                          <span className="text-[8px] font-mono text-gray-600 uppercase tracking-wider">Transition</span>
                          <span className="text-xs font-mono font-bold text-gray-200">{match.transitionSpeed}</span>
                        </div>
                        <div className="flex flex-col gap-0.5 text-center">
                          <span className="text-[8px] font-mono text-gray-600 uppercase tracking-wider">Insights</span>
                          <span className="text-xs font-mono font-bold text-emerald-400">{match.insightCount}</span>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Actions Bar */}
                  <div className="px-5 pb-5 pt-3 flex items-center gap-2 border-t border-gray-900/30 bg-black/10 flex-shrink-0">
                    <button
                      id={`view-analysis-${match.id}`}
                      disabled={!isCompleted || isOpening || isBusy}
                      onClick={() => handleOpenDashboard(match)}
                      className={`flex-1 flex items-center justify-center gap-1.5 py-2.5 rounded-xl text-[11px] font-mono font-bold transition-all ${
                        isCompleted
                          ? 'bg-emerald-500 text-black hover:bg-emerald-400 shadow-md shadow-emerald-950/20 active:scale-[0.98]'
                          : 'bg-gray-900/50 text-gray-600 border border-gray-800 cursor-not-allowed'
                      }`}
                    >
                      {isOpening ? (
                        <><Loader2 size={12} className="animate-spin" /> Load...</>
                      ) : (
                        <><BarChart2 size={12} /> Open Analysis</>
                      )}
                    </button>

                    {isCompleted && (
                      <button
                        onClick={() => onViewPlayers(match.id)}
                        className="p-2.5 rounded-xl bg-black/40 border border-gray-900 text-gray-400 hover:text-emerald-400 hover:border-emerald-500/20 transition-all"
                        title="View Player Reports"
                      >
                        <Users size={12} />
                      </button>
                    )}

                    <button
                      onClick={() => handleDuplicate(match.id)}
                      disabled={isBusy}
                      className="p-2.5 rounded-xl bg-black/40 border border-gray-900 text-gray-400 hover:text-emerald-400 hover:border-emerald-500/20 transition-all disabled:opacity-50"
                      title="Duplicate Analysis"
                    >
                      <Copy size={12} />
                    </button>

                    <button
                      onClick={() => handleExport(match)}
                      disabled={isBusy}
                      className="p-2.5 rounded-xl bg-black/40 border border-gray-900 text-gray-400 hover:text-emerald-400 hover:border-emerald-500/20 transition-all disabled:opacity-50"
                      title="Export Analysis Details"
                    >
                      <Download size={12} />
                    </button>

                    <button
                      onClick={() => handleDelete(match.id)}
                      disabled={isBusy}
                      className="p-2.5 rounded-xl bg-black/40 border border-gray-900 text-gray-400 hover:text-red-400 hover:border-red-500/20 transition-all disabled:opacity-50"
                      title="Delete Analysis"
                    >
                      <Trash2 size={12} />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
