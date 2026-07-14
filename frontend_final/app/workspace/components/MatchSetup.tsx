"use client";
import React, { useState, useRef } from 'react';
import {
  UploadCloud, Search, Users, Shield, CheckCircle,
  FileVideo, ArrowRight, ArrowLeft, Settings, Trash,
  Plus, Edit3, Save, X, Activity, Loader2
} from 'lucide-react';
import { getFixtureLineup } from '@/lib/services/footballApi';
import { useMatchSearch } from '@/hooks/useMatchSearch';
import type { FixtureSearchResult } from '@/types/lineup';

interface MatchSetupPlayer {
  id: string;
  name: string;
  number: number;
  position: string;
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

// ── Mock Initial Players Generators ───────────────────────────────────
function getMockRoster(teamName: string, isTeamA: boolean): MatchSetupPlayer[] {
  const prefix = isTeamA ? 'A' : 'B';
  const positions = [
    { pos: 'GK', count: 1 },
    { pos: 'DF', count: 4 },
    { pos: 'MF', count: 4 },
    { pos: 'FW', count: 2 },
  ];
  
  const startingPlayers: MatchSetupPlayer[] = [];
  let currentNum = 1;

  positions.forEach(({ pos, count }) => {
    for (let i = 0; i < count; i++) {
      startingPlayers.push({
        id: `p-${prefix}-${currentNum}`,
        name: `${teamName} Player ${currentNum}`,
        number: currentNum === 1 && pos === 'GK' ? 1 : currentNum,
        position: pos,
        isStarting: true
      });
      currentNum++;
    }
  });

  const benchPlayers: MatchSetupPlayer[] = [];
  for (let i = 1; i <= 5; i++) {
    benchPlayers.push({
      id: `p-${prefix}-bench-${i}`,
      name: `${teamName} Sub ${i}`,
      number: currentNum,
      position: i === 1 ? 'GK' : i <= 3 ? 'DF' : 'MF',
      isStarting: false
    });
    currentNum++;
  }

  return [...startingPlayers, ...benchPlayers];
}

export function MatchSetup() {
  const [step, setStep] = useState<number>(1);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  // ── Step 1 State: Video ──────────────────────────────────────────────
  const [videoFile, setVideoFile] = useState<{ name: string; size: string; duration: string } | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  // ── Step 2 State: Fixture (live search via useMatchSearch) ───────────
  const {
    query: searchQuery,
    setQuery: setSearchQuery,
    results: searchResults,
    loading: searchLoading,
    error: searchError,
    hasSearched: searchHasSearched,
  } = useMatchSearch();
  const [selectedFixture, setSelectedFixture] = useState<FixtureSearchResult | null>(null);
  const [lineupLoading, setLineupLoading] = useState(false);
  const [lineupError, setLineupError] = useState<string | null>(null);
  const [lineupReady, setLineupReady] = useState(false);
  const pendingFixtureRef = useRef<FixtureSearchResult | null>(null);

  // ── Step 3 & 4 State: Lineups & Verification ────────────────────────
  const [teamAName, setTeamAName] = useState('Paris Saint-Germain');
  const [teamBName, setTeamBName] = useState('Inter Milan');
  const [teamAFormation, setTeamAFormation] = useState('4-4-2');
  const [teamBFormation, setTeamBFormation] = useState('3-5-2');

  const [teamAPlayers, setTeamAPlayers] = useState<MatchSetupPlayer[]>(() => getMockRoster('Team A', true));
  const [teamBPlayers, setTeamBPlayers] = useState<MatchSetupPlayer[]>(() => getMockRoster('Team B', false));

  // Editing state for table verifier
  const [editingPlayerId, setEditingPlayerId] = useState<string | null>(null);
  const [editName, setEditName] = useState('');
  const [editNumber, setEditNumber] = useState<number>(0);
  const [editPos, setEditPos] = useState('MF');

  // ── Step 1 Handlers ──────────────────────────────────────────────────
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
      setVideoFile({
        name: file.name,
        size: `${sizeMB} MB`,
        duration: '90:00 (Mocked)'
      });
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
      setVideoFile({
        name: file.name,
        size: `${sizeMB} MB`,
        duration: '90:00 (Mocked)'
      });
    }
  };

  const triggerMockUpload = () => {
    setVideoFile({
      name: 'psg_vs_inter_final_2026_cam1.mp4',
      size: '2.4 GB',
      duration: '94:12'
    });
  };

  // ── Step 2 Handlers ──────────────────────────────────────────────────
  const handleSelectFixture = async (fix: FixtureSearchResult) => {
    pendingFixtureRef.current = fix;
    setSelectedFixture(fix);
    setLineupLoading(true);
    setLineupError(null);
    setLineupReady(false);

    try {
      const lineup = await getFixtureLineup(fix.id);
      setTeamAName(lineup.team_a.name);
      setTeamBName(lineup.team_b.name);
      setTeamAFormation(lineup.team_a.formation);
      setTeamBFormation(lineup.team_b.formation);
      setTeamAPlayers(lineup.team_a.players);
      setTeamBPlayers(lineup.team_b.players);
      setLineupReady(true);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Football data service unavailable.';
      setLineupError(msg);
    } finally {
      setLineupLoading(false);
    }
  };

  const handleLineupRetry = () => {
    if (pendingFixtureRef.current) {
      void handleSelectFixture(pendingFixtureRef.current);
    }
  };

  // ── Edit Handlers ────────────────────────────────────────────────────
  const startEdit = (player: MatchSetupPlayer) => {
    setEditingPlayerId(player.id);
    setEditName(player.name);
    setEditNumber(player.number);
    setEditPos(player.position);
  };

  const saveEdit = (team: 'A' | 'B') => {
    if (!editingPlayerId) return;
    const updateFn = (players: MatchSetupPlayer[]) =>
      players.map(p =>
        p.id === editingPlayerId
          ? { ...p, name: editName, number: editNumber, position: editPos }
          : p
      );

    if (team === 'A') {
      setTeamAPlayers(updateFn);
    } else {
      setTeamBPlayers(updateFn);
    }
    setEditingPlayerId(null);
  };

  const toggleStarting = (id: string, team: 'A' | 'B') => {
    const updateFn = (players: MatchSetupPlayer[]) =>
      players.map(p => (p.id === id ? { ...p, isStarting: !p.isStarting } : p));
    if (team === 'A') {
      setTeamAPlayers(updateFn);
    } else {
      setTeamBPlayers(updateFn);
    }
  };

  const deletePlayer = (id: string, team: 'A' | 'B') => {
    const updateFn = (players: MatchSetupPlayer[]) => players.filter(p => p.id !== id);
    if (team === 'A') {
      setTeamAPlayers(updateFn);
    } else {
      setTeamBPlayers(updateFn);
    }
  };

  const addPlayer = (team: 'A' | 'B') => {
    const id = `new-p-${Date.now()}`;
    const newPlayer: MatchSetupPlayer = {
      id,
      name: 'New Player',
      number: 99,
      position: 'MF',
      isStarting: false
    };

    if (team === 'A') {
      setTeamAPlayers([...teamAPlayers, newPlayer]);
      startEdit(newPlayer);
    } else {
      setTeamBPlayers([...teamBPlayers, newPlayer]);
      startEdit(newPlayer);
    }
  };

  // ── Save Setup Handler (Step 5) ──────────────────────────────────────
  const handleSaveSetup = () => {
    const finalSetup: MatchSetupData = {
      match_id: `match-${Date.now()}`,
      fixture_name: selectedFixture?.name || `${teamAName} vs ${teamBName}`,
      competition: selectedFixture?.competition || 'Friendly Match',
      video_filename: videoFile?.name || 'no_video.mp4',
      video_size: videoFile?.size || '0 MB',
      video_duration: videoFile?.duration || '00:00',
      team_a: {
        name: teamAName,
        formation: teamAFormation,
        players: teamAPlayers,
      },
      team_b: {
        name: teamBName,
        formation: teamBFormation,
        players: teamBPlayers,
      },
      created_at: new Date().toISOString()
    };

    // Store in localStorage
    try {
      const stored = localStorage.getItem('gaffer-match-setups');
      const setupsList = stored ? JSON.parse(stored) : [];
      setupsList.push(finalSetup);
      localStorage.setItem('gaffer-match-setups', JSON.stringify(setupsList));
      setSuccessMsg("Match Setup successfully stored locally in workspace storage.");
    } catch (e) {
      console.error("Local storage error:", e);
      setSuccessMsg("Error saving setup to local storage.");
    }
  };

  const resetForm = () => {
    setStep(1);
    setVideoFile(null);
    setSelectedFixture(null);
    setSearchQuery('');
    setSuccessMsg(null);
    setLineupLoading(false);
    setLineupError(null);
    setLineupReady(false);
  };

  // ── Helper UI Layout variables ────────────────────────────────────────
  const stepsList = [
    { label: 'Upload Video', desc: 'Add match file' },
    { label: 'Select Fixture', desc: 'Identify match' },
    { label: 'Import Lineups', desc: 'Set formations' },
    { label: 'Verify Teams', desc: 'Final review' },
    { label: 'Ready', desc: 'Save config' }
  ];

  return (
    <div className="h-full w-full bg-[#050805] flex flex-col font-sans overflow-hidden">
      
      {/* ── Header ────────────────────────────────────────────────────── */}
      <div className="px-8 pt-8 pb-4 border-b border-gray-900/60 bg-gradient-to-b from-[#0a120a] to-[#050805]">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
            <Settings size={18} className="text-emerald-500 animate-spin-slow" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gray-100 tracking-tight">Match Setup Wizard</h1>
            <p className="text-xs text-gray-500 font-mono mt-1">Pre-configure tactical rosters and fixture metadata before pipeline runtime</p>
          </div>
        </div>

        {/* Progress Tracker at Top */}
        <div className="mt-8 relative max-w-5xl mx-auto px-4">
          <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-gray-900 -translate-y-1/2 z-0" />
          <div 
            className="absolute top-1/2 left-0 h-0.5 bg-emerald-500 transition-all duration-500 -translate-y-1/2 z-0"
            style={{ width: `${((step - 1) / (stepsList.length - 1)) * 100}%` }}
          />
          <div className="relative flex justify-between z-10">
            {stepsList.map((s, idx) => {
              const num = idx + 1;
              const isActive = num === step;
              const isPast = num < step;
              return (
                <div key={idx} className="flex flex-col items-center">
                  <button
                    onClick={() => {
                      // Allow back and forward if already filled
                      if (num < step || (num === 2 && videoFile) || (num === 3 && selectedFixture)) {
                        setStep(num);
                      }
                    }}
                    disabled={num > step && !(num === 2 && videoFile)}
                    className={`h-8 w-8 rounded-full border flex items-center justify-center font-mono text-xs font-bold transition-all duration-300 ${
                      isActive 
                        ? 'bg-emerald-500 text-black border-emerald-400 shadow-[0_0_15px_rgba(16,185,129,0.4)]'
                        : isPast
                        ? 'bg-[#111a12] text-emerald-400 border-emerald-500/30'
                        : 'bg-black text-gray-600 border-gray-900 cursor-not-allowed'
                    }`}
                  >
                    {isPast ? <CheckCircle size={14} className="text-emerald-400" /> : num}
                  </button>
                  <span className={`text-[10px] font-bold mt-2 font-mono uppercase tracking-wider ${isActive ? 'text-emerald-400' : isPast ? 'text-gray-400' : 'text-gray-600'}`}>
                    {s.label}
                  </span>
                  <span className="text-[8px] text-gray-700 font-mono mt-0.5 hidden sm:inline">
                    {s.desc}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* ── Wizard Main Content Container ───────────────────────────────── */}
      <div className="flex-1 overflow-y-auto custom-scrollbar px-8 py-6 flex flex-col justify-between max-w-6xl mx-auto w-full">
        
        {/* Step-specific views */}
        {/* justify-start for step 4 (tall tables) so rows are not cut off above
            the scroll origin; justify-center for shorter steps 1-3 & 5. */}
        <div className={`flex-1 flex flex-col min-h-[380px] my-4 ${step === 4 ? 'justify-start' : 'justify-center'}`}>
          
          {/* STEP 1: Upload Video */}
          {step === 1 && (
            <div className="w-full max-w-2xl mx-auto flex flex-col gap-6 animate-fade-in">
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`border-2 border-dashed rounded-2xl p-12 flex flex-col items-center justify-center transition-all ${
                  isDragging
                    ? 'border-emerald-500 bg-[#111a12]/30 shadow-[0_0_30px_rgba(16,185,129,0.15)]'
                    : videoFile
                    ? 'border-emerald-500/40 bg-[#111a12]/10'
                    : 'border-gray-800 bg-black/40 hover:border-emerald-500/40 hover:bg-[#111a12]/10'
                }`}
              >
                <input
                  type="file"
                  id="match-setup-file"
                  accept="video/mp4,video/quicktime,video/mkv"
                  className="hidden"
                  onChange={handleFileUpload}
                />
                
                <UploadCloud size={48} className={`mb-4 transition-colors ${videoFile ? 'text-emerald-500' : 'text-gray-600'}`} />
                <h3 className="text-base font-bold text-gray-200 mb-2">Drag and drop your match video</h3>
                <p className="text-xs text-gray-600 font-mono mb-6">Support MP4, MOV, MKV files up to 15GB</p>
                
                <label
                  htmlFor="match-setup-file"
                  className="px-6 py-2.5 bg-emerald-500 hover:bg-emerald-400 text-black font-bold rounded-xl text-xs transition-colors cursor-pointer"
                >
                  Browse Files
                </label>
              </div>

              {/* Mock upload helper */}
              <div className="flex items-center justify-between bg-black/30 border border-gray-900 p-4 rounded-xl">
                <div className="flex flex-col gap-1">
                  <span className="text-[10px] font-mono text-gray-500 uppercase tracking-wider">Demo / Sandbox Utility</span>
                  <span className="text-xs text-gray-400">Load mock video profile to skip file uploading</span>
                </div>
                <button
                  id="mock-upload-btn"
                  onClick={triggerMockUpload}
                  className="px-4 py-2 bg-gray-900 border border-gray-800 text-gray-400 hover:text-emerald-400 hover:border-emerald-500/30 text-xs font-mono font-bold rounded-lg transition-colors"
                >
                  Load Mock Video
                </button>
              </div>

              {/* Uploaded File summary */}
              {videoFile && (
                <div className="bg-[#111a12]/50 border border-emerald-500/20 rounded-xl p-5 flex items-start gap-4">
                  <div className="p-2.5 bg-emerald-500/10 rounded-lg text-emerald-400">
                    <FileVideo size={22} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h4 className="text-sm font-bold text-gray-200 truncate">{videoFile.name}</h4>
                    <div className="flex gap-4 mt-2 text-[10px] font-mono text-gray-500">
                      <span>Size: <strong className="text-gray-300">{videoFile.size}</strong></span>
                      <span>Duration: <strong className="text-gray-300">{videoFile.duration}</strong></span>
                    </div>
                  </div>
                  <button 
                    onClick={() => setVideoFile(null)} 
                    className="p-1 text-gray-600 hover:text-red-400 transition-colors"
                  >
                    <X size={16} />
                  </button>
                </div>
              )}
            </div>
          )}

          {/* STEP 2: Fixture Selector */}
          {step === 2 && (
            <div className="w-full max-w-2xl mx-auto flex flex-col gap-6 animate-fade-in">
              <div className="relative">
                <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 text-gray-600 h-4 w-4" />
                <input
                  id="fixture-search-box"
                  type="text"
                  placeholder="Search fixture... (e.g. PSG, Real Madrid, Premier League)"
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                  className="w-full bg-black/40 border border-gray-900 rounded-xl py-3 pl-10 pr-4 text-xs text-gray-200 focus:outline-none focus:border-emerald-500/50 transition-colors font-mono placeholder:text-gray-700"
                />
              </div>

              <div className="border border-gray-900 bg-black/20 rounded-xl overflow-hidden max-h-[250px] overflow-y-auto">
                <div className="bg-black/40 px-4 py-2 border-b border-gray-900 text-[10px] font-mono text-gray-600 uppercase tracking-widest">
                  Fixtures Found ({searchResults.length})
                </div>
                {searchLoading ? (
                  <div className="p-8 text-center text-xs text-gray-500 font-mono flex items-center justify-center gap-2">
                    <Loader2 size={13} className="animate-spin text-emerald-500" />
                    Searching fixtures...
                  </div>
                ) : searchError ? (
                  <div className="p-8 text-center text-xs text-red-400 font-mono">
                    {searchError}
                  </div>
                ) : searchQuery.trim().length < 2 ? (
                  <div className="p-8 text-center text-xs text-gray-600 font-mono">
                    Type at least 2 characters to search fixtures.
                  </div>
                ) : searchHasSearched && searchResults.length === 0 ? (
                  <div className="p-8 text-center text-xs text-gray-600 font-mono">
                    No matches found. Try another team name.
                  </div>
                ) : (
                  searchResults.map(fix => {
                    const isSelected = selectedFixture?.id === fix.id;
                    return (
                      <button
                        key={fix.id}
                        id={`select-fixture-${fix.id}`}
                        onClick={() => { void handleSelectFixture(fix); }}
                        className={`w-full flex items-center justify-between px-4 py-3.5 border-b border-gray-900/50 text-left transition-all ${
                          isSelected 
                            ? 'bg-[#111a12]/50 text-emerald-400' 
                            : 'hover:bg-white/[0.01] text-gray-300'
                        }`}
                      >
                        <div className="flex-1 min-w-0">
                          <span className="text-xs font-bold block">{fix.name}</span>
                          <span className="text-[10px] font-mono text-gray-600 mt-0.5 block">{fix.competition}</span>
                        </div>
                        {isSelected && (
                          <CheckCircle size={16} className="text-emerald-400 ml-2" />
                        )}
                      </button>
                    );
                  })
                )}
              </div>

              {/* Lineup fetch status strips */}
              {lineupLoading && (
                <div className="bg-black/30 border border-gray-900 rounded-xl px-4 py-3 flex items-center gap-3 text-xs font-mono text-gray-400">
                  <Loader2 size={13} className="animate-spin text-emerald-500 shrink-0" />
                  Loading official lineup...
                </div>
              )}
              {lineupError && (
                <div className="bg-red-950/20 border border-red-900/40 rounded-xl px-4 py-3 flex items-center justify-between gap-4">
                  <span className="text-xs font-mono text-red-400 min-w-0 truncate">{lineupError}</span>
                  <button
                    onClick={handleLineupRetry}
                    className="shrink-0 text-[10px] font-mono font-bold px-3 py-1.5 bg-red-500/10 border border-red-500/30 text-red-400 rounded-lg hover:bg-red-500/20 transition-colors"
                  >
                    Retry
                  </button>
                </div>
              )}
              {selectedFixture && lineupReady && (
                <div className="bg-[#0a120a] border border-emerald-500/20 rounded-xl px-4 py-3 text-[10px] font-mono text-gray-500 flex flex-wrap gap-x-5 gap-y-1">
                  <span className="text-emerald-400 font-bold">{selectedFixture.competition}</span>
                  {selectedFixture.date && (
                    <span>{new Date(selectedFixture.date).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })}</span>
                  )}
                  {selectedFixture.venue && <span>{selectedFixture.venue}</span>}
                </div>
              )}

              {/* Custom Fixture Fallback */}
              <div className="bg-[#0a0f0a] border border-gray-900 p-5 rounded-xl flex flex-col gap-4">
                <span className="text-[10px] font-mono text-gray-500 uppercase tracking-wider block">Custom Fixture Entry</span>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div className="flex flex-col gap-1.5">
                    <label className="text-[10px] font-mono text-gray-600 uppercase">Home Team</label>
                    <input
                      id="custom-home-team"
                      type="text"
                      value={teamAName}
                      onChange={e => {
                        setTeamAName(e.target.value);
                        setSelectedFixture({ id: '', name: `${e.target.value} vs ${teamBName}`, competition: 'Custom Fixture', date: '', homeTeam: e.target.value, awayTeam: teamBName });
                        setLineupReady(true);
                        setLineupError(null);
                      }}
                      className="bg-black/40 border border-gray-900 rounded-lg p-2 text-xs text-gray-300 font-mono focus:outline-none focus:border-emerald-500/50"
                    />
                  </div>
                  <div className="flex flex-col gap-1.5">
                    <label className="text-[10px] font-mono text-gray-600 uppercase">Away Team</label>
                    <input
                      id="custom-away-team"
                      type="text"
                      value={teamBName}
                      onChange={e => {
                        setTeamBName(e.target.value);
                        setSelectedFixture({ id: '', name: `${teamAName} vs ${e.target.value}`, competition: 'Custom Fixture', date: '', homeTeam: teamAName, awayTeam: e.target.value });
                        setLineupReady(true);
                        setLineupError(null);
                      }}
                      className="bg-black/40 border border-gray-900 rounded-lg p-2 text-xs text-gray-300 font-mono focus:outline-none focus:border-emerald-500/50"
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* STEP 3: Lineup Selection */}
          {step === 3 && (
            <div className="w-full flex flex-col gap-6 animate-fade-in">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                
                {/* Team A Lineup configuration */}
                <div className="bg-[#0a0f0a] border border-gray-900 rounded-xl p-5 flex flex-col gap-4">
                  <div className="flex justify-between items-center border-b border-gray-900 pb-3">
                    <h4 className="text-sm font-bold text-gray-200 flex items-center gap-2">
                      <Shield size={14} className="text-emerald-400" />
                      {teamAName}
                    </h4>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-mono text-gray-600">Formation:</span>
                      <select
                        id="formation-team-a"
                        value={teamAFormation}
                        onChange={e => setTeamAFormation(e.target.value)}
                        className="bg-black border border-gray-900 text-gray-300 text-xs font-mono rounded px-2 py-1 focus:outline-none focus:border-emerald-500/50 cursor-pointer"
                      >
                        <option value="4-3-3">4-3-3</option>
                        <option value="4-4-2">4-4-2</option>
                        <option value="3-5-2">3-5-2</option>
                        <option value="4-2-3-1">4-2-3-1</option>
                      </select>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <span className="text-[9px] font-mono text-gray-600 uppercase tracking-wider block mb-2">Starting XI</span>
                      <div className="space-y-1.5 max-h-[360px] overflow-y-auto custom-scrollbar pr-1">
                        {teamAPlayers.filter(p => p.isStarting).map(p => (
                          <div key={p.id} className="flex items-center justify-between bg-black/30 border border-gray-900/40 rounded px-2 py-1 text-xs">
                            <span className="font-mono text-[10px] text-gray-600 bg-gray-900 px-1.5 py-0.5 rounded mr-2 shrink-0">{p.number}</span>
                            <span className="text-gray-300 truncate flex-1">{p.name}</span>
                            <span className="text-[9px] font-mono text-emerald-500 uppercase ml-2 shrink-0">{p.position}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div>
                      <span className="text-[9px] font-mono text-gray-600 uppercase tracking-wider block mb-2">Substitutes</span>
                      <div className="space-y-1.5 max-h-[220px] overflow-y-auto custom-scrollbar pr-1">
                        {teamAPlayers.filter(p => !p.isStarting).map(p => (
                          <div key={p.id} className="flex items-center justify-between bg-black/30 border border-gray-900/40 rounded px-2 py-1 text-xs">
                            <span className="font-mono text-[10px] text-gray-600 bg-gray-900 px-1.5 py-0.5 rounded mr-2 shrink-0">{p.number}</span>
                            <span className="text-gray-300 truncate flex-1">{p.name}</span>
                            <span className="text-[9px] font-mono text-gray-600 uppercase ml-2 shrink-0">{p.position}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Team B Lineup configuration */}
                <div className="bg-[#0a0f0a] border border-gray-900 rounded-xl p-5 flex flex-col gap-4">
                  <div className="flex justify-between items-center border-b border-gray-900 pb-3">
                    <h4 className="text-sm font-bold text-gray-200 flex items-center gap-2">
                      <Shield size={14} className="text-blue-400" />
                      {teamBName}
                    </h4>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-mono text-gray-600">Formation:</span>
                      <select
                        id="formation-team-b"
                        value={teamBFormation}
                        onChange={e => setTeamBFormation(e.target.value)}
                        className="bg-black border border-gray-900 text-gray-300 text-xs font-mono rounded px-2 py-1 focus:outline-none focus:border-emerald-500/50 cursor-pointer"
                      >
                        <option value="4-3-3">4-3-3</option>
                        <option value="4-4-2">4-4-2</option>
                        <option value="3-5-2">3-5-2</option>
                        <option value="4-2-3-1">4-2-3-1</option>
                      </select>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <span className="text-[9px] font-mono text-gray-600 uppercase tracking-wider block mb-2">Starting XI</span>
                      <div className="space-y-1.5 max-h-[360px] overflow-y-auto custom-scrollbar pr-1">
                        {teamBPlayers.filter(p => p.isStarting).map(p => (
                          <div key={p.id} className="flex items-center justify-between bg-black/30 border border-gray-900/40 rounded px-2 py-1 text-xs">
                            <span className="font-mono text-[10px] text-gray-600 bg-gray-900 px-1.5 py-0.5 rounded mr-2 shrink-0">{p.number}</span>
                            <span className="text-gray-300 truncate flex-1">{p.name}</span>
                            <span className="text-[9px] font-mono text-emerald-500 uppercase ml-2 shrink-0">{p.position}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div>
                      <span className="text-[9px] font-mono text-gray-600 uppercase tracking-wider block mb-2">Substitutes</span>
                      <div className="space-y-1.5 max-h-[220px] overflow-y-auto custom-scrollbar pr-1">
                        {teamBPlayers.filter(p => !p.isStarting).map(p => (
                          <div key={p.id} className="flex items-center justify-between bg-black/30 border border-gray-900/40 rounded px-2 py-1 text-xs">
                            <span className="font-mono text-[10px] text-gray-600 bg-gray-900 px-1.5 py-0.5 rounded mr-2 shrink-0">{p.number}</span>
                            <span className="text-gray-300 truncate flex-1">{p.name}</span>
                            <span className="text-[9px] font-mono text-gray-600 uppercase ml-2 shrink-0">{p.position}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

              </div>
            </div>
          )}

          {/* STEP 4: Verification Grid */}
          {step === 4 && (
            <div className="w-full flex flex-col gap-6 animate-fade-in">
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                
                {/* Verification Table Team A */}
                <div className="bg-[#0a0f0a] border border-gray-900 rounded-xl p-5 flex flex-col gap-3">
                  <div className="flex justify-between items-center border-b border-gray-900 pb-2">
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-mono text-gray-500 uppercase">Edit Team A:</span>
                      <input
                        id="verify-name-team-a"
                        type="text"
                        value={teamAName}
                        onChange={e => setTeamAName(e.target.value)}
                        className="bg-black/60 border border-gray-900 px-2 py-1 text-xs text-emerald-400 font-bold rounded focus:outline-none focus:border-emerald-500/50 font-sans"
                      />
                    </div>
                    <button
                      onClick={() => addPlayer('A')}
                      className="flex items-center gap-1 text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded border border-emerald-500/20 hover:bg-emerald-500/20"
                    >
                      <Plus size={10} /> Add Player
                    </button>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs font-mono">
                      <thead>
                        <tr className="border-b border-gray-900 text-gray-600 uppercase text-[9px]">
                          <th className="py-2 w-10">No.</th>
                          <th className="py-2">Name</th>
                          <th className="py-2 w-16">Pos</th>
                          <th className="py-2 w-20">Status</th>
                          <th className="py-2 w-24 text-right">Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {teamAPlayers.map(p => {
                          const isEditing = editingPlayerId === p.id;
                          return (
                            <tr key={p.id} className="border-b border-gray-900/40 hover:bg-white/[0.01]">
                              <td className="py-2.5">
                                {isEditing ? (
                                  <input
                                    type="number"
                                    value={editNumber}
                                    onChange={e => setEditNumber(parseInt(e.target.value) || 0)}
                                    className="w-10 bg-black border border-gray-900 p-0.5 text-center text-gray-200"
                                  />
                                ) : (
                                  <span className="text-gray-500">{p.number}</span>
                                )}
                              </td>
                              <td className="py-2.5 font-sans">
                                {isEditing ? (
                                  <input
                                    type="text"
                                    value={editName}
                                    onChange={e => setEditName(e.target.value)}
                                    className="w-full bg-black border border-gray-900 p-0.5 text-xs text-gray-200"
                                  />
                                ) : (
                                  <span className="text-gray-300 font-medium">{p.name}</span>
                                )}
                              </td>
                              <td className="py-2.5 text-gray-400">
                                {isEditing ? (
                                  <select
                                    value={editPos}
                                    onChange={e => setEditPos(e.target.value)}
                                    className="bg-black border border-gray-900 p-0.5 text-[10px] text-gray-200"
                                  >
                                    <option value="GK">GK</option>
                                    <option value="DF">DF</option>
                                    <option value="MF">MF</option>
                                    <option value="FW">FW</option>
                                  </select>
                                ) : (
                                  p.position
                                )}
                              </td>
                              <td className="py-2.5">
                                <button
                                  onClick={() => toggleStarting(p.id, 'A')}
                                  className={`text-[9px] uppercase px-1.5 py-0.5 rounded font-mono font-bold ${
                                    p.isStarting 
                                      ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' 
                                      : 'bg-gray-900 text-gray-600 border border-gray-800'
                                  }`}
                                >
                                  {p.isStarting ? 'Starting' : 'Bench'}
                                </button>
                              </td>
                              <td className="py-2.5 text-right">
                                <div className="flex justify-end gap-1.5">
                                  {isEditing ? (
                                    <>
                                      <button onClick={() => saveEdit('A')} className="p-1 bg-emerald-500/20 border border-emerald-500/30 rounded text-emerald-400 hover:bg-emerald-500/30">
                                        <Save size={10} />
                                      </button>
                                      <button onClick={() => setEditingPlayerId(null)} className="p-1 bg-gray-900 border border-gray-850 rounded text-gray-400 hover:text-gray-200">
                                        <X size={10} />
                                      </button>
                                    </>
                                  ) : (
                                    <>
                                      <button onClick={() => startEdit(p)} className="p-1 hover:text-emerald-400 text-gray-500">
                                        <Edit3 size={11} />
                                      </button>
                                      <button onClick={() => deletePlayer(p.id, 'A')} className="p-1 hover:text-red-400 text-gray-600">
                                        <Trash size={11} />
                                      </button>
                                    </>
                                  )}
                                </div>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Verification Table Team B */}
                <div className="bg-[#0a0f0a] border border-gray-900 rounded-xl p-5 flex flex-col gap-3">
                  <div className="flex justify-between items-center border-b border-gray-900 pb-2">
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-mono text-gray-500 uppercase">Edit Team B:</span>
                      <input
                        id="verify-name-team-b"
                        type="text"
                        value={teamBName}
                        onChange={e => setTeamBName(e.target.value)}
                        className="bg-black/60 border border-gray-900 px-2 py-1 text-xs text-blue-400 font-bold rounded focus:outline-none focus:border-emerald-500/50 font-sans"
                      />
                    </div>
                    <button
                      onClick={() => addPlayer('B')}
                      className="flex items-center gap-1 text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded border border-emerald-500/20 hover:bg-emerald-500/20"
                    >
                      <Plus size={10} /> Add Player
                    </button>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs font-mono">
                      <thead>
                        <tr className="border-b border-gray-900 text-gray-600 uppercase text-[9px]">
                          <th className="py-2 w-10">No.</th>
                          <th className="py-2">Name</th>
                          <th className="py-2 w-16">Pos</th>
                          <th className="py-2 w-20">Status</th>
                          <th className="py-2 w-24 text-right">Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {teamBPlayers.map(p => {
                          const isEditing = editingPlayerId === p.id;
                          return (
                            <tr key={p.id} className="border-b border-gray-900/40 hover:bg-white/[0.01]">
                              <td className="py-2.5">
                                {isEditing ? (
                                  <input
                                    type="number"
                                    value={editNumber}
                                    onChange={e => setEditNumber(parseInt(e.target.value) || 0)}
                                    className="w-10 bg-black border border-gray-900 p-0.5 text-center text-gray-200"
                                  />
                                ) : (
                                  <span className="text-gray-500">{p.number}</span>
                                )}
                              </td>
                              <td className="py-2.5 font-sans">
                                {isEditing ? (
                                  <input
                                    type="text"
                                    value={editName}
                                    onChange={e => setEditName(e.target.value)}
                                    className="w-full bg-black border border-gray-900 p-0.5 text-xs text-gray-200"
                                  />
                                ) : (
                                  <span className="text-gray-300 font-medium">{p.name}</span>
                                )}
                              </td>
                              <td className="py-2.5 text-gray-400">
                                {isEditing ? (
                                  <select
                                    value={editPos}
                                    onChange={e => setEditPos(e.target.value)}
                                    className="bg-black border border-gray-900 p-0.5 text-[10px] text-gray-200"
                                  >
                                    <option value="GK">GK</option>
                                    <option value="DF">DF</option>
                                    <option value="MF">MF</option>
                                    <option value="FW">FW</option>
                                  </select>
                                ) : (
                                  p.position
                                )}
                              </td>
                              <td className="py-2.5">
                                <button
                                  onClick={() => toggleStarting(p.id, 'B')}
                                  className={`text-[9px] uppercase px-1.5 py-0.5 rounded font-mono font-bold ${
                                    p.isStarting 
                                      ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' 
                                      : 'bg-gray-900 text-gray-600 border border-gray-800'
                                  }`}
                                >
                                  {p.isStarting ? 'Starting' : 'Bench'}
                                </button>
                              </td>
                              <td className="py-2.5 text-right">
                                <div className="flex justify-end gap-1.5">
                                  {isEditing ? (
                                    <>
                                      <button onClick={() => saveEdit('B')} className="p-1 bg-emerald-500/20 border border-emerald-500/30 rounded text-emerald-400 hover:bg-emerald-500/30">
                                        <Save size={10} />
                                      </button>
                                      <button onClick={() => setEditingPlayerId(null)} className="p-1 bg-gray-900 border border-gray-850 rounded text-gray-400 hover:text-gray-200">
                                        <X size={10} />
                                      </button>
                                    </>
                                  ) : (
                                    <>
                                      <button onClick={() => startEdit(p)} className="p-1 hover:text-emerald-400 text-gray-500">
                                        <Edit3 size={11} />
                                      </button>
                                      <button onClick={() => deletePlayer(p.id, 'B')} className="p-1 hover:text-red-400 text-gray-600">
                                        <Trash size={11} />
                                      </button>
                                    </>
                                  )}
                                </div>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>

              </div>
            </div>
          )}

          {/* STEP 5: Success Summary */}
          {step === 5 && (
            <div className="w-full max-w-2xl mx-auto flex flex-col gap-6 animate-fade-in text-center">
              {successMsg ? (
                <div className="bg-[#111a12]/30 border border-emerald-500/20 rounded-2xl p-8 flex flex-col items-center">
                  <div className="h-16 w-16 bg-emerald-500/10 border border-emerald-500/20 rounded-full flex items-center justify-center mb-4">
                    <CheckCircle className="text-emerald-400" size={32} />
                  </div>
                  <h3 className="text-lg font-bold text-gray-200 mb-2">Match Pre-Configuration Ready</h3>
                  <p className="text-xs text-gray-500 font-mono max-w-md mb-6">{successMsg}</p>
                  
                  <div className="w-full text-left bg-black/60 border border-gray-900 p-4 rounded-xl font-mono text-[10px] text-gray-400 space-y-1 max-h-[180px] overflow-y-auto custom-scrollbar">
                    <span className="text-emerald-400 font-bold block mb-2 border-b border-gray-900 pb-1">Stored Meta Structure</span>
                    <div>match_id: {`match-${Date.now()}`}</div>
                    <div>fixture_name: {selectedFixture?.name || `${teamAName} vs ${teamBName}`}</div>
                    <div>competition: {selectedFixture?.competition || 'Friendly Match'}</div>
                    <div>video: {videoFile?.name || 'mock_cam1.mp4'}</div>
                    <div>team_a: {teamAName} ({teamAFormation}) · {teamAPlayers.length} players</div>
                    <div>team_b: {teamBName} ({teamBFormation}) · {teamBPlayers.length} players</div>
                  </div>

                  <button
                    onClick={resetForm}
                    className="mt-6 px-6 py-2.5 bg-emerald-500 hover:bg-emerald-400 text-black font-bold rounded-xl text-xs transition-colors"
                  >
                    Setup Another Match
                  </button>
                </div>
              ) : (
                <div className="bg-black/30 border border-gray-900 rounded-2xl p-8 flex flex-col items-center">
                  <Activity className="text-emerald-500 animate-pulse mb-4" size={40} />
                  <h3 className="text-lg font-bold text-gray-200 mb-2">Confirm Setup Details</h3>
                  <p className="text-xs text-gray-500 max-w-md mb-6">
                    Review and finalize the local configuration. Saving will write match lineups to the workspace database.
                  </p>
                  
                  <button
                    id="save-setup-btn"
                    onClick={handleSaveSetup}
                    className="px-8 py-3 bg-emerald-500 hover:bg-emerald-400 text-black font-bold rounded-xl text-xs transition-all tracking-wide flex items-center gap-2 shadow-lg shadow-emerald-950/20"
                  >
                    <CheckCircle size={14} /> Save Configuration
                  </button>
                </div>
              )}
            </div>
          )}

        </div>

        {/* ── Navigation Buttons ────────────────────────────────────────── */}
        <div className="flex items-center justify-between border-t border-gray-900/60 pt-6 mt-4">
          <button
            onClick={() => setStep(step - 1)}
            disabled={step === 1 || !!successMsg}
            className={`flex items-center gap-1.5 px-4 py-2 border rounded-xl text-xs font-mono font-bold transition-all ${
              step === 1 || successMsg
                ? 'border-gray-900 text-gray-700 cursor-not-allowed'
                : 'border-gray-800 text-gray-400 hover:text-gray-250 hover:bg-white/[0.01]'
            }`}
          >
            <ArrowLeft size={14} /> Back
          </button>
          
          <button
            id={`setup-next-step-${step}`}
            onClick={() => setStep(step + 1)}
            disabled={
              (step === 1 && !videoFile) ||
              (step === 2 && (!selectedFixture || !lineupReady)) ||
              step === 5 ||
              !!successMsg
            }
            className={`flex items-center gap-1.5 px-6 py-2.5 rounded-xl text-xs font-mono font-bold transition-all ${
              (step === 1 && !videoFile) ||
              (step === 2 && (!selectedFixture || !lineupReady)) ||
              step === 5 ||
              successMsg
                ? 'bg-gray-950 text-gray-650 border border-gray-900 cursor-not-allowed'
                : 'bg-emerald-500 hover:bg-emerald-400 text-black shadow-md shadow-emerald-950/20'
            }`}
          >
            Next <ArrowRight size={14} />
          </button>
        </div>

      </div>
    </div>
  );
}
