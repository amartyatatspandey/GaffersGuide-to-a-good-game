"use client";
import React, { useState, useEffect } from 'react';
import { BookOpen, Check, RefreshCw, AlertCircle, Info } from 'lucide-react';

interface DictionaryTabProps {
  dictionary: Record<string, string>;
  useAltNames: boolean;
  onUpdateDictionary: (newDict: Record<string, string>) => void;
  onToggleAltNames: (val: boolean) => void;
}

const DEFAULT_ROWS = [
  { key: "team_0", type: "Team", label: "Team A (Red)", defaultAlt: "Red Team" },
  { key: "team_1", type: "Team", label: "Team B (Blue)", defaultAlt: "Blue Team" },
  { key: "Possession", type: "Category", label: "Possession", defaultAlt: "Possession" },
  { key: "No Possession", type: "Category", label: "No Possession", defaultAlt: "No Possession" },
  { key: "Goal", type: "Event", label: "Goal", defaultAlt: "Goal" },
  { key: "Shot", type: "Event", label: "Shot", defaultAlt: "Shot" },
  { key: "Defense", type: "Category", label: "Defense", defaultAlt: "Defense" },
  { key: "Recovery", type: "Event", label: "Recovery", defaultAlt: "Recovery" },
  { key: "Attack", type: "Category", label: "Attack", defaultAlt: "Attack" },
  { key: "P1", type: "Player", label: "Player 1", defaultAlt: "Player 1" },
  { key: "P2", type: "Player", label: "Player 2", defaultAlt: "Player 2" },
  { key: "P3", type: "Player", label: "Player 3", defaultAlt: "Player 3" },
  { key: "P4", type: "Player", label: "Player 4", defaultAlt: "Player 4" },
  { key: "P5", type: "Player", label: "Player 5", defaultAlt: "Player 5" },
  { key: "P6", type: "Player", label: "Player 6", defaultAlt: "Player 6" },
  { key: "P7", type: "Player", label: "Player 7", defaultAlt: "Player 7" },
  { key: "P8", type: "Player", label: "Player 8", defaultAlt: "Player 8" },
  { key: "P9", type: "Player", label: "Player 9", defaultAlt: "Player 9" },
  { key: "P10", type: "Player", label: "Player 10", defaultAlt: "Player 10" },
  { key: "P11", type: "Player", label: "Player 11", defaultAlt: "Player 11" },
  { key: "P12", type: "Player", label: "Player 12", defaultAlt: "Player 12" },
  { key: "P13", type: "Player", label: "Player 13", defaultAlt: "Player 13" },
  { key: "P14", type: "Player", label: "Player 14", defaultAlt: "Player 14" },
  { key: "P15", type: "Player", label: "Player 15", defaultAlt: "Player 15" },
  { key: "P16", type: "Player", label: "Player 16", defaultAlt: "Player 16" },
  { key: "P17", type: "Player", label: "Player 17", defaultAlt: "Player 17" },
  { key: "P18", type: "Player", label: "Player 18", defaultAlt: "Player 18" },
  { key: "P19", type: "Player", label: "Player 19", defaultAlt: "Player 19" },
  { key: "P20", type: "Player", label: "Player 20", defaultAlt: "Player 20" },
];

export function DictionaryTab({
  dictionary,
  useAltNames,
  onUpdateDictionary,
  onToggleAltNames,
}: DictionaryTabProps) {
  const [localDict, setLocalDict] = useState<Record<string, string>>({});
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState<'all' | 'Team' | 'Category' | 'Event' | 'Player'>('all');
  const [saveSuccess, setSaveSuccess] = useState(false);

  useEffect(() => {
    // Populate local dict state with existing dictionary values, fallback to defaultAlt
    const initial: Record<string, string> = {};
    DEFAULT_ROWS.forEach(row => {
      initial[row.key] = dictionary[row.key] !== undefined ? dictionary[row.key] : row.defaultAlt;
    });
    setLocalDict(initial);
  }, [dictionary]);

  const handleInputChange = (key: string, val: string) => {
    setLocalDict(prev => ({
      ...prev,
      [key]: val,
    }));
  };

  const handleSave = () => {
    onUpdateDictionary(localDict);
    setSaveSuccess(true);
    setTimeout(() => setSaveSuccess(false), 2000);
  };

  const handleReset = () => {
    if (window.confirm("Are you sure you want to reset all alt names to default settings?")) {
      const resetDict: Record<string, string> = {};
      DEFAULT_ROWS.forEach(row => {
        resetDict[row.key] = row.defaultAlt;
      });
      setLocalDict(resetDict);
      onUpdateDictionary(resetDict);
    }
  };

  const filteredRows = DEFAULT_ROWS.filter(row => {
    const matchesSearch = row.label.toLowerCase().includes(searchTerm.toLowerCase()) || 
                          row.key.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          (localDict[row.key] || '').toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = filterType === 'all' || row.type === filterType;
    return matchesSearch && matchesFilter;
  });

  return (
    <div className="h-full w-full bg-[#0a0f0a] flex flex-col animate-fade-in font-sans p-8 overflow-y-auto custom-scrollbar">
      {/* Header */}
      <div className="flex items-center justify-between mb-8 border-b border-gray-900 pb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-200 tracking-tight flex items-center gap-3">
            <BookOpen className="text-emerald-500" />
            Category / Descriptor Names Dictionary
          </h1>
          <p className="text-sm text-gray-500 mt-2 font-mono">
            Map telemetry codes and player tracking tags to custom display names for reports, timeline, HUD, and AI analysis.
          </p>
        </div>

        <div className="flex items-center gap-4 bg-[#111a12] border border-emerald-500/20 px-5 py-3 rounded-xl shadow-lg">
          <span className="text-xs font-mono text-gray-400">Alt Names Mode</span>
          <button
            onClick={() => onToggleAltNames(!useAltNames)}
            className={`w-12 h-6 flex items-center rounded-full p-1 cursor-pointer transition-all duration-300 ${
              useAltNames ? 'bg-emerald-500' : 'bg-gray-800'
            }`}
            aria-label="Toggle Alt Names"
          >
            <div
              className={`bg-black w-4 h-4 rounded-full shadow-md transform transition-transform duration-300 ${
                useAltNames ? 'translate-x-6' : 'translate-x-0'
              }`}
            />
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_3fr] gap-8 items-start">
        {/* Left Card: Controls and Info */}
        <div className="flex flex-col gap-6">
          <div className="bg-[#111a12]/30 border border-gray-900 rounded-xl p-5 shadow-md space-y-4">
            <h2 className="text-xs font-bold text-gray-400 font-mono uppercase tracking-widest flex items-center gap-2">
              <Info size={14} className="text-emerald-500" /> System Guide
            </h2>
            <p className="text-xs text-gray-400 leading-relaxed font-sans">
              Remapped names are used contextually. For instance, when <strong className="text-emerald-400">Marcus Rashford</strong> is assigned to <strong className="text-gray-300">P5</strong>:
            </p>
            <ul className="text-xs text-gray-500 space-y-2 font-mono pl-4 list-disc">
              <li>Head badge HUD renders <span className="text-emerald-400/90">Marcus Rashford</span></li>
              <li>Chat foot-notes map P5 requests correctly</li>
              <li>Player statistics display alt names</li>
            </ul>
          </div>

          <div className="bg-[#111a12]/30 border border-gray-900 rounded-xl p-5 shadow-md space-y-4">
            <div className="text-xs font-bold text-gray-400 font-mono uppercase tracking-widest">Controls</div>
            <div className="flex flex-col gap-2">
              <button
                onClick={handleSave}
                className="w-full bg-emerald-600 hover:bg-emerald-500 text-black font-bold py-2.5 px-4 rounded-lg text-sm transition-all flex items-center justify-center gap-2 shadow-lg shadow-emerald-950/20"
              >
                {saveSuccess ? (
                  <>
                    <Check size={16} /> Saved Successfully
                  </>
                ) : (
                  "Apply Changes"
                )}
              </button>
              <button
                onClick={handleReset}
                className="w-full bg-transparent border border-gray-800 hover:border-red-500/50 hover:bg-red-500/5 text-gray-400 hover:text-red-400 font-bold py-2 px-4 rounded-lg text-xs transition-all flex items-center justify-center gap-2"
              >
                <RefreshCw size={12} /> Reset to Defaults
              </button>
            </div>
          </div>
        </div>

        {/* Right Card: Data Grid Table */}
        <div className="bg-[#111a12]/20 border border-gray-900 rounded-2xl p-6 shadow-2xl flex flex-col gap-6">
          {/* Filters Bar */}
          <div className="flex flex-col md:flex-row gap-4 items-center justify-between">
            <div className="flex flex-wrap gap-1 bg-[#0a0f0a] border border-gray-900 rounded-lg p-1 w-full md:w-auto">
              {(['all', 'Team', 'Category', 'Event', 'Player'] as const).map((type) => (
                <button
                  key={type}
                  onClick={() => setFilterType(type)}
                  className={`px-3 py-1.5 rounded-md text-[10px] font-bold font-mono uppercase transition-all ${
                    filterType === type 
                      ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' 
                      : 'text-gray-500 hover:text-gray-300'
                  }`}
                >
                  {type}
                </button>
              ))}
            </div>

            <input
              type="text"
              placeholder="Filter by name or code..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full md:w-64 bg-[#0a0f0a] border border-gray-900 rounded-lg px-4 py-2 text-xs text-gray-300 focus:outline-none focus:border-emerald-500 transition-colors"
            />
          </div>

          {/* Table Container */}
          <div className="overflow-x-auto rounded-xl border border-gray-900 bg-[#0a0f0a]/50">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-gray-900 bg-[#0c120d]/60 font-mono text-gray-500 uppercase tracking-wider text-[10px]">
                  <th className="py-3.5 px-4 font-bold">Descriptor Type</th>
                  <th className="py-3.5 px-4 font-bold">Internal Tag / Code</th>
                  <th className="py-3.5 px-4 font-bold">Default Display Name</th>
                  <th className="py-3.5 px-4 font-bold w-1/3">Remapped Display Name (Alt Name)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-900/40 text-gray-300 font-sans">
                {filteredRows.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="py-8 text-center text-gray-600 font-mono italic">
                      No matching mappings found.
                    </td>
                  </tr>
                ) : (
                  filteredRows.map((row) => (
                    <tr 
                      key={row.key} 
                      className="hover:bg-emerald-500/[0.02] transition-colors"
                    >
                      <td className="py-3 px-4">
                        <span className={`px-2 py-0.5 rounded text-[9px] font-bold font-mono uppercase ${
                          row.type === 'Team' ? 'bg-blue-500/10 text-blue-400 border border-blue-500/10' :
                          row.type === 'Category' ? 'bg-amber-500/10 text-amber-400 border border-amber-500/10' :
                          row.type === 'Event' ? 'bg-purple-500/10 text-purple-400 border border-purple-500/10' :
                          'bg-emerald-500/10 text-emerald-400 border border-emerald-500/10'
                        }`}>
                          {row.type}
                        </span>
                      </td>
                      <td className="py-3 px-4 font-mono text-gray-500">{row.key}</td>
                      <td className="py-3 px-4 font-medium">{row.label}</td>
                      <td className="py-2 px-4">
                        <input
                          type="text"
                          value={localDict[row.key] !== undefined ? localDict[row.key] : ""}
                          onChange={(e) => handleInputChange(row.key, e.target.value)}
                          placeholder={row.defaultAlt}
                          className="w-full bg-[#111a12] border border-gray-900 hover:border-gray-800 focus:border-emerald-500/50 rounded-md px-3 py-1.5 text-xs text-gray-200 outline-none transition-all"
                        />
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          <div className="flex items-center gap-2 text-[10px] text-gray-600 font-mono">
            <AlertCircle size={12} className="text-gray-700" />
            Remember to click &quot;Apply Changes&quot; to compile alterations.
          </div>
        </div>
      </div>
    </div>
  );
}
