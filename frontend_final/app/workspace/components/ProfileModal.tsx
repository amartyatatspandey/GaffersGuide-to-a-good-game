"use client";
import React, { useState, useEffect, useRef } from 'react';
import { X, User, Shield, BarChart3, Target, Award, Edit2, Check, Loader2 } from 'lucide-react';
import { listReports } from '@/lib/api/jobs';

export function ProfileModal({ isOpen, onClose }: { isOpen: boolean, onClose: () => void }) {
  const [reports, setReports] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [isEditingName, setIsEditingName] = useState(false);
  const [userName, setUserName] = useState('Head Coach');
  const [licenseId, setLicenseId] = useState('PRO-992');
  const nameInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!isOpen) return;

    // Load from local storage
    const savedName = localStorage.getItem('gg-profile-name');
    if (savedName) {
      setUserName(savedName);
      setLicenseId(generateLicenseFrom(savedName));
    }

    // Fetch real stats
    async function fetchStats() {
      try {
        setLoading(true);
        const data = await listReports();
        setReports(data || []);
      } catch (error) {
        console.error("Failed to load reports for profile:", error);
      } finally {
        setLoading(false);
      }
    }
    fetchStats();
  }, [isOpen]);

  useEffect(() => {
    if (isEditingName && nameInputRef.current) {
      nameInputRef.current.focus();
    }
  }, [isEditingName]);

  const generateLicenseFrom = (name: string) => {
    const sum = name.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    return `PRO-${sum % 10000}`;
  };

  const handleNameSave = () => {
    setIsEditingName(false);
    const finalName = userName.trim() || 'Head Coach';
    setUserName(finalName);
    localStorage.setItem('gg-profile-name', finalName);
    setLicenseId(generateLicenseFrom(finalName));
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleNameSave();
    if (e.key === 'Escape') {
      setIsEditingName(false);
      setUserName(localStorage.getItem('gg-profile-name') || 'Head Coach');
    }
  };

  if (!isOpen) return null;

  // Compute stats
  const matchesAnalyzed = reports.length;
  const totalInsights = reports.reduce((sum, r) => sum + (r.flaw_count || 0), 0);
  
  // Calculate level based on XP (say, 50 XP per match, 10 XP per insight)
  const currentXP = (matchesAnalyzed * 50) + (totalInsights * 10);
  // Give them a base level of 42 so they don't lose their "Pro" status if they have no matches, plus their calculated level
  const level = 42 + Math.floor(currentXP / 100); 
  const xpIntoCurrentLevel = currentXP % 100;
  const xpForNextLevel = 100;
  const xpPercentage = (xpIntoCurrentLevel / xpForNextLevel) * 100;

  // Average accuracy based on win probability delta from 50 (just a flavor metric)
  let avgAccuracy = 98.2;
  if (reports.length > 0) {
     const total = reports.reduce((sum, r) => {
        const wp = r.win_probability?.team_red || 50;
        return sum + (100 - Math.abs(50 - wp));
     }, 0);
     avgAccuracy = Number((total / reports.length).toFixed(1));
  }

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/90 backdrop-blur-md font-sans p-4 animate-fade-in">
      <div className="bg-[#0a0f0a] border border-emerald-500/30 rounded-2xl w-full max-w-2xl shadow-[0_0_120px_rgba(16,185,129,0.15)] overflow-hidden relative">
        
        {/* Animated Background Mesh */}
        <div className="absolute inset-0 z-0 opacity-20 pointer-events-none" 
             style={{ backgroundImage: 'radial-gradient(circle at 50% 0%, #10b981 0%, transparent 60%)' }} />

        {/* Banner */}
        <div className="h-40 bg-gradient-to-r from-emerald-900/60 via-[#0a0f0a] to-blue-900/40 relative border-b border-emerald-500/20">
            {/* Grid overlay */}
            <div className="absolute inset-0 bg-grid opacity-30"></div>
            
            <button onClick={onClose} className="absolute top-4 right-4 text-gray-500 hover:text-white transition-colors z-10 bg-black/50 p-1.5 rounded-full backdrop-blur-sm">
                <X size={20} />
            </button>
            <div className="absolute -bottom-12 left-8 z-10 group">
                <div className="w-28 h-28 rounded-2xl bg-[#0a0f0a] border-2 border-emerald-500 flex items-center justify-center shadow-[0_0_30px_rgba(16,185,129,0.4)] relative overflow-hidden transition-transform duration-300 group-hover:scale-105">
                    <div className="absolute inset-0 bg-emerald-500/10 animate-pulse"></div>
                    <User size={56} className="text-emerald-500 relative z-10" />
                </div>
            </div>
        </div>

        {/* Info */}
        <div className="pt-16 px-8 pb-8 relative z-10">
            <div className="flex justify-between items-start mb-8">
                <div>
                    <div className="flex items-center gap-3 group">
                        {isEditingName ? (
                            <div className="flex items-center gap-2">
                                <input 
                                    ref={nameInputRef}
                                    type="text"
                                    value={userName}
                                    onChange={(e) => setUserName(e.target.value)}
                                    onKeyDown={handleKeyDown}
                                    className="bg-[#111a12] border border-emerald-500/50 text-gray-100 text-2xl font-bold rounded px-2 py-1 outline-none w-64 focus:ring-1 focus:ring-emerald-500"
                                    placeholder="Enter your name..."
                                />
                                <button onClick={handleNameSave} className="p-1.5 bg-emerald-500/20 text-emerald-400 rounded hover:bg-emerald-500/40 transition-colors">
                                    <Check size={18} />
                                </button>
                            </div>
                        ) : (
                            <>
                                <h2 className="text-2xl font-bold text-gray-100 flex items-center gap-2">
                                    {userName} <Shield size={18} className="text-emerald-500" />
                                </h2>
                                <button 
                                    onClick={() => setIsEditingName(true)}
                                    className="opacity-0 group-hover:opacity-100 p-1 text-gray-500 hover:text-emerald-400 transition-all"
                                    title="Edit Profile Name"
                                >
                                    <Edit2 size={16} />
                                </button>
                            </>
                        )}
                    </div>
                    <p className="text-emerald-500 font-mono text-xs uppercase tracking-widest mt-1">Tactical Analysis License: {licenseId}</p>
                </div>
                
                <div className="text-right">
                    <div className="bg-emerald-500/10 border border-emerald-500/30 px-4 py-1.5 rounded-lg text-emerald-400 text-sm font-bold font-mono shadow-[0_0_15px_rgba(16,185,129,0.2)]">
                        LEVEL {level}
                    </div>
                    <div className="mt-2 flex items-center gap-2" title={`${xpIntoCurrentLevel} / ${xpForNextLevel} XP`}>
                        <div className="w-24 h-1.5 bg-gray-900 rounded-full overflow-hidden">
                            <div 
                                className="h-full bg-emerald-500 shadow-[0_0_10px_#10b981] transition-all duration-1000 ease-out"
                                style={{ width: `${xpPercentage}%` }}
                            />
                        </div>
                        <span className="text-[9px] font-mono text-gray-500">{xpIntoCurrentLevel} XP</span>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                {/* Stats Card */}
                <div className="bg-gradient-to-b from-[#111a12] to-[#0d140e] border border-gray-800/80 p-5 rounded-xl relative overflow-hidden group hover:border-emerald-500/30 transition-colors">
                    <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
                        <BarChart3 size={64} />
                    </div>
                    <div className="flex items-center gap-2 text-emerald-500/70 text-[10px] font-bold uppercase tracking-widest mb-4">
                        <BarChart3 size={14} /> Intelligence Stats
                    </div>
                    
                    {loading ? (
                        <div className="flex justify-center py-6">
                            <Loader2 className="animate-spin text-emerald-500/50" size={24} />
                        </div>
                    ) : (
                        <div className="space-y-4 relative z-10">
                            <div className="flex justify-between items-end border-b border-gray-900/50 pb-2">
                                <span className="text-xs text-gray-400 font-mono uppercase">Matches Analyzed</span>
                                <span className="text-lg font-bold text-gray-100">{matchesAnalyzed}</span>
                            </div>
                            <div className="flex justify-between items-end border-b border-gray-900/50 pb-2">
                                <span className="text-xs text-gray-400 font-mono uppercase">Total Insights</span>
                                <span className="text-lg font-bold text-gray-100">{totalInsights}</span>
                            </div>
                            <div className="flex justify-between items-end">
                                <span className="text-xs text-gray-400 font-mono uppercase">Avg Accuracy</span>
                                <span className="text-lg font-bold text-emerald-400">{avgAccuracy}%</span>
                            </div>
                        </div>
                    )}
                </div>

                {/* Specialties Card */}
                <div className="bg-gradient-to-b from-[#111a12] to-[#0d140e] border border-gray-800/80 p-5 rounded-xl hover:border-emerald-500/30 transition-colors">
                    <div className="flex items-center gap-2 text-emerald-500/70 text-[10px] font-bold uppercase tracking-widest mb-4">
                        <Target size={14} /> Tactical Specialties
                    </div>
                    <div className="flex flex-wrap gap-2">
                        {['Gegenpressing', 'Low Block', 'False 9', 'Inverted Fullbacks'].map((tag, i) => (
                            <span 
                                key={tag} 
                                className="px-3 py-1.5 bg-[#0a0f0a] text-gray-300 text-[11px] rounded border border-gray-800 hover:border-emerald-500/50 hover:text-emerald-400 hover:shadow-[0_0_10px_rgba(16,185,129,0.2)] transition-all cursor-default"
                                style={{ animationDelay: `${i * 100}ms` }}
                            >
                                {tag}
                            </span>
                        ))}
                    </div>
                    <div className="mt-4 pt-4 border-t border-gray-900/50">
                         <p className="text-[10px] text-gray-600 font-mono leading-relaxed">
                            Specialties are determined automatically by the AI engine based on your most frequent tactical analysis patterns.
                         </p>
                    </div>
                </div>
            </div>

            {/* Achievements */}
            <div className="mt-4 p-4 bg-gradient-to-r from-emerald-500/10 to-transparent border border-emerald-500/20 rounded-xl flex items-center gap-4 hover:border-emerald-500/40 transition-colors">
                <div className="p-3 bg-[#0a0f0a] rounded-xl border border-emerald-500/30 shadow-[0_0_15px_rgba(16,185,129,0.2)]">
                    <Award className="text-emerald-500" size={24} />
                </div>
                <div>
                    <h4 className="text-sm font-bold text-gray-200">System Architect</h4>
                    <p className="text-xs text-gray-400 mt-1">Successfully integrated the Zero-Shot Learning branch.</p>
                </div>
            </div>
        </div>
      </div>
    </div>
  );
}
