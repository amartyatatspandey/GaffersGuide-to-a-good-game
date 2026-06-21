"use client";
import React, { useState, useRef, useEffect, useMemo } from "react";
import { 
  Bot, 
  User, 
  Send, 
  Save, 
  CheckCircle, 
  Loader2, 
  Activity,
  Zap,
  TrendingUp,
  Shield,
  MessageSquare,
  Clock,
  ChevronRight,
  Maximize2,
  Play,
  Film
} from "lucide-react";
import VideoHUD from "./VideoHUD";
import RadarWidget from "./radar/RadarWidget";
import { InsightCard } from "./InsightCard";
import { saveTacticalReport } from "@/lib/api/reports";
import { getApiBaseUrl, getAuthHeaders } from "@/lib/apiBase";
import { SaveResultsModal } from "./SaveResultsModal";

export default function TacticalDashboard({ 
  job, 
  coachAdvice, 
  coachError,
  useAltNames = false,
  dictionary = {},
  aiPromptOverride = null,
  setAiPromptOverride
}: { 
  job: any;
  coachAdvice: any;
  coachError: string | null;
  useAltNames?: boolean;
  dictionary?: Record<string, string>;
  aiPromptOverride?: string | null;
  setAiPromptOverride?: (val: string | null) => void;
}) {
  const [activeIndex, setActiveIndex] = useState(0);
  const [promptInput, setPromptInput] = useState("");
  const [chatHistory, setChatHistory] = useState<any[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'success' | 'error' | 'rendering'>('idle');
  const [llmEngine, setLlmEngine] = useState<"local" | "cloud">("cloud");
  const hasAutoPrompted = useRef(false);

  // Filters State
  const [filterType, setFilterType] = useState<'category' | 'player' | null>(null);
  const [filterVal, setFilterVal] = useState<string | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const syncEngine = () => {
      const pref = localStorage.getItem("gaffer-engine-type") === "cloud" ? "cloud" : "local";
      setLlmEngine(pref);
    };
    syncEngine();
    window.addEventListener("gaffer-engine-changed", syncEngine);
    return () => window.removeEventListener("gaffer-engine-changed", syncEngine);
  }, []);

  // Sync prompts and seek commands from external views
  useEffect(() => {
    if (aiPromptOverride) {
      setPromptInput(aiPromptOverride);
      setAiPromptOverride?.(null);
      setTimeout(() => {
        const botElement = document.getElementById("ai-chat-assistant");
        if (botElement) {
          botElement.scrollIntoView({ behavior: 'smooth' });
        }
      }, 200);
    }
  }, [aiPromptOverride, setAiPromptOverride]);

  useEffect(() => {
    const onPlayClipEvent = (e: Event) => {
      const detail = (e as CustomEvent).detail;
      if (detail && typeof detail.startTimeS === 'number') {
        handlePlayClip(detail.startTimeS);
      }
    };
    window.addEventListener("gaffer-play-clip", onPlayClipEvent);
    return () => window.removeEventListener("gaffer-play-clip", onPlayClipEvent);
  }, []);

  useEffect(() => {
    // Automatically prompt to save when the analysis fully completes
    if (coachAdvice?.advice_items?.length > 0 && !hasAutoPrompted.current && !job?.isHistorical) {
      hasAutoPrompted.current = true;
      setIsModalOpen(true);
    }
  }, [coachAdvice, job]);

  const videoRef = useRef<HTMLVideoElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const handlePlayClip = (startTimeS: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = startTimeS;
      void videoRef.current.play();
    }
  };

  // Robust video source binding
  const videoSrc = useMemo(() => {
    if (job?.file) {
      return URL.createObjectURL(job.file);
    }
    return job?.videoUrl || job?.video_url || "";
  }, [job]);

  // Parse the evidence string when summary_data is null
  // Evidence format: "Tactical Power: Red X vs Blue Y. Win Probability: Red X% | Blue Y%..."
  const parseEvidenceMetrics = (evidence: string) => {
    const parsed: any = { team_0: {}, team_1: {}, win_probability: {} };
    const tp = evidence.match(/Tactical Power: Red ([\d.]+) vs Blue ([\d.]+)/);
    if (tp) {
      parsed.team_0.tactical_power = parseFloat(tp[1]);
      parsed.team_1.tactical_power = parseFloat(tp[2]);
      parsed.team_red_score = parseFloat(tp[1]);
      parsed.team_blue_score = parseFloat(tp[2]);
    }
    const wp = evidence.match(/Win Probability: Red ([\d.]+)%[^|]*\| Blue ([\d.]+)%/);
    if (wp) {
      parsed.win_probability.team_red = parseFloat(wp[1]);
      parsed.win_probability.team_blue = parseFloat(wp[2]);
      parsed.team_0.win_prob = parseFloat(wp[1]);
      parsed.team_1.win_prob = parseFloat(wp[2]);
    }
    const cp = evidence.match(/Compactness: Red ([\d.]+) \/ Blue ([\d.]+)/);
    if (cp) {
      parsed.team_0.compactness = parseFloat(cp[1]);
      parsed.team_1.compactness = parseFloat(cp[2]);
    }
    const ts = evidence.match(/Transition Speed: Red ([\d.]+) \/ Blue ([\d.]+)/);
    if (ts) {
      parsed.team_0.transition_speed = parseFloat(ts[1]);
      parsed.team_1.transition_speed = parseFloat(ts[2]);
    }
    return parsed;
  };

  // Global summary metrics — prefer summary_data, fall back to parsing evidence string
  const globalMetrics = useMemo(() => {
    const summaryItem = coachAdvice?.advice_items?.find((i: any) => i.flaw === 'Match Summary');
    if (!summaryItem) return {};
    if (summaryItem.summary_data?.team_0) return summaryItem.summary_data;
    return parseEvidenceMetrics(summaryItem.evidence || '');
  }, [coachAdvice]);

  const timeline = useMemo(() => {
    if (!coachAdvice?.advice_items) return [];
    const summaryItem = coachAdvice.advice_items.find((i: any) => i.flaw === 'Match Summary');
    const summaryMetrics = summaryItem?.summary_data?.team_0
      ? summaryItem.summary_data
      : parseEvidenceMetrics(summaryItem?.evidence || '');
    return coachAdvice.advice_items.map((item: any) => ({
      time: item.timestamp || (item.frame_idx !== undefined ? `F:${item.frame_idx}` : "00:00"),
      title: item.flaw || "Tactical Phase",
      summary: item.tactical_instruction || item.evidence || "System analysis in progress...",
      minute: item.minute || "0",
      // blue = team_1, red = team_0 (consistent with InsightCard teamMetrics mapping)
      blueTeam: {
        payload: item.flaw === 'Match Summary'
          ? item.evidence
          : (item.team === 'team_1' ? item.tactical_instruction : item.evidence || ""),
        keywords: []
      },
      redTeam: {
        payload: item.flaw === 'Match Summary'
          ? item.evidence
          : (item.team === 'team_0' ? item.tactical_instruction : item.evidence || ""),
        keywords: []
      },
      // Always use global summary metrics so numbers are never null
      metrics: summaryMetrics,
      evidenceClips: item.evidence_clips || [],
      threatContext: item.threat_context || null,
      eventCountSummary: item.event_count_summary || null,
    }));
  }, [coachAdvice]);

  // Compute filtered timeline
  const filteredTimeline = useMemo(() => {
    if (!filterType || !filterVal) return timeline;

    return timeline.filter((item: any) => {
      if (filterType === 'category') {
        const textToSearch = `${item.title} ${item.summary}`.toLowerCase();
        return textToSearch.includes(filterVal.toLowerCase()) || 
               item.title.toLowerCase().includes(filterVal.toLowerCase());
      }
      if (filterType === 'player') {
        const pidStr = filterVal;
        const playerRemap = dictionary[`P${pidStr}`] || `Player ${pidStr}`;
        const textToSearch = `${item.title} ${item.summary}`.toLowerCase();
        
        const inText = textToSearch.includes(`player ${pidStr}`) || 
                       textToSearch.includes(`p${pidStr}`) ||
                       textToSearch.includes(playerRemap.toLowerCase());
                       
        const inThreat = item.threatContext?.top_threats?.some((t: any) => t.player_id.toString() === pidStr);
        const inClips = item.evidenceClips?.some((c: any) => c.highlight_player_ids?.includes(parseInt(pidStr)));

        return inText || inThreat || inClips;
      }
      return true;
    });
  }, [timeline, filterType, filterVal, dictionary]);

  const activeSlide = timeline[activeIndex];

  const handleSaveBoth = async () => {
    if (!coachAdvice) return;
    setIsSaving(true);
    try {
      // 1. Save the report to the backend
      await saveTacticalReport(job.jobId, coachAdvice, job?.file?.name);

      // 2. Download the video using fetch so we can send auth headers
      setSaveStatus('rendering');
      const base = getApiBaseUrl();
      const downloadUrl = `${base}/api/v1/elite/jobs/${job.jobId}/video/download`;
      const response = await fetch(downloadUrl, { headers: getAuthHeaders() });

      if (!response.ok) {
        throw new Error(`Video download failed: ${response.status}`);
      }

      const blob = await response.blob();
      const blobUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = blobUrl;
      link.setAttribute("download", `GaffersGuide_TacticalRadar_${job.jobId}.mp4`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      setTimeout(() => URL.revokeObjectURL(blobUrl), 5000);

      setSaveStatus('success');
      setTimeout(() => {
        setSaveStatus('idle');
        setIsModalOpen(false);
      }, 2000);
    } catch (err) {
      console.error('Save both failed:', err);
      setSaveStatus('error');
      setTimeout(() => setSaveStatus('idle'), 3000);
    } finally {
      setIsSaving(false);
    }
  };

  const handleSaveReportOnly = async () => {
    if (!coachAdvice) return;
    setIsSaving(true);
    try {
      await saveTacticalReport(job.jobId, coachAdvice, job?.file?.name);
      setSaveStatus('success');
      setTimeout(() => {
        setSaveStatus('idle');
        setIsModalOpen(false);
      }, 2000);
    } catch (err) {
      setSaveStatus('error');
      setTimeout(() => setSaveStatus('idle'), 3000);
    } finally {
      setIsSaving(false);
    }
  };

  const handleSendPrompt = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!promptInput.trim()) return;

    const userMsg = { id: Date.now(), role: "user", text: promptInput };
    setChatHistory((prev) => [...prev, userMsg]);
    
    let finalMessage = promptInput;
    if (useAltNames && Object.keys(dictionary).length > 0) {
      const translationNotes = Object.entries(dictionary)
        .filter(([_, value]) => value && value.trim())
        .map(([key, value]) => `* "${value}" means "${key}"`)
        .join("\n");
      if (translationNotes) {
        finalMessage += `\n\n[Coaching Dictionary / Alt Name Translations]:\n${translationNotes}`;
      }
    }

    setPromptInput("");
    setIsTyping(true);

    try {
      const base = getApiBaseUrl();
      const response = await fetch(`${base}/api/v1/chat`, {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          ...getAuthHeaders()
        },
        body: JSON.stringify({
          message: finalMessage,
          job_id: job?.jobId,
          llm_engine: llmEngine,
          context: {
            flaw: activeSlide?.title,
            analysis: activeSlide?.summary,
            match_summary: coachAdvice?.advice_items?.find((i: any) => i.flaw === 'Match Summary')?.evidence
          }
        }),
      });

      if (!response.ok) {
        throw new Error(`Chat API failed: ${response.status}`);
      }

      const data = await response.json();
      setChatHistory((prev) => [...prev, { 
        id: Date.now() + 1, 
        role: "assistant", 
        text: data.reply || data.response || "No response from tactical engine.",
        evidence: data.evidence || null
      }]);
    } catch (err) {
      console.error(err);
      setChatHistory((prev) => [...prev, { 
        id: Date.now() + 1, 
        role: "assistant", 
        text: "Tactical engine temporarily offline. Verify backend connectivity." 
      }]);
    } finally {
      setIsTyping(false);
    }
  };

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [chatHistory]);

  return (
    <div className="flex flex-col h-full w-full bg-[#050805] font-sans text-gray-200 overflow-y-auto [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-thumb]:bg-gray-800">
      
      {/* SECTION 1: TOP ROW (Timeline | Video | Radar) */}
      <div className="grid grid-cols-[0.9fr_2fr_1.8fr] h-[500px] min-h-[500px] w-full gap-4 p-4">
        
        {/* Timeline (0.9fr) */}
        <div className="flex flex-col bg-[#0a0f0a] border border-gray-900 rounded-xl overflow-hidden shadow-2xl">
          <div className="p-4 border-b border-gray-900 flex justify-between items-center bg-[#111a12]/20">
            <h2 className="text-[10px] font-bold font-mono uppercase tracking-[0.3em] text-gray-500">Match Events</h2>
            <button onClick={() => setIsModalOpen(true)} className="text-gray-500 hover:text-emerald-500 transition-all p-1 hover:bg-emerald-500/10 rounded">
              <Save size={14} />
            </button>
          </div>
          
          {/* Clip Filter Section */}
          <div className="p-3 border-b border-gray-900 bg-black/10 space-y-2">
            <div className="flex items-center justify-between text-[9px] font-mono text-gray-500">
              <span className="uppercase tracking-widest font-bold">Filters & Toggles</span>
              <button 
                onClick={() => {
                  setFilterType(null);
                  setFilterVal(null);
                }}
                className="text-[9px] hover:text-emerald-400 underline transition-colors"
              >
                Clear
              </button>
            </div>
            
            <div className="flex gap-2 text-[9px] font-mono">
              <div className="flex items-center gap-1 text-gray-500">
                <span className={`h-1.5 w-1.5 rounded-full ${useAltNames ? 'bg-emerald-500' : 'bg-gray-800'}`} />
                <span>Alt Names</span>
              </div>
              <div className="flex items-center gap-1 text-gray-500">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
                <span>Analysis Data</span>
              </div>
            </div>

            <div className="flex flex-wrap gap-1 pt-1 max-h-[85px] overflow-y-auto [&::-webkit-scrollbar]:w-1 [&::-webkit-scrollbar-thumb]:bg-gray-800">
              {['Goal', 'Shot', 'Defense', 'Recovery', 'Possession'].map(cat => {
                const isSelected = filterType === 'category' && filterVal === cat;
                return (
                  <button
                    key={cat}
                    onClick={() => {
                      setFilterType(isSelected ? null : 'category');
                      setFilterVal(isSelected ? null : cat);
                    }}
                    className={`px-2 py-0.5 rounded text-[8px] font-mono uppercase font-bold transition-all border ${
                      isSelected 
                        ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30' 
                        : 'bg-transparent text-gray-600 border-gray-900 hover:text-gray-400 hover:border-gray-800'
                    }`}
                  >
                    {useAltNames && dictionary[cat] ? dictionary[cat] : cat}
                  </button>
                );
              })}

              {[1, 2, 3, 4, 5, 6, 7, 8].map(pid => {
                const playerLabel = useAltNames && dictionary[`P${pid}`] ? dictionary[`P${pid}`] : `P${pid}`;
                const isSelected = filterType === 'player' && filterVal === pid.toString();
                return (
                  <button
                    key={pid}
                    onClick={() => {
                      setFilterType(isSelected ? null : 'player');
                      setFilterVal(isSelected ? null : pid.toString());
                    }}
                    className={`px-2 py-0.5 rounded text-[8px] font-mono uppercase font-bold transition-all border ${
                      isSelected 
                        ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30' 
                        : 'bg-transparent text-gray-600 border-gray-900 hover:text-gray-400 hover:border-gray-800'
                    }`}
                  >
                    {playerLabel}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-3 space-y-2 [&::-webkit-scrollbar]:w-1 [&::-webkit-scrollbar-thumb]:bg-gray-800">
            {filteredTimeline.map((data: any, i: number) => {
              const originalIndex = timeline.findIndex((t: any) => t.title === data.title && t.time === data.time);
              const isCurrentActive = originalIndex === activeIndex;
              return (
                <button
                  key={i}
                  onClick={() => setActiveIndex(originalIndex)}
                  className={`w-full p-4 text-left rounded-xl transition-all border ${
                    isCurrentActive ? "border-emerald-500/50 bg-emerald-500/10 shadow-[0_0_20px_rgba(16,185,129,0.05)]" : "border-gray-900/50 bg-transparent hover:bg-gray-900/50"
                  }`}
                >
                  <div className="flex items-center gap-2 mb-1.5">
                    <Clock size={10} className={isCurrentActive ? 'text-emerald-500' : 'text-gray-600'} />
                    <span className={`text-[10px] font-mono ${isCurrentActive ? 'text-emerald-500 font-bold' : 'text-gray-600'}`}>{data.time}</span>
                  </div>
                  <div className={`text-xs font-bold tracking-tight ${isCurrentActive ? 'text-white' : 'text-gray-500'}`}>{data.title}</div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Video HUD (2fr) */}
        <div className="flex flex-col bg-[#0a0f0a] border border-gray-900 rounded-xl overflow-hidden shadow-2xl group">
          <div className="p-3 border-b border-gray-900 flex justify-between items-center bg-[#111a12]/10">
            <div className="flex items-center gap-2">
              <div className="h-2 w-2 rounded-full bg-red-600 animate-pulse" />
              <span className="text-[10px] font-bold font-mono uppercase tracking-widest text-gray-500">Live Telemetry Feed</span>
            </div>
            <Maximize2 size={12} className="text-gray-700 hover:text-gray-400 cursor-pointer" />
          </div>
          <div className="flex-1 min-h-0 bg-black">
             <VideoHUD
               videoRef={videoRef}
               videoSrc={videoSrc}
               jobId={job?.jobId ?? null}
               status={job?.tracking ? "completed" : "pending"}
               onDownload={() => {}}
               trackingData={job?.tracking ?? null}
               useAltNames={useAltNames}
               dictionary={dictionary}
             />
          </div>
        </div>

        {/* Tactical Radar (1.8fr) */}
        <div className="flex flex-col bg-[#0a0f0a] border border-gray-900 rounded-xl overflow-hidden shadow-2xl">
          <div className="p-3 border-b border-gray-900 flex justify-between items-center bg-[#111a12]/10">
            <div className="flex items-center gap-2">
              <Zap size={12} className="text-emerald-500" />
              <span className="text-[10px] font-bold font-mono uppercase tracking-widest text-gray-500">Expanded Tactical Radar</span>
            </div>
            <span className="text-[10px] font-mono text-emerald-500/50">105m x 68m</span>
          </div>
          <div className="flex-1 min-h-0">
             <RadarWidget videoRef={videoRef} trackingData={job?.tracking ?? null} />
          </div>
        </div>
      </div>

      {/* SECTION 2: METRICS ROW — sourced from global summary metrics, not per-slide */}
      <div className="grid grid-cols-4 gap-4 px-4 pb-4">
        <div className="bg-[#0a0f0a] border border-gray-900 rounded-xl p-6 flex flex-col gap-2 shadow-lg">
          <div className="flex items-center gap-2 text-[10px] font-bold font-mono uppercase text-gray-600 tracking-widest">
            <Activity size={14} className="text-emerald-500" />
            Tactical Power
          </div>
          <div className="text-3xl font-bold text-emerald-400 tracking-tighter">
            {globalMetrics?.team_0?.tactical_power?.toFixed(1) ?? globalMetrics?.team_red_score?.toFixed(1) ?? "—"}
            <span className="text-xs text-gray-500 ml-1">/ {globalMetrics?.team_1?.tactical_power?.toFixed(1) ?? globalMetrics?.team_blue_score?.toFixed(1) ?? "—"}</span>
          </div>
          <div className="text-[9px] font-mono text-gray-600">Red / Blue (composite)</div>
        </div>
        <div className="bg-[#0a0f0a] border border-gray-900 rounded-xl p-6 flex flex-col gap-2 shadow-lg">
          <div className="flex items-center gap-2 text-[10px] font-bold font-mono uppercase text-gray-600 tracking-widest">
            <TrendingUp size={14} className="text-cyan-500" />
            Win Probability
          </div>
          <div className="text-3xl font-bold text-cyan-400 tracking-tighter">
            {globalMetrics?.win_probability?.team_red ?? globalMetrics?.team_0?.win_prob ?? "—"}
            <span className="text-xs text-gray-600 ml-1">%</span>
          </div>
        </div>
        <div className="bg-[#0a0f0a] border border-gray-900 rounded-xl p-6 flex flex-col gap-2 shadow-lg">
          <div className="flex items-center gap-2 text-[10px] font-bold font-mono uppercase text-gray-600 tracking-widest">
            <Shield size={14} className="text-amber-500" />
            Compactness
          </div>
          <div className="text-3xl font-bold text-amber-400 tracking-tighter">
            {globalMetrics?.team_0?.compactness != null
              ? (globalMetrics.team_0.compactness > 70 ? "High" : globalMetrics.team_0.compactness > 40 ? "Med" : "Low")
              : "—"}
          </div>
        </div>
        <div className="bg-[#0a0f0a] border border-gray-900 rounded-xl p-6 flex flex-col gap-2 shadow-lg">
          <div className="flex items-center gap-2 text-[10px] font-bold font-mono uppercase text-gray-600 tracking-widest">
            <Zap size={14} className="text-purple-500" />
            Transition Speed
          </div>
          <div className="text-3xl font-bold text-purple-400 tracking-tighter">
            {globalMetrics?.team_0?.transition_speed != null
              ? (globalMetrics.team_0.transition_speed > 70 ? "Fast" : globalMetrics.team_0.transition_speed > 40 ? "Norm" : "Slow")
              : "—"}
          </div>
        </div>
      </div>

      {/* SECTION 3: REPORT SECTIONS */}
      <div className="px-4 pb-8">
        <div className="bg-[#0a0f0a] border border-gray-900 rounded-2xl p-8 shadow-2xl">
          <div className="mb-6 flex items-center justify-between border-b border-gray-900 pb-4">
            <h2 className="text-xl font-bold tracking-tight text-white">Match Intelligence Report</h2>
            <div className="text-[10px] font-mono text-gray-600 uppercase tracking-widest">Pipeline Analysis Active</div>
          </div>
          
          {activeSlide ? (
            <InsightCard
              key={activeIndex}
              title={activeSlide.title}
              minute={activeSlide.minute}
              blueTeam={activeSlide.blueTeam}
              redTeam={activeSlide.redTeam}
              metrics={activeSlide.metrics}
              evidenceClips={activeSlide.evidenceClips}
              threatContext={activeSlide.threatContext}
              eventCountSummary={activeSlide.eventCountSummary}
              onPlayClip={handlePlayClip}
              useAltNames={useAltNames}
              dictionary={dictionary}
            />
          ) : (
            <div className="h-48 flex flex-col items-center justify-center gap-2 text-gray-500 text-sm italic font-mono animate-pulse">
              <Loader2 className="animate-spin text-emerald-500" size={24} />
              <span>Synchronizing tactical debrief...</span>
            </div>
          )}
        </div>
      </div>

      {/* SECTION 4: BOTTOM AI ASSISTANT */}
      <div id="ai-chat-assistant" className="px-4 pb-12">
        <div className="bg-[#111a12]/30 border border-emerald-500/20 rounded-2xl overflow-hidden shadow-[0_0_50px_rgba(16,185,129,0.05)]">
          <div className="p-5 border-b border-emerald-500/10 bg-[#111a12]/40 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-emerald-500/20 rounded-lg">
                <Bot size={20} className="text-emerald-500" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-white tracking-tight">Tactical AI Assistant</h3>
                <p className="text-[10px] text-emerald-500/60 font-mono uppercase tracking-widest">Match-Specific Follow-Up Engine</p>
              </div>
            </div>
            <div className="flex items-center gap-4 text-[10px] font-mono text-gray-600">
               <span>Engine: {llmEngine === 'cloud' ? 'Cloud (Gemini / OpenAI)' : 'Ollama / Llama 3'}</span>
               <span className="h-1 w-1 bg-gray-800 rounded-full" />
               <span>Context: Active Phase</span>
            </div>
          </div>
          
          <div className="h-[400px] flex flex-col relative">
            <div ref={scrollRef} className="flex-1 overflow-y-auto p-8 space-y-6 [&::-webkit-scrollbar]:w-1 [&::-webkit-scrollbar-thumb]:bg-gray-800 pb-32">
              {chatHistory.length === 0 && (
                <div className="h-full flex flex-col items-center justify-center space-y-4 opacity-30 text-center">
                  <MessageSquare size={48} className="text-gray-600" />
                  <div className="space-y-1">
                    <p className="text-sm font-bold text-gray-300">Ready for tactical inquiry</p>
                    <p className="text-xs text-gray-500 max-w-xs">Ask about specific player roles, transitional phases, or philosophical adjustments.</p>
                  </div>
                </div>
              )}
              {chatHistory.map((msg) => (
                <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[70%] p-5 rounded-2xl shadow-xl ${
                    msg.role === 'user' 
                      ? 'bg-[#1a241c] border border-emerald-500/30 text-white rounded-tr-none' 
                      : 'bg-[#0a0f0a] border border-gray-800 text-gray-300 rounded-tl-none'
                  }`}>
                    <div className="text-[10px] font-mono uppercase tracking-widest text-gray-600 mb-2">{msg.role === 'user' ? 'Coach' : 'Gaffer Assistant'}</div>
                    <div className="text-sm leading-relaxed">{msg.text}</div>
                    
                    {msg.evidence && (
                      <div className="mt-4 pt-4 border-t border-emerald-500/10 space-y-4 font-sans">
                        {/* Evidence Clips */}
                        {msg.evidence.clips && msg.evidence.clips.length > 0 && (
                          <div className="space-y-2">
                            <div className="flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-wider text-emerald-500/80">
                              <Film size={12} />
                              <span>Supporting Video Evidence</span>
                            </div>
                            <div className="grid grid-cols-1 gap-2">
                              {msg.evidence.clips.map((clip: any, clipIdx: number) => (
                                <button
                                  key={clipIdx}
                                  onClick={() => handlePlayClip(clip.start_time_s)}
                                  className="w-full text-left p-3 rounded-lg bg-black/40 border border-gray-800 hover:border-emerald-500/30 hover:bg-[#111a12]/20 transition-all flex items-center justify-between group"
                                >
                                  <div className="flex-1 min-w-0 pr-3">
                                    <div className="text-xs font-bold text-gray-200 truncate group-hover:text-emerald-400 transition-colors">
                                      {clip.label || clip.event_name || "Evidence Clip"}
                                    </div>
                                    <div className="text-[9px] font-mono text-gray-500 mt-0.5">
                                      Timestamp: {Math.floor(clip.start_time_s / 60)}:{(Math.floor(clip.start_time_s % 60)).toString().padStart(2, '0')} - {Math.floor(clip.end_time_s / 60)}:{(Math.floor(clip.end_time_s % 60)).toString().padStart(2, '0')}
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

                        {/* Threat Players */}
                        {msg.evidence.top_threats && msg.evidence.top_threats.length > 0 && (
                          <div className="space-y-2">
                            <div className="flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-wider text-emerald-500/80">
                              <User size={12} />
                              <span>Key Attacking Threats</span>
                            </div>
                            <div className="grid grid-cols-1 gap-2">
                              {msg.evidence.top_threats.map((threat: any, threatIdx: number) => (
                                <div key={threatIdx} className="p-3 rounded-lg bg-black/30 border border-gray-800/80 space-y-2">
                                  <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-2">
                                      <div className={`h-5 w-5 rounded-full flex items-center justify-center text-[10px] font-bold ${threat.team_id === 'team_0' ? 'bg-red-500/20 text-red-400' : 'bg-blue-500/20 text-blue-400'}`}>
                                        P{threat.player_id}
                                      </div>
                                      <span className="text-xs font-bold text-gray-300 uppercase">
                                        {threat.team_id === 'team_0' ? 'Red Team' : 'Blue Team'}
                                      </span>
                                    </div>
                                    <div className="flex items-center gap-1">
                                      <span className="text-[10px] font-mono text-gray-500">Threat:</span>
                                      <span className="text-xs font-bold text-emerald-400 font-mono">{threat.threat_score.toFixed(1)}</span>
                                    </div>
                                  </div>
                                  
                                  {/* Threat score progress bar */}
                                  <div className="h-1 bg-gray-900 rounded-full overflow-hidden">
                                    <div 
                                      className={`h-full ${threat.team_id === 'team_0' ? 'bg-red-500' : 'bg-blue-500'}`} 
                                      style={{ width: `${threat.threat_score}%` }} 
                                    />
                                  </div>
                                  
                                  {threat.explanation && (
                                    <p className="text-[10px] text-gray-400 leading-normal font-sans">
                                      {threat.explanation}
                                    </p>
                                  )}
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {isTyping && (
                <div className="flex gap-4 items-center animate-pulse">
                  <div className="h-8 w-8 bg-emerald-500/20 rounded-full flex items-center justify-center">
                    <Bot size={14} className="text-emerald-500" />
                  </div>
                  <div className="h-2 w-24 bg-gray-800 rounded-full" />
                </div>
              )}
            </div>

            <div className="absolute bottom-0 left-0 right-0 p-8 bg-gradient-to-t from-[#050805] via-[#050805] to-transparent">
              <form onSubmit={handleSendPrompt} className="relative mx-auto max-w-4xl">
                <input
                  type="text"
                  value={promptInput}
                  onChange={(e) => setPromptInput(e.target.value)}
                  placeholder="Inquire about transitional overload or defensive compactness..."
                  className="w-full bg-[#111a12] border border-emerald-500/20 rounded-2xl py-5 pl-8 pr-16 text-sm text-white focus:border-emerald-500/50 focus:ring-2 focus:ring-emerald-500/10 outline-none transition-all placeholder:text-gray-700 shadow-2xl"
                />
                <button type="submit" className="absolute right-4 top-1/2 -translate-y-1/2 p-3 bg-emerald-500 text-black rounded-xl hover:bg-emerald-400 transition-all shadow-[0_0_20px_rgba(16,185,129,0.3)]">
                  <Send size={18} />
                </button>
              </form>
            </div>
          </div>
        </div>
      </div>

      <SaveResultsModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSaveBoth={handleSaveBoth}
        onSaveReportOnly={handleSaveReportOnly}
        isSaving={isSaving}
        saveStatus={saveStatus}
      />
    </div>
  );
}
