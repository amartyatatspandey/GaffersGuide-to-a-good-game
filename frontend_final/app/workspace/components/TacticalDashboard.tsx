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
  Film,
  Award
} from "lucide-react";
import VideoHUD from "./VideoHUD";
import RadarWidget from "./radar/RadarWidget";
import { InsightCard } from "./InsightCard";
import { saveTacticalReport, downloadPdfReport } from "@/lib/api/reports";
import { getApiBaseUrl, getAuthHeaders } from "@/lib/apiBase";
import { SaveResultsModal } from "./SaveResultsModal";
import { getTacticalTimeline } from "@/lib/api/jobs";
import { loadJobMappings, resolvePlayerLabel, resolveTeamLabel } from "@/lib/playerMappingUtils";

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
  const [saveStatus, setSaveStatus] = useState<'idle' | 'success' | 'error' | 'rendering' | 'compiling_pdf'>('idle');
  const [llmEngine, setLlmEngine] = useState<"local" | "cloud">("cloud");
  const [reportMode, setReportMode] = useState<'timeline' | 'executive'>('timeline');
  const hasAutoPrompted = useRef(false);

  // Tactical Timeline States
  const [tacticalTimeline, setTacticalTimeline] = useState<any[]>([]);
  const [activeSegmentIdx, setActiveSegmentIdx] = useState<number | null>(null);
  const [loadingTimeline, setLoadingTimeline] = useState(false);

  useEffect(() => {
    if (!job?.jobId) return;
    setLoadingTimeline(true);
    getTacticalTimeline(job.jobId)
      .then(data => {
        setTacticalTimeline(data);
        if (data.length > 0) {
          setActiveSegmentIdx(0);
        } else {
          setActiveSegmentIdx(null);
        }
      })
      .catch(err => {
        console.error("Failed to load tactical timeline:", err);
      })
      .finally(() => {
        setLoadingTimeline(false);
      });
  }, [job?.jobId]);

  // Filters State
  const [filterType, setFilterType] = useState<'category' | 'player' | null>(null);
  const [filterVal, setFilterVal] = useState<string | null>(null);

  // Player identity mappings (from PlayerMapping page stored in localStorage)
  const savedMappings = useMemo(() => loadJobMappings(job?.jobId), [job?.jobId]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const syncEngine = () => {
      const pref = localStorage.getItem("gaffer-engine-type") === "local" ? "local" : "cloud";
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

  const parseAllMetrics = (evidence: string) => {
    const parsed = {
      team_0: {
        tactical_power: 0,
        win_prob: 0,
        compactness: 0,
        press_resistance: 0,
        width_utilization: 0,
        line_staggering: 0,
        transition_speed: 0,
      },
      team_1: {
        tactical_power: 0,
        win_prob: 0,
        compactness: 0,
        press_resistance: 0,
        width_utilization: 0,
        line_staggering: 0,
        transition_speed: 0,
      }
    };

    if (!evidence) return parsed;

    const matchVal = (regex: RegExp) => {
      const match = evidence.match(regex);
      return match ? [parseFloat(match[1]), parseFloat(match[2])] : null;
    };

    const tp = matchVal(/Tactical Power:\s*(?:Red|team_0)?\s*([\d.]+)\s*vs\s*(?:Blue|team_1)?\s*([\d.]+)/i) ||
               matchVal(/Tactical Power:\s*([\d.]+)\s*\/\s*([\d.]+)/i);
    if (tp) {
      parsed.team_0.tactical_power = tp[0];
      parsed.team_1.tactical_power = tp[1];
    }

    const wp = matchVal(/Win Probability:\s*(?:Red|team_0)?\s*([\d.]+)%?\s*\|?\s*(?:Blue|team_1)?\s*([\d.]+)%/i) ||
               matchVal(/Win Probability:\s*(?:Red|team_0)?\s*([\d.]+)%?\s*vs\s*(?:Blue|team_1)?\s*([\d.]+)%/i);
    if (wp) {
      parsed.team_0.win_prob = wp[0];
      parsed.team_1.win_prob = wp[1];
    }

    const cp = matchVal(/Compactness:\s*(?:Red|team_0)?\s*([\d.]+)\s*[\/\s]\s*(?:Blue|team_1)?\s*([\d.]+)/i);
    if (cp) {
      parsed.team_0.compactness = cp[0];
      parsed.team_1.compactness = cp[1];
    }

    const pr = matchVal(/Press Resistance:\s*(?:Red|team_0)?\s*([\d.]+)\s*[\/\s]\s*(?:Blue|team_1)?\s*([\d.]+)/i);
    if (pr) {
      parsed.team_0.press_resistance = pr[0];
      parsed.team_1.press_resistance = pr[1];
    }

    const wu = matchVal(/Width Utilization:\s*(?:Red|team_0)?\s*([\d.]+)\s*[\/\s]\s*(?:Blue|team_1)?\s*([\d.]+)/i);
    if (wu) {
      parsed.team_0.width_utilization = wu[0];
      parsed.team_1.width_utilization = wu[1];
    }

    const ls = matchVal(/Line Staggering:\s*(?:Red|team_0)?\s*([\d.]+)\s*[\/\s]\s*(?:Blue|team_1)?\s*([\d.]+)/i);
    if (ls) {
      parsed.team_0.line_staggering = ls[0];
      parsed.team_1.line_staggering = ls[1];
    }

    const ts = matchVal(/Transition Speed:\s*(?:Red|team_0)?\s*([\d.]+)\s*[\/\s]\s*(?:Blue|team_1)?\s*([\d.]+)/i);
    if (ts) {
      parsed.team_0.transition_speed = ts[0];
      parsed.team_1.transition_speed = ts[1];
    }

    return parsed;
  };

  const summaryItem = useMemo(() => {
    return coachAdvice?.advice_items?.find((i: any) => i.flaw === 'Match Summary');
  }, [coachAdvice]);

  const parsedAllMetrics = useMemo(() => {
    if (!summaryItem) return null;
    return parseAllMetrics(summaryItem.evidence || '');
  }, [summaryItem]);

  const threatContext = useMemo(() => {
    const found = coachAdvice?.advice_items?.find((i: any) => i.threat_context?.top_threats?.length > 0);
    if (found) return found.threat_context;
    return null;
  }, [coachAdvice]);

  const formatBoldText = (text: string) => {
    const parts = text.split(/\*\*(.*?)\*\*/g);
    return parts.map((part, i) => i % 2 === 1 ? <strong key={i} className="text-white font-bold">{part}</strong> : part);
  };

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

  const handleDownloadPdf = async () => {
    if (!job?.jobId) return;
    setIsSaving(true);
    setSaveStatus('compiling_pdf');
    try {
      const blob = await downloadPdfReport(job.jobId);
      const blobUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = blobUrl;
      link.setAttribute("download", `GaffersGuide_TacticalReport_${job.jobId}.pdf`);
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
      console.error('PDF download failed:', err);
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
                const playerLabel = resolvePlayerLabel(pid, { savedMappings, useAltNames, dictionary });
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
             <RadarWidget videoRef={videoRef} trackingData={job?.tracking ?? null} jobId={job?.jobId ?? null} />
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
          <div className="mb-6 flex flex-col sm:flex-row items-start sm:items-center justify-between border-b border-gray-900 pb-4 gap-4">
            <h2 className="text-xl font-bold tracking-tight text-white">Match Analysis</h2>
            <div className="flex bg-black/40 border border-gray-900 rounded-xl p-0.5">
              <button
                onClick={() => setReportMode('timeline')}
                className={`px-4 py-1.5 rounded-lg text-xs font-mono font-bold transition-all ${
                  reportMode === 'timeline'
                    ? 'bg-emerald-500 text-black'
                    : 'text-gray-400 hover:text-gray-200'
                }`}
              >
                Detailed Timeline
              </button>
              <button
                onClick={() => setReportMode('executive')}
                className={`px-4 py-1.5 rounded-lg text-xs font-mono font-bold transition-all ${
                  reportMode === 'executive'
                    ? 'bg-emerald-500 text-black'
                    : 'text-gray-400 hover:text-gray-200'
                }`}
              >
                Executive Report
              </button>
            </div>
          </div>

          {reportMode === 'timeline' ? (
            <>
              {/* Tactical Timeline Horizontal Bar */}
              {tacticalTimeline && tacticalTimeline.length > 0 && (
                <div className="mb-8 bg-black/40 border border-gray-900/60 rounded-2xl p-4">
                  <div className="text-[10px] font-bold font-mono uppercase tracking-[0.2em] text-gray-500 mb-3 flex items-center gap-1.5">
                    <Clock size={12} className="text-emerald-500" />
                    Match Tactical Timeline (Segmented Phases)
                  </div>
                  <div className="flex w-full items-stretch gap-2.5 overflow-x-auto pb-1.5 [&::-webkit-scrollbar]:h-1 [&::-webkit-scrollbar-thumb]:bg-gray-800">
                    {tacticalTimeline.map((seg: any, sIdx: number) => {
                      const isActive = activeSegmentIdx === sIdx;
                      return (
                        <button
                          key={sIdx}
                          onClick={() => setActiveSegmentIdx(sIdx)}
                          className={`flex-1 min-w-[140px] p-3.5 rounded-xl border text-left transition-all ${
                            isActive
                              ? "bg-emerald-500/10 border-emerald-500/40 shadow-[0_0_15px_rgba(16,185,129,0.05)]"
                              : "bg-black/20 border-gray-900/80 hover:border-gray-850 hover:bg-gray-900/20"
                          }`}
                        >
                          <div className="text-[9px] font-mono text-gray-500 font-bold uppercase mb-1 flex justify-between items-center">
                            <span>{seg.label}</span>
                            {isActive && <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />}
                          </div>
                          <div className="text-xs font-bold text-gray-200 truncate">{seg.team_0.phase}</div>
                          <div className="text-[9px] text-gray-500 font-mono mt-0.5 truncate">vs {seg.team_1.phase}</div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Selected Segment Details */}
              {activeSegmentIdx !== null && tacticalTimeline[activeSegmentIdx] && (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 p-6 bg-black/30 border border-gray-900 rounded-2xl mb-8 animate-fade-in">
                  {/* Col 1: Tactical Explanation */}
                  <div className="space-y-4">
                    <div className="text-[10px] font-bold font-mono uppercase tracking-widest text-emerald-500">
                      Segment Tactical Overview
                    </div>
                    
                    {/* Red Team Phase */}
                    <div className="bg-red-500/5 border border-red-500/10 rounded-xl p-4 space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-xs font-bold text-red-400 font-mono">
                          {resolveTeamLabel('team_0', { savedMappings, useAltNames, dictionary }).toUpperCase()}
                        </span>
                        <span className="px-2 py-0.5 rounded bg-red-500/10 text-[9px] font-bold font-mono text-red-400">
                          {tacticalTimeline[activeSegmentIdx].team_0.phase}
                        </span>
                      </div>
                      <p className="text-xs text-gray-400 leading-relaxed">
                        {tacticalTimeline[activeSegmentIdx].team_0.explanation}
                      </p>
                      {tacticalTimeline[activeSegmentIdx].team_0.philosophy_quote && (
                        <div className="border-t border-red-500/10 pt-2 mt-2">
                          <p className="text-[10px] italic text-gray-500">
                            "{tacticalTimeline[activeSegmentIdx].team_0.philosophy_quote}"
                          </p>
                          <p className="text-[8px] font-mono text-red-400/60 mt-0.5 text-right">
                            — {tacticalTimeline[activeSegmentIdx].team_0.philosophy_author}
                          </p>
                        </div>
                      )}
                    </div>

                    {/* Blue Team Phase */}
                    <div className="bg-blue-500/5 border border-blue-500/10 rounded-xl p-4 space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-xs font-bold text-blue-400 font-mono">
                          {resolveTeamLabel('team_1', { savedMappings, useAltNames, dictionary }).toUpperCase()}
                        </span>
                        <span className="px-2 py-0.5 rounded bg-blue-500/10 text-[9px] font-bold font-mono text-blue-400">
                          {tacticalTimeline[activeSegmentIdx].team_1.phase}
                        </span>
                      </div>
                      <p className="text-xs text-gray-400 leading-relaxed">
                        {tacticalTimeline[activeSegmentIdx].team_1.explanation}
                      </p>
                      {tacticalTimeline[activeSegmentIdx].team_1.philosophy_quote && (
                        <div className="border-t border-blue-500/10 pt-2 mt-2">
                          <p className="text-[10px] italic text-gray-500">
                            "{tacticalTimeline[activeSegmentIdx].team_1.philosophy_quote}"
                          </p>
                          <p className="text-[8px] font-mono text-blue-400/60 mt-0.5 text-right">
                            — {tacticalTimeline[activeSegmentIdx].team_1.philosophy_author}
                          </p>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Col 2: Segment Metrics */}
                  <div className="space-y-4">
                    <div className="text-[10px] font-bold font-mono uppercase tracking-widest text-emerald-500">
                      Segment Average Metrics
                    </div>
                    <div className="bg-black/40 border border-gray-900 rounded-xl p-4 space-y-4">
                      {/* Tactical Power */}
                      <div className="space-y-1.5">
                        <div className="flex justify-between text-[10px] font-mono">
                          <span className="text-gray-500">TACTICAL POWER</span>
                          <span className="text-gray-300">
                            {resolveTeamLabel('team_0', { savedMappings, short: true })} <span className="text-emerald-400 font-bold">{tacticalTimeline[activeSegmentIdx].team_0.metrics.tactical_power.toFixed(1)}</span> / {resolveTeamLabel('team_1', { savedMappings, short: true })} <span className="text-emerald-400 font-bold">{tacticalTimeline[activeSegmentIdx].team_1.metrics.tactical_power.toFixed(1)}</span>
                          </span>
                        </div>
                        <div className="h-1.5 bg-gray-950 rounded-full flex overflow-hidden">
                          <div className="bg-red-500 h-full" style={{ width: `${(tacticalTimeline[activeSegmentIdx].team_0.metrics.tactical_power / (tacticalTimeline[activeSegmentIdx].team_0.metrics.tactical_power + tacticalTimeline[activeSegmentIdx].team_1.metrics.tactical_power)) * 100}%` }} />
                          <div className="bg-blue-500 h-full" style={{ width: `${(tacticalTimeline[activeSegmentIdx].team_1.metrics.tactical_power / (tacticalTimeline[activeSegmentIdx].team_0.metrics.tactical_power + tacticalTimeline[activeSegmentIdx].team_1.metrics.tactical_power)) * 100}%` }} />
                        </div>
                      </div>

                      {/* Compactness */}
                      <div className="space-y-1.5">
                        <div className="flex justify-between text-[10px] font-mono">
                          <span className="text-gray-500">SHAPE COMPACTNESS</span>
                          <span className="text-gray-300">
                            {resolveTeamLabel('team_0', { savedMappings, short: true })} <span className="font-bold">{Math.round(tacticalTimeline[activeSegmentIdx].team_0.metrics.compactness)}%</span> / {resolveTeamLabel('team_1', { savedMappings, short: true })} <span className="font-bold">{Math.round(tacticalTimeline[activeSegmentIdx].team_1.metrics.compactness)}%</span>
                          </span>
                        </div>
                        <div className="h-1.5 bg-gray-950 rounded-full flex gap-1">
                          <div className="bg-red-500/80 h-full rounded-full" style={{ width: `${tacticalTimeline[activeSegmentIdx].team_0.metrics.compactness}%` }} />
                          <div className="bg-blue-500/80 h-full rounded-full" style={{ width: `${tacticalTimeline[activeSegmentIdx].team_1.metrics.compactness}%`, marginLeft: 'auto' }} />
                        </div>
                      </div>

                      {/* Transition Speed */}
                      <div className="space-y-1.5">
                        <div className="flex justify-between text-[10px] font-mono">
                          <span className="text-gray-500">TRANSITION SPEED</span>
                          <span className="text-gray-300">
                            {resolveTeamLabel('team_0', { savedMappings, short: true })} <span className="font-bold">{Math.round(tacticalTimeline[activeSegmentIdx].team_0.metrics.transition_speed)}</span> / {resolveTeamLabel('team_1', { savedMappings, short: true })} <span className="font-bold">{Math.round(tacticalTimeline[activeSegmentIdx].team_1.metrics.transition_speed)}</span>
                          </span>
                        </div>
                        <div className="h-1.5 bg-gray-950 rounded-full flex gap-1">
                          <div className="bg-red-500/80 h-full rounded-full" style={{ width: `${tacticalTimeline[activeSegmentIdx].team_0.metrics.transition_speed}%` }} />
                          <div className="bg-blue-500/80 h-full rounded-full" style={{ width: `${tacticalTimeline[activeSegmentIdx].team_1.metrics.transition_speed}%`, marginLeft: 'auto' }} />
                        </div>
                      </div>

                      {/* Defensive Shape */}
                      <div className="space-y-1.5">
                        <div className="flex justify-between text-[10px] font-mono">
                          <span className="text-gray-500">DEFENSIVE SOLIDITY</span>
                          <span className="text-gray-300">
                            {resolveTeamLabel('team_0', { savedMappings, short: true })} <span className="font-bold">{Math.round(tacticalTimeline[activeSegmentIdx].team_0.metrics.defensive_shape)}%</span> / {resolveTeamLabel('team_1', { savedMappings, short: true })} <span className="font-bold">{Math.round(tacticalTimeline[activeSegmentIdx].team_1.metrics.defensive_shape)}%</span>
                          </span>
                        </div>
                        <div className="h-1.5 bg-gray-950 rounded-full flex gap-1">
                          <div className="bg-red-500/80 h-full rounded-full" style={{ width: `${tacticalTimeline[activeSegmentIdx].team_0.metrics.defensive_shape}%` }} />
                          <div className="bg-blue-500/80 h-full rounded-full" style={{ width: `${tacticalTimeline[activeSegmentIdx].team_1.metrics.defensive_shape}%`, marginLeft: 'auto' }} />
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Col 3: Key Events */}
                  <div className="space-y-4">
                    <div className="text-[10px] font-bold font-mono uppercase tracking-widest text-emerald-500">
                      Key Segment Moments (Events)
                    </div>
                    <div className="max-h-[280px] overflow-y-auto space-y-2 [&::-webkit-scrollbar]:w-1 [&::-webkit-scrollbar-thumb]:bg-gray-800 pr-1">
                      {tacticalTimeline[activeSegmentIdx].key_events.length === 0 ? (
                        <div className="text-xs text-gray-600 italic font-mono py-8 text-center bg-black/20 border border-gray-900/60 rounded-xl">
                          No key events registered in this segment.
                        </div>
                      ) : (
                        tacticalTimeline[activeSegmentIdx].key_events.map((evt: any, eIdx: number) => (
                          <button
                            key={eIdx}
                            onClick={(e) => { e.stopPropagation(); handlePlayClip(evt.start_time_s); }}
                            className="w-full text-left p-3 rounded-xl bg-black/40 border border-gray-900 hover:border-emerald-500/30 hover:bg-[#111a12]/20 transition-all flex items-center justify-between group"
                          >
                            <div className="flex-1 min-w-0 pr-3">
                              <div className="text-xs font-bold text-gray-200 truncate group-hover:text-emerald-400 transition-colors">
                                {evt.event_name}
                              </div>
                              <p className="text-[10px] text-gray-500 mt-1 leading-relaxed font-sans line-clamp-2">
                                {evt.description}
                              </p>
                              <div className="flex items-center gap-2 mt-2 text-[9px] font-mono text-gray-600">
                                <span className="flex items-center gap-1">
                                  <Clock size={10} /> 
                                  {Math.floor(evt.start_time_s / 60)}:{(Math.floor(evt.start_time_s % 60)).toString().padStart(2, '0')}
                                </span>
                                <span className="bg-emerald-500/5 text-emerald-500/50 px-1.5 py-0.25 rounded">
                                  {evt.confidence_pct}% Match
                                </span>
                              </div>
                            </div>
                            <div className="h-6 w-6 rounded-full bg-emerald-500/10 flex items-center justify-center text-emerald-500 group-hover:bg-emerald-500/20 group-hover:scale-105 transition-all shrink-0">
                              <Play size={10} fill="currentColor" />
                            </div>
                          </button>
                        ))
                      )}
                    </div>
                  </div>
                </div>
              )}

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
                  savedMappings={savedMappings}
                />
              ) : (
                <div className="h-48 flex flex-col items-center justify-center gap-2 text-gray-500 text-sm italic font-mono animate-pulse">
                  <Loader2 className="animate-spin text-emerald-500" size={24} />
                  <span>Synchronizing tactical debrief...</span>
                </div>
              )}
            </>
          ) : (
            <div className="space-y-8 text-gray-300 font-sans animate-fade-in">
              
              {/* 1. MATCH SUMMARY */}
              <div className="bg-black/30 border border-gray-900 rounded-2xl p-6 space-y-6">
                <div className="flex items-center justify-between border-b border-gray-900/60 pb-3">
                  <h3 className="text-lg font-bold text-gray-200 flex items-center gap-2">
                    <Award size={18} className="text-emerald-500" />
                    1. Match Summary & Performance Matrix
                  </h3>
                  <span className="text-[10px] font-mono text-gray-600 uppercase tracking-wider">Overall Match Indicators</span>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse text-xs font-mono">
                    <thead>
                      <tr className="border-b border-gray-900/60 text-gray-500">
                        <th className="py-2.5">Key Performance Indicator</th>
                        <th className="py-2.5 text-red-400">{resolveTeamLabel('team_0', { savedMappings, useAltNames, dictionary })}</th>
                        <th className="py-2.5 text-blue-400">{resolveTeamLabel('team_1', { savedMappings, useAltNames, dictionary })}</th>
                        <th className="py-2.5 text-emerald-400">Differential Advantage</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-900/40 text-gray-300">
                      {[
                        { label: 'Tactical Power (TPI)', key: 'tactical_power', unit: '' },
                        { label: 'Win Probability', key: 'win_prob', unit: '%' },
                        { label: 'Shape Compactness', key: 'compactness', unit: '%' },
                        { label: 'Press Resistance', key: 'press_resistance', unit: '' },
                        { label: 'Width Utilization', key: 'width_utilization', unit: '' },
                        { label: 'Line Staggering', key: 'line_staggering', unit: '' },
                        { label: 'Transition Speed', key: 'transition_speed', unit: '' }
                      ].map((row, rIdx) => {
                        const summaryMetrics = parsedAllMetrics || {
                          team_0: { tactical_power: 0, win_prob: 0, compactness: 0, press_resistance: 0, width_utilization: 0, line_staggering: 0, transition_speed: 0 },
                          team_1: { tactical_power: 0, win_prob: 0, compactness: 0, press_resistance: 0, width_utilization: 0, line_staggering: 0, transition_speed: 0 }
                        };
                        const val0 = summaryMetrics.team_0[row.key as keyof typeof summaryMetrics.team_0] || 0;
                        const val1 = summaryMetrics.team_1[row.key as keyof typeof summaryMetrics.team_1] || 0;
                        const diff = val0 - val1;
                        let advantageText = 'Balanced';
                        let advantageColor = 'text-gray-500';

                        if (diff > 0.05) {
                          advantageText = `${resolveTeamLabel('team_0', { savedMappings, short: true })} (+${diff.toFixed(1)}${row.unit})`;
                          advantageColor = 'text-red-400';
                        } else if (diff < -0.05) {
                          advantageText = `${resolveTeamLabel('team_1', { savedMappings, short: true })} (+${Math.abs(diff).toFixed(1)}${row.unit})`;
                          advantageColor = 'text-blue-400';
                        }

                        return (
                          <tr key={rIdx} className="hover:bg-white/[0.01] transition-colors">
                            <td className="py-3 font-semibold text-gray-400">{row.label}</td>
                            <td className="py-3 text-red-400 font-bold">{val0.toFixed(1)}{row.unit}</td>
                            <td className="py-3 text-blue-400 font-bold">{val1.toFixed(1)}{row.unit}</td>
                            <td className={`py-3 font-bold ${advantageColor}`}>{advantageText}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                {summaryItem && (
                  <div className="bg-[#111a12]/30 border border-emerald-500/10 rounded-xl p-5 relative overflow-hidden">
                    <div className="absolute top-0 left-0 w-1 h-full bg-emerald-500/40" />
                    <h4 className="text-[10px] font-mono text-emerald-500 uppercase tracking-widest font-bold mb-2">Tactical Summary Verdict</h4>
                    <p className="text-gray-300 text-sm leading-relaxed whitespace-pre-line">
                      {summaryItem.tactical_instruction || summaryItem.evidence}
                    </p>
                  </div>
                )}
              </div>

              {/* 2 & 3. STRENGTHS & WEAKNESSES */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                
                {/* 2. TACTICAL STRENGTHS */}
                <div className="bg-black/30 border border-gray-900 rounded-2xl p-6 space-y-4">
                  <div className="flex items-center justify-between border-b border-gray-900 pb-3">
                    <h3 className="text-base font-bold text-emerald-400 flex items-center gap-2">
                      <TrendingUp size={16} />
                      2. Tactical Strengths
                    </h3>
                  </div>

                  <div className="space-y-4">
                    {/* Team A Strengths */}
                    <div className="space-y-2">
                      <h4 className="text-xs font-bold text-red-400 uppercase tracking-wider font-mono">
                        {resolveTeamLabel('team_0', { savedMappings, useAltNames, dictionary })}
                      </h4>
                      <ul className="space-y-2 text-xs text-gray-400">
                        {(() => {
                          const summaryMetrics = parsedAllMetrics || {
                            team_0: { tactical_power: 0, win_prob: 0, compactness: 0, press_resistance: 0, width_utilization: 0, line_staggering: 0, transition_speed: 0 },
                            team_1: { tactical_power: 0, win_prob: 0, compactness: 0, press_resistance: 0, width_utilization: 0, line_staggering: 0, transition_speed: 0 }
                          };
                          return Object.entries({
                            tactical_power: 'Tactical Power Intensity',
                            win_prob: 'Win Probability Advantage',
                            compactness: 'Defensive Shape Compactness',
                            press_resistance: 'Press Resistance / Build-up Stability',
                            width_utilization: 'Attacking Width Utilization',
                            line_staggering: 'Depth Line Staggering',
                            transition_speed: 'Offensive Transition Velocity'
                          }).map(([key, label]) => {
                            const val0 = summaryMetrics.team_0[key as keyof typeof summaryMetrics.team_0] || 0;
                            const val1 = summaryMetrics.team_1[key as keyof typeof summaryMetrics.team_1] || 0;
                            if (val0 > val1) {
                              return (
                                <li key={key} className="flex items-start gap-2 bg-emerald-500/5 border border-emerald-500/10 rounded-lg p-2">
                                  <span className="text-emerald-400 font-bold">✓</span>
                                  <div>
                                    <span className="text-gray-300 font-bold">{label}</span>: Higher efficiency recorded ({val0.toFixed(1)} vs {val1.toFixed(1)}).
                                  </div>
                                </li>
                              );
                            }
                            return null;
                          });
                        })()}
                      </ul>
                    </div>

                    {/* Team B Strengths */}
                    <div className="space-y-2">
                      <h4 className="text-xs font-bold text-blue-400 uppercase tracking-wider font-mono">
                        {resolveTeamLabel('team_1', { savedMappings, useAltNames, dictionary })}
                      </h4>
                      <ul className="space-y-2 text-xs text-gray-400">
                        {(() => {
                          const summaryMetrics = parsedAllMetrics || {
                            team_0: { tactical_power: 0, win_prob: 0, compactness: 0, press_resistance: 0, width_utilization: 0, line_staggering: 0, transition_speed: 0 },
                            team_1: { tactical_power: 0, win_prob: 0, compactness: 0, press_resistance: 0, width_utilization: 0, line_staggering: 0, transition_speed: 0 }
                          };
                          return Object.entries({
                            tactical_power: 'Tactical Power Intensity',
                            win_prob: 'Win Probability Advantage',
                            compactness: 'Defensive Shape Compactness',
                            press_resistance: 'Press Resistance / Build-up Stability',
                            width_utilization: 'Attacking Width Utilization',
                            line_staggering: 'Depth Line Staggering',
                            transition_speed: 'Offensive Transition Velocity'
                          }).map(([key, label]) => {
                            const val0 = summaryMetrics.team_0[key as keyof typeof summaryMetrics.team_0] || 0;
                            const val1 = summaryMetrics.team_1[key as keyof typeof summaryMetrics.team_1] || 0;
                            if (val1 > val0) {
                              return (
                                <li key={key} className="flex items-start gap-2 bg-emerald-500/5 border border-emerald-500/10 rounded-lg p-2">
                                  <span className="text-emerald-400 font-bold">✓</span>
                                  <div>
                                    <span className="text-gray-300 font-bold">{label}</span>: Higher efficiency recorded ({val1.toFixed(1)} vs {val0.toFixed(1)}).
                                  </div>
                                </li>
                              );
                            }
                            return null;
                          });
                        })()}
                      </ul>
                    </div>
                  </div>
                </div>

                {/* 3. TACTICAL WEAKNESSES */}
                <div className="bg-black/30 border border-gray-900 rounded-2xl p-6 space-y-4">
                  <div className="flex items-center justify-between border-b border-gray-900 pb-3">
                    <h3 className="text-base font-bold text-red-400 flex items-center gap-2">
                      <Shield size={16} />
                      3. Tactical Weaknesses
                    </h3>
                  </div>

                  <div className="space-y-4 max-h-[360px] overflow-y-auto pr-1 [&::-webkit-scrollbar]:w-1 [&::-webkit-scrollbar-thumb]:bg-gray-800">
                    {coachAdvice?.advice_items?.filter((i: any) => i.flaw !== 'Match Summary').length === 0 ? (
                      <div className="text-xs text-gray-500 italic text-center py-10 font-mono">
                        No tactical weaknesses identified.
                      </div>
                    ) : (
                      coachAdvice?.advice_items?.filter((i: any) => i.flaw !== 'Match Summary').map((item: any, idx: number) => {
                        const isRedTeam = item.team === 'team_0';
                        return (
                          <div key={idx} className="bg-red-500/5 border border-red-500/10 rounded-xl p-3.5 space-y-1">
                            <div className="flex justify-between items-center text-[10px] font-mono">
                              <span className={`font-bold ${isRedTeam ? 'text-red-400' : 'text-blue-400'}`}>
                                {resolveTeamLabel(item.team, { savedMappings, useAltNames, dictionary }).toUpperCase()}
                              </span>
                              <span className="px-1.5 py-0.25 rounded bg-red-500/10 text-red-400 font-bold">
                                {item.severity || 'Medium'}
                              </span>
                            </div>
                            <h4 className="text-xs font-bold text-gray-200">{item.flaw}</h4>
                            <p className="text-[11px] text-gray-400 leading-relaxed">
                              {item.evidence}
                            </p>
                          </div>
                        );
                      })
                    )}
                  </div>
                </div>

              </div>

              {/* 4. KEY PLAYERS */}
              <div className="bg-black/30 border border-gray-900 rounded-2xl p-6 space-y-4">
                <div className="flex items-center justify-between border-b border-gray-900 pb-3">
                  <h3 className="text-lg font-bold text-gray-200 flex items-center gap-2">
                    <User size={18} className="text-emerald-500" />
                    4. Key Players & Tactical Threats
                  </h3>
                </div>

                {threatContext?.top_threats && threatContext.top_threats.length > 0 ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {threatContext.top_threats.map((threat: any, idx: number) => {
                      const isTeam0 = threat.team_id === 'team_0';
                      return (
                        <div key={idx} className="bg-black/40 border border-gray-900 rounded-xl p-4 flex flex-col justify-between min-h-[120px]">
                          <div>
                            <div className="flex items-center justify-between mb-2">
                              <div className="flex items-center gap-2">
                                <div className={`h-6 w-6 rounded-full flex items-center justify-center text-xs font-bold ${isTeam0 ? 'bg-red-500/20 text-red-400' : 'bg-blue-500/20 text-blue-400'}`}>
                                  {savedMappings?.mappings?.[String(threat.player_id)]
                                    ? `#${savedMappings.mappings[String(threat.player_id)].number}`
                                    : `P${threat.player_id}`}
                                </div>
                                <span className="text-xs font-bold text-gray-300">
                                  {resolvePlayerLabel(threat.player_id, { savedMappings, useAltNames, dictionary })}
                                </span>
                              </div>
                              <span className="text-[10px] font-mono font-bold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-lg">
                                Threat: {threat.threat_score.toFixed(1)}
                              </span>
                            </div>
                            <p className="text-[11px] text-gray-400 leading-normal mb-3">
                              {threat.explanation}
                            </p>
                          </div>
                          <div className="text-[9px] font-mono text-gray-600 uppercase tracking-wider">
                            Team: {resolveTeamLabel(threat.team_id, { savedMappings, useAltNames, dictionary })}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="text-xs text-gray-500 italic text-center py-8 font-mono bg-black/20 border border-gray-900/60 rounded-xl">
                    No key tactical threat players registered in telemetry logs.
                  </div>
                )}
              </div>

              {/* 5. TRANSITION ANALYSIS */}
              <div className="bg-black/30 border border-gray-900 rounded-2xl p-6 space-y-4">
                <div className="flex items-center justify-between border-b border-gray-900 pb-3">
                  <h3 className="text-lg font-bold text-gray-200 flex items-center gap-2">
                    <Zap size={18} className="text-emerald-500" />
                    5. Transition Analysis
                  </h3>
                </div>

                {(() => {
                  const summaryMetrics = parsedAllMetrics || {
                    team_0: { tactical_power: 0, win_prob: 0, compactness: 0, press_resistance: 0, width_utilization: 0, line_staggering: 0, transition_speed: 0 },
                    team_1: { tactical_power: 0, win_prob: 0, compactness: 0, press_resistance: 0, width_utilization: 0, line_staggering: 0, transition_speed: 0 }
                  };
                  return (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                      <div className="bg-black/40 border border-gray-900 rounded-xl p-4 space-y-4">
                        <h4 className="text-xs font-bold text-gray-400 font-mono uppercase tracking-wider">Transition Velocities</h4>
                        <div className="space-y-4">
                          {/* Team A */}
                          <div className="space-y-1">
                            <div className="flex justify-between text-[11px] font-mono text-gray-400">
                              <span>{resolveTeamLabel('team_0', { savedMappings, useAltNames, dictionary })}</span>
                              <span className="font-bold text-red-400">{summaryMetrics.team_0.transition_speed.toFixed(1)}</span>
                            </div>
                            <div className="h-2 bg-gray-950 rounded-full overflow-hidden">
                              <div className="bg-red-500 h-full rounded-full" style={{ width: `${summaryMetrics.team_0.transition_speed}%` }} />
                            </div>
                          </div>

                          {/* Team B */}
                          <div className="space-y-1">
                            <div className="flex justify-between text-[11px] font-mono text-gray-400">
                              <span>{resolveTeamLabel('team_1', { savedMappings, useAltNames, dictionary })}</span>
                              <span className="font-bold text-blue-400">{summaryMetrics.team_1.transition_speed.toFixed(1)}</span>
                            </div>
                            <div className="h-2 bg-gray-950 rounded-full overflow-hidden">
                              <div className="bg-blue-500 h-full rounded-full" style={{ width: `${summaryMetrics.team_1.transition_speed}%` }} />
                            </div>
                          </div>
                        </div>
                      </div>

                      <div className="bg-black/40 border border-gray-900 rounded-xl p-4 flex flex-col justify-center">
                        <p className="text-xs text-gray-400 leading-relaxed font-sans">
                          Possession shifts trigger vertical telemetry scans monitoring counter-attack patterns. 
                          Higher values reflect prompt defensive block recovery and swift positional redistribution.
                        </p>
                        {coachAdvice?.advice_items?.find((i: any) => i.flaw?.toLowerCase().includes('transition') || i.flaw?.toLowerCase().includes('counter')) && (
                          <div className="mt-3 bg-emerald-500/5 border border-emerald-500/10 rounded-lg p-2.5 text-[11px] text-emerald-400/80">
                            <strong>Insight:</strong> {coachAdvice.advice_items.find((i: any) => i.flaw?.toLowerCase().includes('transition') || i.flaw?.toLowerCase().includes('counter')).evidence}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })()}
              </div>

              {/* 6 & 7. DEFENSIVE & ATTACKING SHAPE */}
              {(() => {
                const summaryMetrics = parsedAllMetrics || {
                  team_0: { tactical_power: 0, win_prob: 0, compactness: 0, press_resistance: 0, width_utilization: 0, line_staggering: 0, transition_speed: 0 },
                  team_1: { tactical_power: 0, win_prob: 0, compactness: 0, press_resistance: 0, width_utilization: 0, line_staggering: 0, transition_speed: 0 }
                };
                return (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    
                    {/* 6. DEFENSIVE SHAPE */}
                    <div className="bg-black/30 border border-gray-900 rounded-2xl p-6 space-y-4">
                      <div className="flex items-center justify-between border-b border-gray-900 pb-3">
                        <h3 className="text-base font-bold text-gray-200 flex items-center gap-2">
                          <Shield size={16} className="text-emerald-500" />
                          6. Defensive Shape & Compactness
                        </h3>
                      </div>

                      <div className="space-y-4">
                        <div className="space-y-1.5">
                          <div className="flex justify-between text-[10px] font-mono text-gray-400">
                            <span>SHAPE COMPACTNESS</span>
                            <span>
                              {resolveTeamLabel('team_0', { savedMappings, short: true })} <span className="font-bold">{summaryMetrics.team_0.compactness}%</span> / {resolveTeamLabel('team_1', { savedMappings, short: true })} <span className="font-bold">{summaryMetrics.team_1.compactness}%</span>
                            </span>
                          </div>
                          <div className="h-2 bg-gray-950 rounded-full flex gap-0.5">
                            <div className="bg-red-500 h-full rounded-l-full" style={{ width: `${summaryMetrics.team_0.compactness}%` }} />
                            <div className="bg-blue-500 h-full rounded-r-full" style={{ width: `${summaryMetrics.team_1.compactness}%`, marginLeft: 'auto' }} />
                          </div>
                        </div>

                        <div className="space-y-1.5">
                          <div className="flex justify-between text-[10px] font-mono text-gray-400">
                            <span>LINE STAGGERING (DEPTH DISCIPLINE)</span>
                            <span>
                              {resolveTeamLabel('team_0', { savedMappings, short: true })} <span className="font-bold">{summaryMetrics.team_0.line_staggering}%</span> / {resolveTeamLabel('team_1', { savedMappings, short: true })} <span className="font-bold">{summaryMetrics.team_1.line_staggering}%</span>
                            </span>
                          </div>
                          <div className="h-2 bg-gray-950 rounded-full flex gap-0.5">
                            <div className="bg-red-500 h-full rounded-l-full" style={{ width: `${summaryMetrics.team_0.line_staggering}%` }} />
                            <div className="bg-blue-500 h-full rounded-r-full" style={{ width: `${summaryMetrics.team_1.line_staggering}%`, marginLeft: 'auto' }} />
                          </div>
                        </div>

                        <p className="text-[11px] text-gray-500 leading-normal">
                          Compactness represents the spatial layout density of players in defensive phases. High staggering shows mature horizontal and vertical line coordination preventing vertical passes.
                        </p>
                      </div>
                    </div>

                    {/* 7. ATTACKING SHAPE */}
                    <div className="bg-black/30 border border-gray-900 rounded-2xl p-6 space-y-4">
                      <div className="flex items-center justify-between border-b border-gray-900 pb-3">
                        <h3 className="text-base font-bold text-gray-200 flex items-center gap-2">
                          <Maximize2 size={16} className="text-emerald-500" />
                          7. Attacking Shape & Width
                        </h3>
                      </div>

                      <div className="space-y-4">
                        <div className="space-y-1.5">
                          <div className="flex justify-between text-[10px] font-mono text-gray-400">
                            <span>WIDTH UTILIZATION (WING OCCUPATION)</span>
                            <span>
                              {resolveTeamLabel('team_0', { savedMappings, short: true })} <span className="font-bold">{summaryMetrics.team_0.width_utilization}%</span> / {resolveTeamLabel('team_1', { savedMappings, short: true })} <span className="font-bold">{summaryMetrics.team_1.width_utilization}%</span>
                            </span>
                          </div>
                          <div className="h-2 bg-gray-950 rounded-full flex gap-0.5">
                            <div className="bg-red-500 h-full rounded-l-full" style={{ width: `${summaryMetrics.team_0.width_utilization}%` }} />
                            <div className="bg-blue-500 h-full rounded-r-full" style={{ width: `${summaryMetrics.team_1.width_utilization}%`, marginLeft: 'auto' }} />
                          </div>
                        </div>

                        <div className="space-y-1.5">
                          <div className="flex justify-between text-[10px] font-mono text-gray-400">
                            <span>PRESS RESISTANCE (BUILD-UP STABILITY)</span>
                            <span>
                              {resolveTeamLabel('team_0', { savedMappings, short: true })} <span className="font-bold">{summaryMetrics.team_0.press_resistance}%</span> / {resolveTeamLabel('team_1', { savedMappings, short: true })} <span className="font-bold">{summaryMetrics.team_1.press_resistance}%</span>
                            </span>
                          </div>
                          <div className="h-2 bg-gray-950 rounded-full flex gap-0.5">
                            <div className="bg-red-500 h-full rounded-l-full" style={{ width: `${summaryMetrics.team_0.press_resistance}%` }} />
                            <div className="bg-blue-500 h-full rounded-r-full" style={{ width: `${summaryMetrics.team_1.press_resistance}%`, marginLeft: 'auto' }} />
                          </div>
                        </div>

                        <p className="text-[11px] text-gray-500 leading-normal">
                          Attacking shape is measured by wing channel occupancy (Width) and structural stability under pressing traps (Press Resistance). Proper stagger values prevent turnover risk.
                        </p>
                      </div>
                    </div>

                  </div>
                );
              })()}

              {/* 8. RECOMMENDED ADJUSTMENTS */}
              <div className="bg-[#111a12]/20 border border-emerald-500/10 rounded-2xl p-6 space-y-4">
                <div className="flex items-center justify-between border-emerald-500/10 pb-3 border-b">
                  <h3 className="text-lg font-bold text-emerald-400 flex items-center gap-2">
                    <CheckCircle size={18} />
                    8. Recommended Coaching Adjustments
                  </h3>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Team A Recommendations */}
                  <div className="space-y-3">
                    <h4 className="text-xs font-bold text-red-400 uppercase tracking-wider font-mono">
                      {resolveTeamLabel('team_0', { savedMappings, useAltNames, dictionary })} Recommendations
                    </h4>
                    <div className="space-y-2">
                      {(() => {
                        const recs: string[] = [];
                        coachAdvice?.advice_items?.forEach((item: any) => {
                          if (item.team === 'team_0' || item.team === 'global') {
                            const steps = item.tactical_instruction_steps || [];
                            steps.forEach((s: string) => {
                              if (s.includes('**') && s.length > 20 && !s.includes('Red Team') && !s.includes('Summary')) {
                                recs.push(s);
                              }
                            });
                          }
                        });

                        if (recs.length === 0) {
                          return <p className="text-xs text-gray-500 italic font-mono">No specific adjustments required.</p>;
                        }

                        return recs.slice(0, 5).map((rec, rIdx) => (
                          <div key={rIdx} className="bg-black/30 border border-gray-900 rounded-xl p-3.5 text-xs leading-relaxed text-gray-300">
                            <span className="font-bold text-emerald-400 font-mono mr-1.5">{rIdx + 1}.</span>
                            <span>{formatBoldText(rec)}</span>
                          </div>
                        ));
                      })()}
                    </div>
                  </div>

                  {/* Team B Recommendations */}
                  <div className="space-y-3">
                    <h4 className="text-xs font-bold text-blue-400 uppercase tracking-wider font-mono">
                      {resolveTeamLabel('team_1', { savedMappings, useAltNames, dictionary })} Recommendations
                    </h4>
                    <div className="space-y-2">
                      {(() => {
                        const recs: string[] = [];
                        coachAdvice?.advice_items?.forEach((item: any) => {
                          if (item.team === 'team_1' || item.team === 'global') {
                            const steps = item.tactical_instruction_steps || [];
                            steps.forEach((s: string) => {
                              if (s.includes('**') && s.length > 20 && !s.includes('Blue Team') && !s.includes('Summary')) {
                                recs.push(s);
                              }
                            });
                          }
                        });

                        if (recs.length === 0) {
                          return <p className="text-xs text-gray-500 italic font-mono">No specific adjustments required.</p>;
                        }

                        return recs.slice(0, 5).map((rec, rIdx) => (
                          <div key={rIdx} className="bg-black/30 border border-gray-900 rounded-xl p-3.5 text-xs leading-relaxed text-gray-300">
                            <span className="font-bold text-emerald-400 font-mono mr-1.5">{rIdx + 1}.</span>
                            <span>{formatBoldText(rec)}</span>
                          </div>
                        ));
                      })()}
                    </div>
                  </div>
                </div>
              </div>

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
               <span>AI Tactical Assistant</span>
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
                                        {savedMappings?.mappings?.[String(threat.player_id)]
                                          ? `#${savedMappings.mappings[String(threat.player_id)].number}`
                                          : `P${threat.player_id}`}
                                      </div>
                                      <span className="text-xs font-bold text-gray-300 uppercase">
                                        {resolvePlayerLabel(threat.player_id, { savedMappings, useAltNames, dictionary })}
                                        {' · '}{resolveTeamLabel(threat.team_id, { savedMappings, useAltNames, dictionary, short: true })}
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
        onDownloadPdf={handleDownloadPdf}
        isSaving={isSaving}
        saveStatus={saveStatus}
      />
    </div>
  );
}
