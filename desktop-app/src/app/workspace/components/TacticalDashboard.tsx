"use client";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { Bot, Clock, Send, User } from "lucide-react";

import { useEngineConfig } from "@/context/EngineContext";
import {
  type CoachAdviceResponse,
  type JobArtifactsResponse,
  getCoachAdvice,
  sendChat,
} from "@/lib/api";
import { useStreamingText } from "@/hooks/useStreamingText";

import { type KeywordConfig, InsightCard } from "./InsightCard";
import { VideoHUD } from "./VideoHUD";

interface TacticalTimelineItem {
  time: string;
  title: string;
  minute: string;
  summary: string;
  payload: string;
  keywords: KeywordConfig[];
}

interface TacticalDashboardProps {
  jobId: string | null;
  initialAdvice: CoachAdviceResponse | null;
  artifacts: JobArtifactsResponse | null;
}

function toKeywordConfigs(adviceText: string): KeywordConfig[] {
  const snippets = adviceText
    .split(/[,.]/)
    .map((snippet) => snippet.trim())
    .filter((snippet) => snippet.length > 0)
    .slice(0, 3);
  const palette: KeywordConfig["color"][] = ["emerald", "amber", "cyan"];
  return snippets.map((text, index) => ({
    text,
    color: palette[index % palette.length],
    role: `Instruction_${index + 1}`,
  }));
}

function toTimeline(advice: CoachAdviceResponse | null): TacticalTimelineItem[] {
  if (!advice || advice.advice_items.length === 0) {
    return [
      {
        time: "00:00 - 90:00",
        title: "No Triggered Tactical Insights",
        minute: "00:00",
        summary: "No tactical triggers exceeded confidence/frequency thresholds.",
        payload:
          "Analysis completed but no strong tactical violations were detected. Try a longer segment, higher-quality footage, or rerun with local LLM enabled for narrative coaching.",
        keywords: [],
      },
    ];
  }

  return advice.advice_items.map((item, index) => {
    const minute = Math.max(0, Math.floor(item.frame_idx / 25 / 60));
    const minuteLabel = `${minute.toString().padStart(2, "0")}:${(item.frame_idx % 60)
      .toString()
      .padStart(2, "0")}`;
    const instruction =
      item.tactical_instruction ??
      item.tactical_instruction_steps.join(". ") ??
      "No tactical instruction available.";
    return {
      time: `Segment ${index + 1}`,
      title: item.flaw || "Tactical Event",
      minute: minuteLabel,
      summary: `${item.team} • ${item.severity}`,
      payload: instruction,
      keywords: toKeywordConfigs(instruction),
    };
  });
}

function StreamingBubble({ text }: { text: string }): React.JSX.Element {
  const { displayedText } = useStreamingText(text, 25);
  return (
    <div className="flex gap-4">
      <div className="w-8 h-8 rounded-full bg-emerald-900/50 flex items-center justify-center flex-shrink-0 border border-emerald-500/50">
        <Bot size={16} className="text-emerald-400" />
      </div>
      <div className="flex-1 bg-gray-800/80 border border-gray-700/50 rounded-2xl rounded-tl-none p-4 text-sm text-gray-300">
        {displayedText}
      </div>
    </div>
  );
}

export function TacticalDashboard({
  jobId,
  initialAdvice,
  artifacts,
}: TacticalDashboardProps): React.JSX.Element {
  const { llmEngine } = useEngineConfig();
  const [advice, setAdvice] = useState<CoachAdviceResponse | null>(initialAdvice);
  const [adviceError, setAdviceError] = useState<string | null>(null);
  const timeline = useMemo(() => toTimeline(advice), [advice]);
  const [activeIndex, setActiveIndex] = useState(0);
  const [promptInput, setPromptInput] = useState("");
  const [chatHistory, setChatHistory] = useState<{ id: string; role: "user" | "ai"; text: string }[]>(
    [],
  );
  const [isTyping, setIsTyping] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [chatHistory, isTyping]);

  useEffect(() => {
    if (!jobId) {
      return;
    }
    let cancelled = false;
    const attemptRefetch = async (): Promise<void> => {
      try {
        const payload = await getCoachAdvice(jobId, { llmEngine });
        if (cancelled) {
          return;
        }
        setAdvice(payload);
        setAdviceError(null);
      } catch (error) {
        if (!cancelled) {
          setAdviceError(error instanceof Error ? error.message : "Advice refresh failed.");
        }
      }
    };
    void attemptRefetch();
    const id = setInterval(() => void attemptRefetch(), 5000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [jobId, llmEngine]);

  const handleSendPrompt = async (event: React.FormEvent): Promise<void> => {
    event.preventDefault();
    const newQuery = promptInput.trim();
    if (!newQuery || !jobId) {
      return;
    }
    setPromptInput("");
    setChatHistory((prev) => [...prev, { id: Date.now().toString(), role: "user", text: newQuery }]);
    setIsTyping(true);
    try {
      const response = await sendChat(jobId, newQuery);
      setChatHistory((prev) => [
        ...prev,
        { id: `${Date.now()}-ai`, role: "ai", text: response.reply || "No response." },
      ]);
    } catch (error) {
      setChatHistory((prev) => [
        ...prev,
        {
          id: `${Date.now()}-error`,
          role: "ai",
          text: error instanceof Error ? error.message : "Chat request failed.",
        },
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  const active = timeline[Math.min(activeIndex, timeline.length - 1)];

  return (
    <div className="flex h-full w-full bg-[#0a0f0a] font-sans relative">
      <div className="w-[30%] border-r border-gray-900 flex flex-col h-full bg-[#111a12]/30 overflow-hidden whitespace-nowrap">
        <div className="p-4 border-b border-gray-900 flex items-center justify-between shadow-sm">
          <h2 className="text-xs font-bold text-gray-500 tracking-widest uppercase font-mono">
            Match Timeline
          </h2>
          <Clock size={16} className="text-gray-600" />
        </div>
        <div className="flex-1 overflow-y-auto p-4 space-y-4 [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-thumb]:bg-gray-800 [&::-webkit-scrollbar-track]:bg-transparent">
          {timeline.map((item, index) => (
            <div
              key={`${item.title}-${index}`}
              onClick={() => setActiveIndex(index)}
              className={`p-4 border rounded-xl cursor-pointer transition-all ${
                index === activeIndex
                  ? "bg-[#111a12] border-emerald-500/30"
                  : "bg-[#0a0f0a] border-gray-800 hover:bg-[#111a12]/80"
              }`}
            >
              <div
                className={`text-[10px] font-mono mb-1 tracking-widest ${
                  index === activeIndex ? "text-emerald-500" : "text-gray-500"
                }`}
              >
                {item.time}
              </div>
              <div className="text-sm font-bold text-gray-300 font-sans tracking-tight">{item.title}</div>
              <div className="text-xs text-gray-500 mt-2 font-sans truncate">{item.summary}</div>
            </div>
          ))}
        </div>
      </div>
      <div className="flex-1 flex flex-col h-full min-w-0 bg-[#050805]">
        <div className="h-[45%] border-b border-gray-900 p-6 flex flex-col justify-center items-center relative overflow-hidden bg-[#050805] flex-shrink-0">
          <div className="absolute top-4 left-4 text-[10px] font-bold text-gray-600 tracking-widest uppercase font-mono z-30 flex items-center gap-2">
            <div className="w-2 h-2 bg-red-600 rounded-full animate-pulse"></div>
            Live IPC Feed
          </div>
          <VideoHUD jobId={jobId} artifacts={artifacts} />
        </div>
        <div className="flex-1 flex flex-row relative h-[55%] bg-[#0a0f0a]">
          <div className="w-[60%] border-r border-gray-900 flex flex-col relative overflow-hidden bg-[#050805]">
            <div className="flex-1 overflow-y-auto p-8 [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-thumb]:bg-gray-800">
              <div className="max-w-3xl mx-auto w-full space-y-6">
                <div className="text-[10px] font-bold text-gray-500 tracking-widest flex justify-between items-center uppercase font-mono">
                  <span>Proactive Insight Generation</span>
                  <span className="text-emerald-500">Telemetry Synced</span>
                </div>
                {adviceError && (
                  <div className="rounded border border-amber-800 bg-amber-950/50 px-3 py-2 text-xs text-amber-300">
                    {adviceError}
                  </div>
                )}
                <InsightCard
                  key={`${active.title}-${active.minute}`}
                  title={active.title}
                  minute={active.minute}
                  payload={active.payload}
                  keywords={active.keywords}
                />
              </div>
            </div>
          </div>
          <div className="w-[40%] flex flex-col relative bg-[#0a0f0a]">
            <div ref={scrollRef} className="flex-1 overflow-y-auto p-6 pb-24 space-y-6 [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-thumb]:bg-gray-800">
              {chatHistory.length === 0 && (
                <div className="flex flex-col items-center justify-center h-full text-center opacity-70 space-y-4 mt-8">
                  <div className="w-12 h-12 rounded-full bg-emerald-900/30 flex items-center justify-center border border-emerald-500/30">
                    <Bot size={24} className="text-emerald-500" />
                  </div>
                  <div className="space-y-2">
                    <p className="text-sm font-semibold text-gray-200 font-sans">Tactical Engine Ready</p>
                    <p className="text-xs text-gray-500 max-w-[250px] mx-auto font-sans leading-relaxed">
                      Ask a job-scoped follow-up. Chat is routed to `/api/v1/chat`.
                    </p>
                  </div>
                </div>
              )}
              {chatHistory.map((message) =>
                message.role === "user" ? (
                  <div key={message.id} className="flex gap-3 justify-end">
                    <div className="max-w-[85%] bg-[#111a12] border border-gray-800 rounded-2xl rounded-tr-none p-3.5 text-sm text-gray-200 font-sans shadow-sm">
                      {message.text}
                    </div>
                    <div className="w-8 h-8 rounded-full bg-gray-800 flex items-center justify-center flex-shrink-0 border border-gray-700">
                      <User size={14} className="text-gray-400" />
                    </div>
                  </div>
                ) : (
                  <StreamingBubble key={message.id} text={message.text} />
                ),
              )}
            </div>
            <div className="absolute bottom-0 left-0 right-0 p-5 bg-gradient-to-t from-[#0a0f0a] via-[#0a0f0a] to-transparent">
              <form onSubmit={(event) => void handleSendPrompt(event)} className="relative flex items-center max-w-lg mx-auto w-full shadow-[0_10px_40px_rgba(0,0,0,0.8)]">
                <input
                  type="text"
                  value={promptInput}
                  onChange={(event) => setPromptInput(event.target.value)}
                  placeholder={jobId ? "Ask a tactical follow-up..." : "No active job id for chat"}
                  className="w-full bg-[#111a12] border border-gray-800 rounded-[24px] pl-6 pr-14 py-4 text-sm text-gray-200 focus:outline-none focus:border-emerald-500/50 focus:bg-[#1a241c] hover:bg-[#1a241c] transition-all font-sans placeholder-gray-600 shadow-inner"
                />
                <button
                  type="submit"
                  disabled={!promptInput.trim() || isTyping || !jobId}
                  className="absolute right-2 top-1/2 -translate-y-1/2 p-2.5 bg-emerald-500/10 text-emerald-500 hover:bg-emerald-500 hover:text-black rounded-full transition-all disabled:opacity-50 disabled:cursor-not-allowed group"
                >
                  <Send size={16} />
                </button>
              </form>
              <div className="text-center mt-3 text-[10px] text-gray-600 font-sans tracking-wide">
                Gaffer&apos;s Engine can make mistakes. Verify telemetry data.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
