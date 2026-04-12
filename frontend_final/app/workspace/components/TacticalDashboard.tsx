"use client";
import React, { useState, useRef, useEffect, useMemo } from "react";
import { InsightCard, KeywordConfig } from "./InsightCard";
import VideoHUD from "./VideoHUD";
import RadarWidget from "./radar/RadarWidget";
import { Clock, Send, Bot, User, Menu, ChevronLeft } from "lucide-react";
import { useStreamingText } from "@/hooks/useStreamingText";
import type { CoachAdviceResponse, CoachingAdviceItem } from "@/lib/api/coach";
import { postCoachChat } from "@/lib/api/coach";

interface TeamInsight {
  payload: string;
  keywords: KeywordConfig[];
}

interface TimelineData {
  time: string;
  title: string;
  minute: string;
  summary: string;
  blueTeam: TeamInsight;
  redTeam: TeamInsight;
}

function rolesToKeywords(roles: string[] | null | undefined): KeywordConfig[] {
  const palette: KeywordConfig["color"][] = ["emerald", "amber", "cyan"];
  return (roles ?? []).slice(0, 8).map((text, i) => ({
    text,
    color: palette[i % palette.length]!,
    role: text,
  }));
}

function adviceItemsToTimeline(items: CoachingAdviceItem[]): TimelineData[] {
  return items.map((item) => {
    const textBody =
      (item.tactical_instruction && item.tactical_instruction.trim()) ||
      (item.tactical_instruction_steps.length > 0
        ? item.tactical_instruction_steps.map((s, i) => `${i + 1}. ${s}`).join("\n")
        : item.evidence);
    const errSuffix = item.llm_error ? `\n\n(LLM note: ${item.llm_error})` : "";
    const bluePayload =
      item.team === "team_0"
        ? `${textBody}${errSuffix}`
        : `(Opposition read.) Finding primarily concerns **team_1**. Evidence: ${item.evidence}${errSuffix}`;
    const redPayload =
      item.team === "team_1"
        ? `${textBody}${errSuffix}`
        : `(Opposition read.) Finding primarily concerns **team_0**. Evidence: ${item.evidence}${errSuffix}`;
    const summaryBits = [
      item.matched_philosophy_author || "",
      item.evidence.slice(0, 96) + (item.evidence.length > 96 ? "…" : ""),
    ]
      .filter(Boolean)
      .join(" · ");
    return {
      time: `Frame ${item.frame_idx}`,
      title: item.flaw || "Tactical finding",
      minute: item.severity || "—",
      summary: summaryBits || item.flaw,
      blueTeam: {
        payload: bluePayload,
        keywords: item.team === "team_0" ? rolesToKeywords(item.fc25_player_roles) : [],
      },
      redTeam: {
        payload: redPayload,
        keywords: item.team === "team_1" ? rolesToKeywords(item.fc25_player_roles) : [],
      },
    };
  });
}

// Sub-component for a streaming AI chat bubble
function StreamingBubble({ text }: { text: string }) {
  const { displayedText } = useStreamingText(text, 25);

  const renderFormattedText = (raw: string) => {
    const parts = raw.split(/(\*\*.*?\*\*)/g);
    return parts.map((part, i) => {
      if (part.startsWith("**") && part.endsWith("**")) {
        return (
          <strong key={i} className="font-bold text-gray-100">
            {part.slice(2, -2)}
          </strong>
        );
      }
      return <span key={i}>{part}</span>;
    });
  };

  return (
    <div className="flex gap-4">
      <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full border border-emerald-500/50 bg-emerald-900/50 shadow-[0_0_10px_rgba(16,185,129,0.2)]">
        <Bot size={16} className="text-emerald-400" />
      </div>
      <div className="flex-1 overflow-x-auto rounded-2xl rounded-tl-none border border-gray-700/50 bg-gray-800/80 p-4 text-sm text-gray-300 [&::-webkit-scrollbar]:h-1.5 [&::-webkit-scrollbar-thumb]:bg-gray-700">
        <div className="min-w-fit pr-2">
          {renderFormattedText(displayedText)}
          <span className="ml-1 inline-block h-4 w-2 animate-pulse bg-emerald-500 align-middle shadow-[0_0_5px_rgba(16,185,129,0.8)]"></span>
        </div>
      </div>
    </div>
  );
}

interface DashboardProps {
  job: { jobId: string; file: File; tracking: any } | null;
  coachAdvice: CoachAdviceResponse | null;
  coachError: string | null;
}

export function TacticalDashboard({ job, coachAdvice, coachError }: DashboardProps) {
  const videoRef = React.useRef<HTMLVideoElement | null>(null);
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [activeIndex, setActiveIndex] = useState(0);
  const [isTimelineCollapsed, setIsTimelineCollapsed] = useState(false);

  const timeline = useMemo(
    () => adviceItemsToTimeline(coachAdvice?.advice_items ?? []),
    [coachAdvice],
  );

  useEffect(() => {
    setActiveIndex((i) => Math.min(i, Math.max(0, timeline.length - 1)));
  }, [timeline.length]);

  useEffect(() => {
    if (!job?.file) {
      setVideoSrc(null);
      return;
    }
    const url = URL.createObjectURL(job.file);
    setVideoSrc(url);
    return () => {
      URL.revokeObjectURL(url);
    };
  }, [job?.file]);

  useEffect(() => {
    // #region agent log
    const frameLen = Array.isArray(job?.tracking?.frames) ? job.tracking.frames.length : -1;
    fetch("http://127.0.0.1:7265/ingest/b94af6c0-0f3f-4385-ab39-095f9a480704", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Debug-Session-Id": "bb63ae",
      },
      body: JSON.stringify({
        sessionId: "bb63ae",
        hypothesisId: "H5",
        location: "TacticalDashboard.tsx:job-tracking",
        message: "Job / tracking snapshot for radar wiring",
        data: {
          hasJob: Boolean(job),
          jobIdPrefix: job?.jobId?.slice(0, 8) ?? null,
          hasTracking: Boolean(job?.tracking),
          frameLen,
          coachItems: coachAdvice?.advice_items?.length ?? -1,
          radarWidgetRenderedInLayout: true,
        },
        timestamp: Date.now(),
      }),
    }).catch(() => {});
    // #endregion
  }, [job, coachAdvice?.advice_items?.length]);

  const [promptInput, setPromptInput] = useState("");
  const [chatHistory, setChatHistory] = useState<{ id: string; role: "user" | "ai"; text: string }[]>(
    [],
  );
  const [isTyping, setIsTyping] = useState(false);
  const [engineFootnote, setEngineFootnote] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  const refreshEngineFootnote = (): void => {
    if (typeof window === "undefined") return;
    const e = localStorage.getItem("gaffer-engine-type");
    const m = localStorage.getItem("gaffer-ollama-model")?.trim() || "llama3";
    setEngineFootnote(
      e === "cloud"
        ? "FastAPI /api/v1/chat · cloud LLM"
        : `FastAPI /api/v1/chat · local Ollama (${m})`,
    );
  };

  useEffect(() => {
    refreshEngineFootnote();
    window.addEventListener("gaffer-engine-changed", refreshEngineFootnote);
    return () => window.removeEventListener("gaffer-engine-changed", refreshEngineFootnote);
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [chatHistory, isTyping]);

  useEffect(() => {
    setChatHistory([]);
    setPromptInput("");
  }, [activeIndex]);

  const handleDownloadTrackingJson = (): void => {
    if (!job?.tracking) return;
    const blob = new Blob([JSON.stringify(job.tracking, null, 2)], {
      type: "application/json",
    });
    const a = document.createElement("a");
    const objectUrl = URL.createObjectURL(blob);
    a.href = objectUrl;
    a.download = `${job.jobId}-tracking.json`;
    a.click();
    URL.revokeObjectURL(objectUrl);
  };

  const handleSendPrompt = async (e: React.FormEvent): Promise<void> => {
    e.preventDefault();
    if (!promptInput.trim()) return;

    const newQuery = promptInput.trim();
    setPromptInput("");
    const userId = Date.now().toString();
    setChatHistory((prev) => [...prev, { id: userId, role: "user", text: newQuery }]);
    setIsTyping(true);

    const llmEngine: "local" | "cloud" =
      typeof window !== "undefined" && localStorage.getItem("gaffer-engine-type") === "cloud"
        ? "cloud"
        : "local";

    // #region agent log
    fetch("http://127.0.0.1:7265/ingest/b94af6c0-0f3f-4385-ab39-095f9a480704", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Debug-Session-Id": "bb63ae",
      },
      body: JSON.stringify({
        sessionId: "bb63ae",
        hypothesisId: "H2",
        location: "TacticalDashboard.tsx:handleSendPrompt",
        message: "Chat submit → FastAPI /api/v1/chat",
        data: { llmEngine, hasJobId: Boolean(job?.jobId) },
        timestamp: Date.now(),
      }),
    }).catch(() => {});
    // #endregion

    try {
      const { reply } = await postCoachChat({
        message: newQuery,
        jobId: job?.jobId ?? null,
        llmEngine,
      });
      // #region agent log
      fetch("http://127.0.0.1:7265/ingest/b94af6c0-0f3f-4385-ab39-095f9a480704", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Debug-Session-Id": "bb63ae",
        },
        body: JSON.stringify({
          sessionId: "bb63ae",
          hypothesisId: "H4",
          location: "TacticalDashboard.tsx:chat-reply",
          message: "FastAPI chat ok",
          data: { replyLen: reply.length },
          timestamp: Date.now(),
        }),
      }).catch(() => {});
      // #endregion
      setChatHistory((prev) => [
        ...prev,
        { id: (Date.now() + 1).toString(), role: "ai", text: reply || "(empty reply)" },
      ]);
    } catch (err) {
      setChatHistory((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: "ai",
          text: `**Chat error:** ${err instanceof Error ? err.message : String(err)}`,
        },
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  const activeSlide = timeline[activeIndex];

  return (
    <div className="relative flex h-full w-full bg-[#0a0f0a] font-sans">
      <div
        className={`${isTimelineCollapsed ? "w-0 border-r-0 opacity-0" : "w-[30%] border-r"} flex h-full flex-shrink-0 flex-col overflow-hidden whitespace-nowrap border-gray-900 bg-[#111a12]/30 transition-all duration-300 ease-in-out`}
      >
        <div className="flex items-center justify-between border-b border-gray-900 p-4 shadow-sm">
          <h2 className="text-xs font-bold font-mono uppercase tracking-widest text-gray-500">
            Match timeline
          </h2>
          <div className="flex items-center gap-2">
            <Clock size={16} className="text-gray-600" />
            <button
              type="button"
              onClick={() => setIsTimelineCollapsed(true)}
              className="ml-1 rounded p-1 text-gray-500 transition-colors hover:bg-gray-800 hover:text-white"
            >
              <ChevronLeft size={16} />
            </button>
          </div>
        </div>

        <div className="flex-1 space-y-4 overflow-y-auto p-4 [&::-webkit-scrollbar-thumb]:bg-gray-800 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar]:w-2">
          {coachError && (
            <div className="rounded-lg border border-amber-800/60 bg-amber-950/30 p-3 text-xs text-amber-100">
              Coaching data: {coachError}
            </div>
          )}
          {timeline.length === 0 && !coachError && coachAdvice && (
            <div className="rounded-lg border border-gray-800 bg-[#0a0f0a] p-4 text-xs text-gray-500">
              <p>No coaching rows in this job report (empty advice list).</p>
              <p className="mt-2 font-mono text-[10px] text-gray-400">
                Pipeline LLM: {String(coachAdvice.pipeline?.llm ?? "—")}
              </p>
              <p className="mt-1 text-[10px] text-gray-600">
                If the rule engine produced no chunk insights, the report can be empty. Otherwise
                check backend logs and Ollama / cloud LLM configuration.
              </p>
            </div>
          )}
          {timeline.length === 0 && !coachError && !coachAdvice && (
            <div className="rounded-lg border border-gray-800 bg-[#0a0f0a] p-4 text-xs text-gray-500">
              Coaching insights load after the job report is fetched. If the pipeline just finished,
              wait a moment or refresh the page.
            </div>
          )}
          {timeline.map((data, i) => (
            <button
              type="button"
              key={`${data.title}-${i}`}
              onClick={() => setActiveIndex(i)}
              className={`w-full rounded-xl border p-4 text-left transition-all ${
                i === activeIndex
                  ? "border-emerald-500/30 bg-[#111a12] shadow-[0_0_15px_rgba(16,185,129,0.05)]"
                  : "border-gray-800 bg-[#0a0f0a] hover:bg-[#111a12]/80"
              }`}
            >
              <div
                className={`mb-1 font-mono text-[10px] tracking-widest ${i === activeIndex ? "text-emerald-500" : "text-gray-500"}`}
              >
                {data.time}
              </div>
              <div className="font-sans text-sm font-bold tracking-tight text-gray-300">{data.title}</div>
              <div className="mt-2 truncate font-sans text-xs text-gray-500">{data.summary}</div>
            </button>
          ))}
        </div>
      </div>

      <div className="flex min-h-0 min-w-0 flex-1 flex-col bg-[#050805]">
        {/* Video + radar side by side */}
        <div className="relative flex h-[45%] min-h-0 flex-shrink-0 flex-col border-b border-gray-900 bg-[#050805]">
          <div className="absolute left-4 top-4 z-30 flex items-center gap-2 font-mono text-[10px] font-bold uppercase tracking-widest text-gray-600">
            {isTimelineCollapsed && (
              <button
                type="button"
                onClick={() => setIsTimelineCollapsed(false)}
                className="mr-2 rounded p-1 text-gray-500 transition-colors hover:bg-gray-800 hover:text-emerald-400"
              >
                <Menu size={14} />
              </button>
            )}
            <span className="h-2 w-2 animate-pulse rounded-full bg-red-600" />
            Live IPC feed
          </div>

          <div className="flex min-h-0 flex-1 flex-row gap-4 px-6 pb-4 pt-12">
            <div className="flex min-h-0 min-w-0 flex-1 flex-col">
              <VideoHUD
                videoRef={videoRef}
                videoSrc={videoSrc}
                jobId={job?.jobId ?? null}
                status={job?.tracking ? "completed" : "pending"}
                onDownload={handleDownloadTrackingJson}
              />
            </div>
            <div className="flex w-[min(380px,36vw)] min-w-[260px] shrink-0 flex-col">
              <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
                <RadarWidget videoRef={videoRef} trackingData={job?.tracking ?? null} />
              </div>
            </div>
          </div>
        </div>

        {/* Insights + chat */}
        <div className="relative flex min-h-0 flex-1 flex-row bg-[#0a0f0a]">
          <div className="flex min-h-0 w-[60%] flex-col border-r border-gray-900 bg-[#050805]">
            <div className="min-h-0 flex-1 overflow-y-auto p-8 [&::-webkit-scrollbar-thumb]:bg-gray-800 [&::-webkit-scrollbar]:w-2">
              <div className="mx-auto w-full max-w-3xl space-y-6">
                <div className="flex items-center justify-between font-mono text-[10px] font-bold uppercase tracking-widest text-gray-500">
                  <span>Proactive insight generation</span>
                  <span className="text-emerald-500">Backend report</span>
                </div>

                {activeSlide ? (
                  <InsightCard
                    key={activeIndex}
                    title={activeSlide.title}
                    minute={activeSlide.minute}
                    blueTeam={activeSlide.blueTeam}
                    redTeam={activeSlide.redTeam}
                  />
                ) : (
                  <div className="rounded-lg border border-gray-800 p-6 text-sm text-gray-500">
                    Select a timeline item to view coaching detail.
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="relative flex w-[40%] flex-col bg-[#0a0f0a]">
            <div
              ref={scrollRef}
              className="flex-1 space-y-6 overflow-y-auto p-6 pb-24 [&::-webkit-scrollbar-thumb]:bg-gray-800 [&::-webkit-scrollbar]:w-2"
            >
              {chatHistory.length === 0 && (
                <div className="mt-8 flex h-full flex-col items-center justify-center space-y-4 text-center opacity-70">
                  <div className="flex h-12 w-12 items-center justify-center rounded-full border border-emerald-500/30 bg-emerald-900/30">
                    <Bot size={24} className="text-emerald-500" />
                  </div>
                  <div className="space-y-2">
                    <p className="font-sans text-sm font-semibold text-gray-200">Tactical engine ready</p>
                    <p className="mx-auto max-w-[250px] font-sans text-xs leading-relaxed text-gray-500">
                      Ask follow-ups; replies use your engine setting (local Ollama / Llama 3 by default)
                      with job report context when available.
                    </p>
                  </div>
                </div>
              )}

              {chatHistory.map((msg) =>
                msg.role === "user" ? (
                  <div key={msg.id} className="animate-fade-in-up flex justify-end gap-3">
                    <div className="max-w-[85%] rounded-2xl rounded-tr-none border border-gray-800 bg-[#111a12] p-3.5 font-sans text-sm text-gray-200 shadow-sm">
                      {msg.text}
                    </div>
                    <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full border border-gray-700 bg-gray-800">
                      <User size={14} className="text-gray-400" />
                    </div>
                  </div>
                ) : (
                  <div key={msg.id} className="animate-fade-in-up">
                    <StreamingBubble text={msg.text} />
                  </div>
                ),
              )}

              {isTyping && (
                <div className="animate-fade-in flex gap-3">
                  <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full border border-emerald-500/50 bg-emerald-900/50 shadow-[0_0_10px_rgba(16,185,129,0.2)]">
                    <Bot size={14} className="animate-pulse text-emerald-400" />
                  </div>
                  <div className="flex items-center gap-1 rounded-2xl rounded-tl-none border border-gray-700/50 bg-gray-800/80 p-4 text-sm">
                    <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-500" />
                    <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-500 [animation-delay:0.2s]" />
                    <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-500 [animation-delay:0.4s]" />
                  </div>
                </div>
              )}
            </div>

            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-[#0a0f0a] via-[#0a0f0a] to-transparent p-5">
              <form
                onSubmit={handleSendPrompt}
                className="relative mx-auto flex w-full max-w-lg items-center shadow-[0_10px_40px_rgba(0,0,0,0.8)]"
              >
                <input
                  type="text"
                  value={promptInput}
                  onChange={(e) => setPromptInput(e.target.value)}
                  placeholder="Ask a tactical follow-up..."
                  className="w-full rounded-[24px] border border-gray-800 bg-[#111a12] py-4 pl-6 pr-14 font-sans text-sm text-gray-200 shadow-inner transition-all placeholder-gray-600 hover:bg-[#1a241c] focus:border-emerald-500/50 focus:bg-[#1a241c] focus:outline-none"
                />
                <button
                  type="submit"
                  disabled={!promptInput.trim() || isTyping}
                  className="group absolute right-2 top-1/2 -translate-y-1/2 rounded-full p-2.5 text-emerald-500 transition-all hover:bg-emerald-500 hover:text-black disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <Send
                    size={16}
                    className={`transition-transform ${promptInput.trim() && !isTyping ? "group-hover:scale-110" : ""}`}
                  />
                </button>
              </form>
              <div className="mt-3 text-center font-sans text-[10px] tracking-wide text-gray-600">
                Gaffer&apos;s engine can make mistakes. Verify telemetry data.
                <span className="mt-1 block font-mono text-gray-500">Chat: {engineFootnote || "…"}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
