import { FormEvent, useEffect, useMemo, useState } from "react";
import {
  ChunkingPolicy,
  DecoderMode,
  HardwareProfile,
  QualityMode,
  RuntimeTarget,
  SlaTier,
  fetchAdvice,
  fetchJobState,
  fetchMetrics,
  fetchReports,
  fetchTracking,
  sendChat,
  uploadJob,
  wsJobUrl,
} from "./api";

export function App() {
  const [file, setFile] = useState<File | null>(null);
  const [decoderMode, setDecoderMode] = useState<DecoderMode>("opencv");
  const [runtimeTarget, setRuntimeTarget] = useState<RuntimeTarget>("nvidia");
  const [hardwareProfile, setHardwareProfile] = useState<HardwareProfile>("l4");
  const [qualityMode, setQualityMode] = useState<QualityMode>("balanced");
  const [chunkingPolicy, setChunkingPolicy] = useState<ChunkingPolicy>("fixed");
  const [maxParallelChunks, setMaxParallelChunks] = useState<number>(2);
  const [targetSlaTier, setTargetSlaTier] = useState<SlaTier>("tier_10m");
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("idle");
  const [step, setStep] = useState<string>("Pending");
  const [stage, setStage] = useState<string>("queued");
  const [runtimeSummary, setRuntimeSummary] = useState<string>("nvidia:l4");
  const [telemetry, setTelemetry] = useState<Record<string, unknown>>({});
  const [chunks, setChunks] = useState<Array<{ chunk_id: string; start_frame: number; end_frame: number }>>([]);
  const [advice, setAdvice] = useState<string>("");
  const [chatReply, setChatReply] = useState<string>("");
  const [trackingSummary, setTrackingSummary] = useState<string>("");
  const [reportsCount, setReportsCount] = useState<number>(0);
  const [metricsSnapshot, setMetricsSnapshot] = useState<string>("");
  const [chatInput, setChatInput] = useState<string>("How should we improve shape?");
  const [error, setError] = useState<string>("");

  const canRun = useMemo(() => file !== null && status !== "processing", [file, status]);

  useEffect(() => {
    if (!jobId) return;
    const ws = new WebSocket(wsJobUrl(jobId));
    ws.onmessage = (event) => {
      const payload = JSON.parse(event.data) as {
        status?: string;
        current_step?: string;
        stage?: string;
        runtime_target?: RuntimeTarget;
        hardware_profile?: HardwareProfile;
        telemetry?: Record<string, unknown>;
        chunks?: Array<{ chunk_id: string; start_frame: number; end_frame: number }>;
        error?: string;
      };
      setStatus(payload.status ?? "unknown");
      setStep(payload.current_step ?? "Unknown");
      setStage(payload.stage ?? "unknown");
      setRuntimeSummary(
        `${payload.runtime_target ?? "unknown"}:${payload.hardware_profile ?? "unknown"}`
      );
      setTelemetry(payload.telemetry ?? {});
      setChunks(payload.chunks ?? []);
      if (payload.error) setError(payload.error);
    };
    return () => ws.close();
  }, [jobId]);

  const runAnalysis = async () => {
    if (!file) return;
    setError("");
    setAdvice("");
    setChatReply("");
    setTrackingSummary("");
    const res = await uploadJob(file, decoderMode, {
      runtimeTarget,
      hardwareProfile,
      qualityMode,
      chunkingPolicy,
      maxParallelChunks,
      targetSlaTier,
    });
    setJobId(res.job_id);
    setStatus("pending");
  };

  const loadOutputs = async () => {
    if (!jobId) return;
    try {
      const [a, t, r, s, m] = await Promise.all([
        fetchAdvice(jobId),
        fetchTracking(jobId),
        fetchReports(),
        fetchJobState(jobId),
        fetchMetrics(),
      ]);
      setAdvice(JSON.stringify(a, null, 2));
      const trackingObj = t as { telemetry?: { total_frames_processed?: number } };
      setTrackingSummary(
        `Frames processed: ${trackingObj.telemetry?.total_frames_processed ?? "unknown"}`
      );
      const reportsObj = r as { reports?: unknown[] };
      setReportsCount((reportsObj.reports ?? []).length);
      setStage(s.stage ?? "unknown");
      setRuntimeSummary(
        `${s.runtime_target ?? "unknown"}:${s.hardware_profile ?? "unknown"}`
      );
      setTelemetry(s.telemetry ?? {});
      setChunks(s.chunks ?? []);
      setMetricsSnapshot(JSON.stringify(m, null, 2));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Output load failed");
    }
  };

  const onChat = async (e: FormEvent) => {
    e.preventDefault();
    if (!jobId) return;
    try {
      const res = await sendChat(jobId, chatInput);
      setChatReply(res.reply);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Chat failed");
    }
  };

  return (
    <div style={{ padding: 20, fontFamily: "Inter, system-ui", color: "#d1d5db", background: "#0b1020", minHeight: "100vh" }}>
      <h1>Gaffers Experiment Desktop</h1>
      <p>Isolated client targeting only <code>/api/exp/*</code> and <code>/ws/exp/*</code>.</p>
      <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 16 }}>
        <input
          type="file"
          accept="video/mp4"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        <select value={decoderMode} onChange={(e) => setDecoderMode(e.target.value as DecoderMode)}>
          <option value="opencv">OpenCV</option>
          <option value="pyav">PyAV</option>
        </select>
        <select value={runtimeTarget} onChange={(e) => setRuntimeTarget(e.target.value as RuntimeTarget)}>
          <option value="nvidia">NVIDIA</option>
          <option value="apple_mps">Apple MPS</option>
          <option value="cpu_fallback">CPU fallback</option>
        </select>
        <select value={hardwareProfile} onChange={(e) => setHardwareProfile(e.target.value as HardwareProfile)}>
          <option value="l4">L4</option>
          <option value="a10">A10</option>
          <option value="a100">A100</option>
          <option value="mps">MPS</option>
          <option value="cpu">CPU</option>
        </select>
        <select value={qualityMode} onChange={(e) => setQualityMode(e.target.value as QualityMode)}>
          <option value="fast">Fast</option>
          <option value="balanced">Balanced</option>
          <option value="high">High</option>
        </select>
        <select value={chunkingPolicy} onChange={(e) => setChunkingPolicy(e.target.value as ChunkingPolicy)}>
          <option value="fixed">Fixed</option>
          <option value="auto">Auto</option>
          <option value="none">None</option>
        </select>
        <select value={targetSlaTier} onChange={(e) => setTargetSlaTier(e.target.value as SlaTier)}>
          <option value="tier_10m">Tier {"<="}10m</option>
          <option value="tier_5m">Tier {"<="}5m</option>
        </select>
        <input
          type="number"
          min={1}
          max={32}
          value={maxParallelChunks}
          onChange={(e) => setMaxParallelChunks(Number(e.target.value))}
          style={{ width: 80 }}
          title="max parallel chunks"
        />
        <button disabled={!canRun} onClick={runAnalysis}>
          Start Analysis
        </button>
        <button disabled={!jobId || status !== "done"} onClick={loadOutputs}>
          Load Outputs
        </button>
      </div>

      <div style={{ marginBottom: 12 }}>
        <strong>Job:</strong> {jobId ?? "none"} | <strong>Status:</strong> {status} |{" "}
        <strong>Step:</strong> {step} | <strong>Stage:</strong> {stage} |{" "}
        <strong>Runtime:</strong> {runtimeSummary}
      </div>
      <details style={{ marginBottom: 12 }}>
        <summary>Stage telemetry</summary>
        <pre>{JSON.stringify(telemetry, null, 2)}</pre>
      </details>
      <details style={{ marginBottom: 12 }}>
        <summary>Chunk plan ({chunks.length})</summary>
        <pre>{JSON.stringify(chunks, null, 2)}</pre>
      </details>

      <form onSubmit={onChat} style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        <input
          value={chatInput}
          onChange={(e) => setChatInput(e.target.value)}
          style={{ flex: 1 }}
        />
        <button type="submit" disabled={!jobId}>
          Send Chat
        </button>
      </form>

      {chatReply && <pre>{chatReply}</pre>}
      {trackingSummary && <p>{trackingSummary}</p>}
      {reportsCount > 0 && <p>Reports available: {reportsCount}</p>}
      {metricsSnapshot && (
        <details>
          <summary>Metrics snapshot</summary>
          <pre>{metricsSnapshot}</pre>
        </details>
      )}
      {advice && (
        <details>
          <summary>Advice payload</summary>
          <pre style={{ whiteSpace: "pre-wrap" }}>{advice}</pre>
        </details>
      )}
      {error && <p style={{ color: "#f87171" }}>{error}</p>}
    </div>
  );
}
