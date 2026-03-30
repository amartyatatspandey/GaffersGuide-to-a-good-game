import { useMemo, useRef, useState } from "react";
import RadarWidget from "../components/radar/RadarWidget";
import type { TrackingFrame, TrackingPayload } from "../lib/trackingTypes";

type JobStatus = "pending" | "processing" | "done" | "error";

interface HomeProps {
  baseUrl: string;
}

interface JobCreateResponse {
  job_id: string;
  status: JobStatus;
  cv_engine: "local" | "cloud";
  llm_engine: "local" | "cloud";
}

function toWsUrl(apiBase: string, jobId: string): string {
  if (!apiBase) {
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    return `${proto}://${window.location.host}/ws/jobs/${jobId}`;
  }
  const wsBase = apiBase.replace(/^http/, "ws");
  return `${wsBase}/ws/jobs/${jobId}`;
}

export default function Home({ baseUrl }: HomeProps) {
  const apiBase = baseUrl ? baseUrl.replace(/\/$/, "") : "";
  const [file, setFile] = useState<File | null>(null);
  const [cvEngine, setCvEngine] = useState<"local" | "cloud">("local");
  const [llmEngine, setLlmEngine] = useState<"local" | "cloud">("cloud");
  const [jobId, setJobId] = useState<string>("");
  const [status, setStatus] = useState<JobStatus | "idle">("idle");
  const [step, setStep] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [currentFrame, setCurrentFrame] = useState<number>(0);
  const [trackingLoaded, setTrackingLoaded] = useState(false);
  const [activeFrameFallback, setActiveFrameFallback] = useState<boolean | null>(null);
  const trackingRef = useRef<TrackingPayload | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const overlayUrl = useMemo(() => {
    if (!jobId) return "";
    return apiBase
      ? `${apiBase}/api/v1/jobs/${jobId}/overlay`
      : `/api/v1/jobs/${jobId}/overlay`;
  }, [apiBase, jobId]);

  const fetchTracking = async (id: string): Promise<void> => {
    const url = apiBase
      ? `${apiBase}/api/v1/jobs/${id}/tracking`
      : `/api/v1/jobs/${id}/tracking`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Tracking fetch failed: HTTP ${res.status}`);
    const json = (await res.json()) as TrackingPayload;
    trackingRef.current = json;
    setTrackingLoaded(true);
  };

  const startJob = async (): Promise<void> => {
    if (!file) {
      setError("Select an MP4 file first.");
      return;
    }
    setError("");
    trackingRef.current = null;
    setTrackingLoaded(false);
    setActiveFrameFallback(null);
    setCurrentFrame(0);
    setStatus("pending");
    setStep("Uploading");

    const form = new FormData();
    form.append("file", file);
    form.append("cv_engine", cvEngine);
    form.append("llm_engine", llmEngine);

    const createUrl = apiBase ? `${apiBase}/api/v1/jobs` : "/api/v1/jobs";
    const res = await fetch(createUrl, { method: "POST", body: form });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`Job creation failed: HTTP ${res.status} ${txt}`);
    }
    const payload = (await res.json()) as JobCreateResponse;
    setJobId(payload.job_id);
    setStatus(payload.status);

    const ws = new WebSocket(toWsUrl(apiBase, payload.job_id));
    ws.onmessage = async (event) => {
      const msg = JSON.parse(event.data) as {
        status: JobStatus;
        current_step: string;
        error?: string;
      };
      setStatus(msg.status);
      setStep(msg.current_step);
      if (msg.status === "done") {
        ws.close();
        try {
          await fetchTracking(payload.job_id);
        } catch (e) {
          setError(e instanceof Error ? e.message : "Failed to load tracking data.");
        }
      } else if (msg.status === "error") {
        ws.close();
        setError(msg.error ?? "Job failed.");
      }
    };
    ws.onerror = () => setError("WebSocket disconnected.");
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="mb-2 text-2xl font-bold text-gray-900">CV Pipeline + Tactical Radar</h1>
        <p className="text-sm text-gray-600">
          Upload a match, run the full CV pipeline, then review synchronized overlay video and radar frames.
        </p>
      </div>

      <div className="grid gap-4 rounded-lg border bg-white p-4 shadow-sm md:grid-cols-4">
        <input
          type="file"
          accept=".mp4,video/mp4"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          className="md:col-span-2"
        />
        <select
          value={cvEngine}
          onChange={(e) => setCvEngine(e.target.value as "local" | "cloud")}
          className="rounded border px-2 py-1"
        >
          <option value="local">CV: local</option>
          <option value="cloud">CV: cloud</option>
        </select>
        <select
          value={llmEngine}
          onChange={(e) => setLlmEngine(e.target.value as "local" | "cloud")}
          className="rounded border px-2 py-1"
        >
          <option value="cloud">LLM: cloud</option>
          <option value="local">LLM: local</option>
        </select>
        <button
          onClick={() => void startJob()}
          className="rounded bg-blue-600 px-4 py-2 font-medium text-white hover:bg-blue-700 md:col-span-4"
        >
          Start Analysis
        </button>
      </div>

      <div className="rounded-lg border bg-white p-4 text-sm shadow-sm">
        <p><strong>Job:</strong> {jobId || "-"}</p>
        <p><strong>Status:</strong> {status}</p>
        <p><strong>Step:</strong> {step || "-"}</p>
        {trackingRef.current && (
          <p>
            <strong>Optical-flow fallback frames:</strong>{" "}
            {trackingRef.current.telemetry.frames_optical_flow_fallback}
          </p>
        )}
        {activeFrameFallback != null && (
          <p>
            <strong>Current frame fallback:</strong>{" "}
            {activeFrameFallback ? "yes" : "no"}
          </p>
        )}
        {error && <p className="text-red-600"><strong>Error:</strong> {error}</p>}
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-lg border bg-white p-3 shadow-sm">
          <h2 className="mb-2 text-sm font-semibold text-gray-800">Overlay Video</h2>
          {jobId ? (
            <video ref={videoRef} controls className="w-full rounded" src={overlayUrl} />
          ) : (
            <p className="text-sm text-gray-500">Run a job to load video.</p>
          )}
        </div>
        <RadarWidget
          videoRef={videoRef}
          trackingData={trackingLoaded ? trackingRef.current : null}
          onFrameChange={(idx: number, frame: TrackingFrame | undefined) => {
            setCurrentFrame(idx);
            setActiveFrameFallback(
              frame?.used_optical_flow_fallback != null
                ? frame.used_optical_flow_fallback
                : null,
            );
          }}
        />
      </div>
      <p className="text-xs text-gray-500">Synced frame index: {currentFrame}</p>
    </div>
  );
}
