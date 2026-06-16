/**
 * React hook for chunked video uploads with progress, retry, cancel, and resume.
 *
 * Usage:
 *   const upload = useChunkedUpload();
 *   upload.startUpload(file, { cvEngine, llmEngine, ... });
 *   // render upload.progress, upload.phase, upload.speed, upload.timeRemaining
 */

import { useState, useCallback, useRef } from 'react';
import {
  CHUNK_SIZE,
  initUpload,
  uploadChunk,
  completeUpload,
  getUploadStatus,
  type JobOpts,
  type CompleteUploadResponse,
} from '@/lib/api/chunkedUpload';

// ── Types ───────────────────────────────────────────────────────────────

export type UploadPhase =
  | 'idle'
  | 'initializing'
  | 'uploading'
  | 'assembling'
  | 'done'
  | 'error';

export interface ChunkedUploadState {
  /** 0–100 overall percentage */
  progress: number;
  phase: UploadPhase;
  /** Human-readable speed, e.g. "12.4 MB/s" */
  speed: string;
  /** Human-readable ETA, e.g. "~3m 42s" */
  timeRemaining: string;
  chunksUploaded: number;
  totalChunks: number;
  error: string | null;
  /** Start (or restart) the upload. Resolves when the job is created. */
  startUpload: (file: File, opts: JobOpts) => Promise<CompleteUploadResponse>;
  /** Cancel an in-progress upload. */
  cancelUpload: () => void;
  /** Retry after an error using the same file & opts. */
  retryUpload: () => void;
  /** Reset to idle state. */
  reset: () => void;
}

// ── Constants ───────────────────────────────────────────────────────────

const MAX_RETRIES_PER_CHUNK = 3;
const RETRY_BASE_DELAY_MS = 1_000;
const SPEED_WINDOW = 5; // rolling average over last N chunks

// ── Helpers ─────────────────────────────────────────────────────────────

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatDuration(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) return '—';
  if (seconds < 60) return `~${Math.ceil(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.ceil(seconds % 60);
  if (m < 60) return `~${m}m ${s}s`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return `~${h}h ${rm}m`;
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

// ── Hook ────────────────────────────────────────────────────────────────

export function useChunkedUpload(): ChunkedUploadState {
  const [progress, setProgress] = useState(0);
  const [phase, setPhase] = useState<UploadPhase>('idle');
  const [speed, setSpeed] = useState('—');
  const [timeRemaining, setTimeRemaining] = useState('—');
  const [chunksUploaded, setChunksUploaded] = useState(0);
  const [totalChunks, setTotalChunks] = useState(0);
  const [error, setError] = useState<string | null>(null);

  // Refs for cancel/retry
  const abortRef = useRef<AbortController | null>(null);
  const lastFileRef = useRef<File | null>(null);
  const lastOptsRef = useRef<JobOpts | null>(null);
  const uploadIdRef = useRef<string | null>(null);

  const startUpload = useCallback(
    async (file: File, opts: JobOpts): Promise<CompleteUploadResponse> => {
      // Store for retry
      lastFileRef.current = file;
      lastOptsRef.current = opts;

      // Abort any previous upload
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      const numChunks = Math.ceil(file.size / CHUNK_SIZE);
      setTotalChunks(numChunks);
      setChunksUploaded(0);
      setProgress(0);
      setSpeed('—');
      setTimeRemaining('—');
      setError(null);

      // ── Phase 1: Initialize ─────────────────────────────────────────
      setPhase('initializing');

      let uploadId: string;
      let alreadyReceived: Set<number>;

      // Check if we have a previous upload_id to resume
      const prevUploadId = uploadIdRef.current;
      if (prevUploadId) {
        try {
          const status = await getUploadStatus(prevUploadId);
          uploadId = prevUploadId;
          alreadyReceived = new Set(status.received_chunks);
          setChunksUploaded(alreadyReceived.size);
          setProgress((alreadyReceived.size / numChunks) * 100);
        } catch {
          // Previous session expired — start fresh
          const initRes = await initUpload(file.name, file.size, numChunks);
          uploadId = initRes.upload_id;
          alreadyReceived = new Set();
        }
      } else {
        const initRes = await initUpload(file.name, file.size, numChunks);
        uploadId = initRes.upload_id;
        alreadyReceived = new Set();
      }

      uploadIdRef.current = uploadId;

      // ── Phase 2: Upload chunks ──────────────────────────────────────
      setPhase('uploading');

      const chunkTimesMs: number[] = []; // for speed estimation
      let uploaded = alreadyReceived.size;

      for (let i = 0; i < numChunks; i++) {
        if (controller.signal.aborted) {
          throw new DOMException('Upload cancelled', 'AbortError');
        }

        // Skip already-received chunks (resume)
        if (alreadyReceived.has(i)) continue;

        const start = i * CHUNK_SIZE;
        const end = Math.min(start + CHUNK_SIZE, file.size);
        const chunkBlob = file.slice(start, end);

        // Retry loop for this chunk
        let lastErr: Error | null = null;
        for (let attempt = 0; attempt < MAX_RETRIES_PER_CHUNK; attempt++) {
          if (controller.signal.aborted) {
            throw new DOMException('Upload cancelled', 'AbortError');
          }

          try {
            const chunkStart = Date.now();

            await uploadChunk(
              uploadId,
              i,
              chunkBlob,
              controller.signal,
              // Sub-chunk progress: interpolate between chunk boundaries
              (loaded, total) => {
                const chunkFraction = total > 0 ? loaded / total : 0;
                const overallProgress =
                  ((uploaded + chunkFraction) / numChunks) * 100;
                setProgress(Math.min(overallProgress, 99.9));
              },
            );

            const elapsed = Date.now() - chunkStart;
            chunkTimesMs.push(elapsed);

            // Update speed & ETA using rolling window
            const recentTimes = chunkTimesMs.slice(-SPEED_WINDOW);
            const avgTimeMs =
              recentTimes.reduce((a, b) => a + b, 0) / recentTimes.length;
            const bytesPerSec = (chunkBlob.size / avgTimeMs) * 1000;
            setSpeed(`${formatBytes(bytesPerSec)}/s`);

            const remainingChunks = numChunks - uploaded - 1;
            const etaSec = (remainingChunks * avgTimeMs) / 1000;
            setTimeRemaining(formatDuration(etaSec));

            uploaded++;
            setChunksUploaded(uploaded);
            setProgress((uploaded / numChunks) * 100);

            lastErr = null;
            break; // chunk succeeded
          } catch (err) {
            lastErr = err instanceof Error ? err : new Error(String(err));

            if (
              lastErr.name === 'AbortError' ||
              controller.signal.aborted
            ) {
              throw lastErr;
            }

            // Exponential backoff before retry
            if (attempt < MAX_RETRIES_PER_CHUNK - 1) {
              const delay = RETRY_BASE_DELAY_MS * Math.pow(2, attempt);
              await sleep(delay);
            }
          }
        }

        if (lastErr) {
          setPhase('error');
          setError(
            `Failed to upload chunk ${i + 1}/${numChunks} after ${MAX_RETRIES_PER_CHUNK} attempts: ${lastErr.message}`,
          );
          throw lastErr;
        }
      }

      // ── Phase 3: Assemble & create job ──────────────────────────────
      setPhase('assembling');
      setProgress(100);
      setSpeed('—');
      setTimeRemaining('—');

      try {
        const result = await completeUpload(uploadId, opts);
        setPhase('done');
        uploadIdRef.current = null; // session consumed
        return result;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        setPhase('error');
        setError(`Failed to finalize upload: ${msg}`);
        throw err;
      }
    },
    [],
  );

  const cancelUpload = useCallback(() => {
    abortRef.current?.abort();
    setPhase('idle');
    setProgress(0);
    setError(null);
    setSpeed('—');
    setTimeRemaining('—');
    // Keep uploadIdRef so we can resume if desired
  }, []);

  const retryUpload = useCallback(() => {
    const file = lastFileRef.current;
    const opts = lastOptsRef.current;
    if (!file || !opts) return;
    // startUpload will check uploadIdRef for resume
    void startUpload(file, opts);
  }, [startUpload]);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setPhase('idle');
    setProgress(0);
    setError(null);
    setSpeed('—');
    setTimeRemaining('—');
    setChunksUploaded(0);
    setTotalChunks(0);
    uploadIdRef.current = null;
    lastFileRef.current = null;
    lastOptsRef.current = null;
  }, []);

  return {
    progress,
    phase,
    speed,
    timeRemaining,
    chunksUploaded,
    totalChunks,
    error,
    startUpload,
    cancelUpload,
    retryUpload,
    reset,
  };
}
