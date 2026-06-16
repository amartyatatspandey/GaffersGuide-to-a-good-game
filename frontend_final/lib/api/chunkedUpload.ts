/**
 * Low-level API client for the chunked upload protocol.
 *
 * Protocol:
 *   1. initUpload()     → POST /api/v1/upload/init
 *   2. uploadChunk()    → POST /api/v1/upload/chunk   (repeated per chunk)
 *   3. completeUpload() → POST /api/v1/upload/complete
 *   4. getUploadStatus()→ GET  /api/v1/upload/{id}/status  (for resume)
 */

import { getApiBaseUrl, getAuthHeaders } from '@/lib/apiBase';

// ── Types ───────────────────────────────────────────────────────────────

export interface InitUploadResponse {
  upload_id: string;
  chunk_size: number;
}

export interface ChunkUploadResponse {
  upload_id: string;
  chunk_index: number;
  received: number;
  total: number;
  complete: boolean;
}

export interface CompleteUploadResponse {
  job_id: string;
  status: string;
}

export interface UploadStatusResponse {
  upload_id: string;
  filename: string;
  total_chunks: number;
  received_chunks: number[];
  complete: boolean;
}

export interface JobOpts {
  cvEngine: string;
  llmEngine: string;
  qualityProfile: string;
  chunkingInterval: string;
}

// ── Constants ───────────────────────────────────────────────────────────

/** Must match backend CHUNK_SIZE_BYTES (10 MB). */
export const CHUNK_SIZE = 10 * 1024 * 1024;

// ── API Functions ───────────────────────────────────────────────────────

export async function initUpload(
  filename: string,
  totalSize: number,
  totalChunks: number,
): Promise<InitUploadResponse> {
  const base = getApiBaseUrl();
  const form = new FormData();
  form.append('filename', filename);
  form.append('total_size', totalSize.toString());
  form.append('total_chunks', totalChunks.toString());

  const res = await fetch(`${base}/api/v1/upload/init`, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: form,
  });

  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Failed to initialize upload: ${detail}`);
  }

  return res.json();
}

/**
 * Upload a single chunk using XMLHttpRequest for progress events.
 *
 * Returns a promise that resolves with the server response.
 * Accepts an AbortSignal for cancellation support.
 */
export function uploadChunk(
  uploadId: string,
  chunkIndex: number,
  chunkBlob: Blob,
  signal?: AbortSignal,
  onChunkProgress?: (loaded: number, total: number) => void,
): Promise<ChunkUploadResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const base = getApiBaseUrl();

    xhr.open('POST', `${base}/api/v1/upload/chunk`);

    const headers = getAuthHeaders();
    for (const key in headers) {
      xhr.setRequestHeader(key, headers[key]);
    }

    // Wire up abort signal
    if (signal) {
      if (signal.aborted) {
        reject(new DOMException('Upload cancelled', 'AbortError'));
        return;
      }
      signal.addEventListener('abort', () => {
        xhr.abort();
        reject(new DOMException('Upload cancelled', 'AbortError'));
      });
    }

    // Sub-chunk progress via XHR upload events
    if (onChunkProgress) {
      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
          onChunkProgress(e.loaded, e.total);
        }
      });
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch {
          reject(new Error('Invalid JSON response from chunk upload'));
        }
      } else {
        reject(new Error(`Chunk upload failed (HTTP ${xhr.status}): ${xhr.responseText}`));
      }
    };

    xhr.onerror = () => reject(new Error('Network error during chunk upload'));
    xhr.ontimeout = () => reject(new Error('Chunk upload timed out'));

    // 5 minute timeout per chunk — extremely generous for 10 MB
    xhr.timeout = 5 * 60 * 1000;

    const form = new FormData();
    form.append('upload_id', uploadId);
    form.append('chunk_index', chunkIndex.toString());
    form.append('file', chunkBlob, `chunk_${chunkIndex}`);

    xhr.send(form);
  });
}

export async function completeUpload(
  uploadId: string,
  opts: JobOpts,
): Promise<CompleteUploadResponse> {
  const base = getApiBaseUrl();
  const form = new FormData();
  form.append('upload_id', uploadId);
  form.append('cv_engine', opts.cvEngine);
  form.append('llm_engine', opts.llmEngine);
  form.append('quality_profile', opts.qualityProfile);
  form.append('chunking_interval', opts.chunkingInterval);

  const res = await fetch(`${base}/api/v1/upload/complete`, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: form,
  });

  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Failed to finalize upload: ${detail}`);
  }

  return res.json();
}

export async function getUploadStatus(
  uploadId: string,
): Promise<UploadStatusResponse> {
  const base = getApiBaseUrl();
  const res = await fetch(`${base}/api/v1/upload/${uploadId}/status`, { headers: getAuthHeaders() });

  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Failed to get upload status: ${detail}`);
  }

  return res.json();
}
