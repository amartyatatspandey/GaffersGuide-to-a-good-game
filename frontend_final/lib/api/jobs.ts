import { getApiBaseUrl } from '../apiBase';

export interface JobResponse {
  job_id: string;
  status: string;
}

export async function createJob(file: File): Promise<JobResponse> {
  const formData = new FormData();
  // Ensure the backend expects 'file' or adapt to 'video' based on standard contract
  formData.append('file', file);
  
  const res = await fetch(`${getApiBaseUrl()}/api/v1/jobs`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    throw new Error(`Failed to create job: ${res.statusText}`);
  }

  return res.json();
}

export async function getTracking(jobId: string) {
  const res = await fetch(`${getApiBaseUrl()}/api/v1/jobs/${jobId}/artifacts/tracking`);
  if (!res.ok) {
    throw new Error('Tracking not ready or not found');
  }
  return res.json();
}
