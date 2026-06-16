import { getApiBaseUrl, getAuthHeaders } from "@/lib/apiBase";

function getReportsApiBase() {
  return `${getApiBaseUrl()}/api/v1/elite/reports`;
}

/**
 * Saves a tactical report to the persistent backend store.
 * Includes match metadata, timeline segments, coaching insights, and spatial telemetry.
 */
export async function saveTacticalReport(jobId: string, coachAdvice: any, videoFilename?: string) {
  try {
    // Derive a clean title from the uploaded filename (strip extension)
    const videoTitle = videoFilename
      ? videoFilename.replace(/\.[^.]+$/, '').replace(/[_-]/g, ' ')
      : coachAdvice?.video_title || `Match ${new Date().toLocaleDateString()}`;

    const reportData = {
      job_id: jobId,
      timestamp: new Date().toISOString(),
      video_title: videoTitle,
      metadata: coachAdvice?.metadata || {},
      pipeline: coachAdvice?.pipeline || {},
      advice_items: coachAdvice?.advice_items || [],
      summary_data: coachAdvice?.summary_data || {},
      quality_profile: coachAdvice?.quality_profile || 'balanced',
      is_manual_save: true
    };

    const response = await fetch(`${getReportsApiBase()}/save`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...getAuthHeaders(),
      },
      body: JSON.stringify(reportData),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Failed to save report: ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Error in saveTacticalReport:", error);
    throw error;
  }
}


/**
 * Fetches the list of all saved reports.
 */
export async function listSavedReports() {
  try {
    const response = await fetch(`${getReportsApiBase()}`, {
      method: "GET",
      headers: {
        ...getAuthHeaders(),
      },
    });

    if (!response.ok) {
      throw new Error("Failed to fetch reports list");
    }

    return await response.json();
  } catch (error) {
    console.error("Error in listSavedReports:", error);
    return [];
  }
}

/**
 * Deletes a saved report by ID.
 */
export async function deleteSavedReport(reportId: string) {
  try {
    const response = await fetch(`${getReportsApiBase()}/${reportId}`, {
      method: "DELETE",
      headers: {
        ...getAuthHeaders(),
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Failed to delete report: ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Error in deleteSavedReport:", error);
    throw error;
  }
}

