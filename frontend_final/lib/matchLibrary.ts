/**
 * matchLibrary.ts
 * Local-first persistence for completed match analyses.
 * Stored in localStorage under `gaffer-match-library`.
 * Does NOT touch tracking, pipeline, or mapping systems.
 */

export interface LibraryMatch {
  id: string;              // job_id / report_id — unique key
  title: string;           // "PSG vs Inter" or derived from filename
  teamA: string;
  teamB: string;
  competition: string;
  videoName: string;
  duration: string;        // e.g. "90:12"
  date: string;            // ISO string of when analysis was saved
  analysisDate: string;    // ISO string of when analysis was generated
  status: 'completed' | 'processing' | 'pending' | 'error';
  tacticalScore: number;   // 0–100, max of team_0/team_1 tactical power
  tacticalPowerRed: number;
  tacticalPowerBlue: number;
  compactness: number;
  transitionSpeed: number;
  insightCount: number;
  thumbnail: string | null; // data-URL or null
  /** Serialised CoachAdviceResponse for re-opening */
  coachAdvice: any;
  /** Serialised compact telemetry summary (NOT full frames — too large) */
  telemetrySummary: any;
}

const LIBRARY_KEY = 'gaffer-match-library';

function readLibrary(): LibraryMatch[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = localStorage.getItem(LIBRARY_KEY);
    if (!raw) return [];
    return JSON.parse(raw) as LibraryMatch[];
  } catch {
    return [];
  }
}

function writeLibrary(entries: LibraryMatch[]): void {
  if (typeof window === 'undefined') return;
  try {
    localStorage.setItem(LIBRARY_KEY, JSON.stringify(entries));
  } catch (e) {
    console.error('[matchLibrary] Failed to write library:', e);
  }
}

/** Upsert a match entry. Existing entry with same id is replaced. */
export function upsertMatch(entry: LibraryMatch): void {
  const lib = readLibrary();
  const idx = lib.findIndex(m => m.id === entry.id);
  if (idx >= 0) {
    lib[idx] = entry;
  } else {
    lib.unshift(entry); // newest first
  }
  writeLibrary(lib);
}

/** Read all entries, newest first. */
export function listLibrary(): LibraryMatch[] {
  return readLibrary();
}

/** Delete by ID. Returns true if found and removed. */
export function deleteFromLibrary(id: string): boolean {
  const lib = readLibrary();
  const next = lib.filter(m => m.id !== id);
  if (next.length === lib.length) return false;
  writeLibrary(next);
  return true;
}

/** Duplicate an entry with a new ID and appended "(Copy)" suffix. */
export function duplicateInLibrary(id: string): LibraryMatch | null {
  const lib = readLibrary();
  const original = lib.find(m => m.id === id);
  if (!original) return null;
  const copy: LibraryMatch = {
    ...original,
    id: `${original.id}-copy-${Date.now()}`,
    title: `${original.title} (Copy)`,
    date: new Date().toISOString(),
  };
  lib.unshift(copy);
  writeLibrary(lib);
  return copy;
}

/** Export a single match as a downloadable JSON file. */
export function exportMatchAsJSON(match: LibraryMatch): void {
  if (typeof window === 'undefined') return;
  const payload = JSON.stringify(match, null, 2);
  const blob = new Blob([payload], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `GaffersGuide_${match.title.replace(/\s+/g, '_')}_${match.id.slice(0, 8)}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Build a LibraryMatch from the in-memory state that page.tsx holds.
 * Pass null tracking to skip telemetry summary (it's large).
 */
export function buildLibraryEntry(
  jobId: string,
  fileName: string,
  coachAdvice: any,
  tracking: any | null,
): LibraryMatch {
  const adviceItems: any[] = coachAdvice?.advice_items ?? [];
  const summary = coachAdvice?.summary_data ?? coachAdvice?.pipeline ?? {};

  // Derive tactical power from summary or advice items
  const tpRed =
    summary?.team_0?.tactical_power ??
    summary?.team_red_score ??
    0;
  const tpBlue =
    summary?.team_1?.tactical_power ??
    summary?.team_blue_score ??
    0;

  const compactness =
    summary?.team_0?.compactness ??
    summary?.team_1?.compactness ??
    0;
  const transitionSpeed =
    summary?.team_0?.transition_speed ??
    summary?.team_1?.transition_speed ??
    0;

  const title =
    fileName
      ? fileName.replace(/\.[^.]+$/, '').replace(/[_-]/g, ' ')
      : `Match ${jobId.slice(0, 8)}`;

  // Try to parse "PSG vs Inter" style titles
  const vsParts = title.split(/\s+vs\s+/i);
  const teamA = vsParts[0]?.trim() || 'Team A';
  const teamB = vsParts[1]?.trim() || 'Team B';

  // Lightweight telemetry summary — just aggregate counts, NOT full frames
  const telemetrySummary = tracking
    ? {
        total_frames: tracking.frames?.length ?? 0,
        telemetry: tracking.telemetry ?? null,
      }
    : null;

  return {
    id: jobId,
    title,
    teamA,
    teamB,
    competition: coachAdvice?.competition || coachAdvice?.metadata?.competition || '',
    videoName: fileName || '',
    duration: coachAdvice?.metadata?.duration || coachAdvice?.pipeline?.duration || '',
    date: new Date().toISOString(),
    analysisDate: coachAdvice?.generated_at || new Date().toISOString(),
    status: 'completed',
    tacticalScore: Math.round(Math.max(tpRed, tpBlue)),
    tacticalPowerRed: tpRed,
    tacticalPowerBlue: tpBlue,
    compactness: Math.round(compactness),
    transitionSpeed: Math.round(transitionSpeed),
    insightCount: adviceItems.length,
    thumbnail: null, // thumbnails are generated by MatchesHub from canvas
    coachAdvice,
    telemetrySummary,
  };
}
