/**
 * playerMappingUtils.ts
 *
 * Shared helpers for reading the `gaffer-player-mappings` localStorage key
 * and resolving tracked player IDs to real identity data (name, jersey, team).
 *
 * DO NOT import anything from Next.js or React here — this must stay pure TS
 * so it can be called safely inside canvas draw loops and SSR guards.
 */

export interface MappedPlayer {
  id: string;
  name: string;
  number: number;
  position: "GK" | "DF" | "MF" | "FW";
  team: "A" | "B";
}

export interface JobPlayerMappings {
  setup_id: string;
  team_a_name: string;
  team_b_name: string;
  /** Key is the tracked player numeric ID as a string */
  mappings: Record<string, MappedPlayer>;
}

/**
 * Read the saved mappings for a specific jobId from localStorage.
 * Returns null if nothing is saved or parsing fails.
 * Safe to call server-side (returns null when window is unavailable).
 */
export function loadJobMappings(jobId: string | null | undefined): JobPlayerMappings | null {
  if (!jobId || typeof window === "undefined") return null;
  try {
    const raw = localStorage.getItem("gaffer-player-mappings");
    if (!raw) return null;
    const all = JSON.parse(raw) as Record<string, JobPlayerMappings>;
    return all[jobId] ?? null;
  } catch {
    return null;
  }
}

/**
 * Resolve a tracked numeric player ID to a display label.
 *
 * Resolution priority:
 *  1. savedMappings (PlayerMapping page)
 *  2. dictionary + useAltNames (DictionaryTab)
 *  3. Fallback: "Player {id}"
 */
export function resolvePlayerLabel(
  playerId: number | string,
  opts: {
    savedMappings?: JobPlayerMappings | null;
    useAltNames?: boolean;
    dictionary?: Record<string, string>;
    /** When true, prepend jersey number: "#10 Mbappe" */
    includeNumber?: boolean;
  } = {}
): string {
  const { savedMappings, useAltNames, dictionary, includeNumber } = opts;
  const key = String(playerId);

  if (savedMappings?.mappings?.[key]) {
    const p = savedMappings.mappings[key];
    if (includeNumber) return `#${p.number} ${p.name}`;
    return p.name;
  }

  const dictKey = `P${playerId}`;
  if (useAltNames && dictionary?.[dictKey]) {
    return dictionary[dictKey];
  }

  return `Player ${playerId}`;
}

/**
 * Resolve a team_id ("team_0" | "team_1") to a display name.
 *
 * Resolution priority:
 *  1. savedMappings team_a_name / team_b_name
 *     (team_0 = "Red" = Team A by convention used in PlayerMapping.tsx)
 *  2. dictionary + useAltNames
 *  3. Fallback: "Red Team" / "Blue Team"
 */
export function resolveTeamLabel(
  teamId: string | null | undefined,
  opts: {
    savedMappings?: JobPlayerMappings | null;
    useAltNames?: boolean;
    dictionary?: Record<string, string>;
    /** Short label: "Red" vs "Red Team" */
    short?: boolean;
  } = {}
): string {
  const { savedMappings, useAltNames, dictionary, short } = opts;

  if (!teamId || teamId === "unknown") return "Unclassified";

  if (savedMappings) {
    if (teamId === "team_0" && savedMappings.team_a_name) return savedMappings.team_a_name;
    if (teamId === "team_1" && savedMappings.team_b_name) return savedMappings.team_b_name;
  }

  if (useAltNames && dictionary?.[teamId]) return dictionary[teamId];

  if (short) return teamId === "team_0" ? "Red" : "Blue";
  return teamId === "team_0" ? "Red Team" : "Blue Team";
}
