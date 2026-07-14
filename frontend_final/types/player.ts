/**
 * Canonical player types for Gaffer's Guide.
 *
 * `Player` matches `MatchSetupPlayer` (MatchSetup.tsx lines 10-16 and 14-20,
 * PlayerMapping.tsx lines 14-20) so it slots directly into the existing
 * localStorage `gaffer-match-setups` handoff without reshaping.
 *
 * `PlayerPosition` is the closed union used throughout — identical to the
 * position literals already enforced in PlayerMapping.tsx.
 */

export type PlayerPosition = 'GK' | 'DF' | 'MF' | 'FW';

/** A single player entry as stored in a lineup / match setup. */
export interface Player {
  /** Stable string ID, e.g. "psg-7" (local) or "<api-football-player-id>" (live). */
  id: string;
  name: string;
  number: number;
  position: PlayerPosition;
  isStarting: boolean;
}

/**
 * Partial player shape returned by API-Football lineup endpoints before
 * it is normalised into a full `Player`. Only used inside the football
 * provider layer (Phase 2) — consumers should always receive `Player`.
 */
export interface ApiFootballPlayer {
  player: {
    id: number;
    name: string;
    number: number;
    pos: string; // 'G' | 'D' | 'M' | 'F'
  };
  statistics?: Array<{ games: { minutes: number | null } }>;
}

/** Maps API-Football position codes to our internal `PlayerPosition`. */
export function normalisePosition(apiPos: string): PlayerPosition {
  switch (apiPos?.toUpperCase()) {
    case 'G':  return 'GK';
    case 'D':  return 'DF';
    case 'M':  return 'MF';
    case 'F':  return 'FW';
    default:   return 'MF';
  }
}
