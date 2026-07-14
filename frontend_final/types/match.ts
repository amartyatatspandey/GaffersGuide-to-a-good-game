/**
 * Match / fixture types for Gaffer's Guide.
 *
 * `MatchSetup` is the canonical stored record — a 1-to-1 mirror of the
 * existing `MatchSetupData` interface (MatchSetup.tsx lines 24-34) so the
 * localStorage key `gaffer-match-setups` requires no migration.
 *
 * Nothing downstream (PlayerMapping.tsx, the save/load path) needs to change
 * when Phase 4 starts writing real data here — the shape is identical.
 */

import type { Lineup } from './lineup';

/**
 * The record persisted to `localStorage['gaffer-match-setups']`.
 * Field names and types are kept byte-for-byte compatible with
 * `MatchSetupData` in MatchSetup.tsx so existing read/write code
 * continues to work without modification.
 */
export interface MatchSetup {
  match_id: string;
  fixture_name: string;
  competition: string;
  video_filename: string;
  video_size: string;
  video_duration: string;
  team_a: Lineup;
  team_b: Lineup;
  created_at: string;
  /**
   * API-Football fixture ID, present once the user picks a live fixture
   * (Phase 4 onward). Absent for manually-entered custom fixtures.
   */
  api_fixture_id?: string;
}

/** Re-export sub-types for consumers that import from a single location. */
export type { Lineup } from './lineup';
export type { Player, PlayerPosition } from './player';
export type { FixtureSearchResult, FixtureLineup } from './lineup';
