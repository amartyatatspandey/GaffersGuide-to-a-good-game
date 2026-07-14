/**
 * Lineup / team-setup types for Gaffer's Guide.
 *
 * `Lineup` mirrors `TeamSetup` in `MatchSetup.tsx` — same field names and shape
 * so existing code can adopt this type with no data reshaping.
 */

import type { Player } from './player';

/** One team's lineup as stored inside a `MatchSetup` record. */
export interface Lineup {
  name: string;
  formation: string;
  players: Player[];
}

/**
 * A compact fixture descriptor returned by `GET /api/v1/fixtures/search`.
 * Used to populate the fixture search dropdown before a full lineup is fetched.
 */
export interface FixtureSearchResult {
  /** API-Football fixture ID. */
  id: string;
  name: string;        // e.g. "PSG vs Inter"
  competition: string; // e.g. "Champions League Final"
  date: string;        // ISO 8601
  venue?: string;
  homeTeam: string;
  awayTeam: string;
}

/**
 * Full lineup detail returned by `GET /api/v1/fixtures/{id}/lineup`.
 * Wraps two `Lineup` objects (home / away) plus fixture-level metadata.
 */
export interface FixtureLineup {
  fixture_id: string;
  fixture_name: string;
  competition: string;
  team_a: Lineup;
  team_b: Lineup;
}
