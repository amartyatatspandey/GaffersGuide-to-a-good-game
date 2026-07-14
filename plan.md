# Plan: Dynamic Match Identification & Player Mapping

Source: `Gaffers_Guide_Dynamic_Match_Identification_Feature_Specification.docx`
Updated against real codebase investigation (see notes below).

Goal: replace the hardcoded fixtures/lineups in `MatchSetup.tsx` and
`fixtureImportService.ts` with a live search-and-select flow backed by
API-Football, without touching CV/tracking/analysis/reports, and without
changing how `PlayerMapping.tsx` reads its data (localStorage `gaffer-match-setups`
+ `/api/v1/player-mappings/{job_id}`).

### Confirmed real architecture (from Cursor investigation)
- Hardcoded fixture list: `MatchSetup.tsx` lines 37-44, `MOCK_FIXTURES` (6 entries)
- Hardcoded lineup data: `lib/services/fixtureImportService.ts` lines 22-311, `REAL_LINEUPS`, accessed via `getFixtureLineup(fixtureId)`
- Selection handler: `MatchSetup.tsx` `handleSelectFixture` (lines 169-188)
- Data handoff: saved to `localStorage['gaffer-match-setups']` (`MatchSetupData` shape, lines 24-34, 253-281) — **not** props or a direct API call
- `PlayerMapping.tsx` reads that localStorage key (lines 212-223) plus `job.tracking.frames` (lines 274-380), and fetches/saves mappings via `GET/POST /api/v1/player-mappings/{job_id}`
- Auto-suggestion UI already exists in `PlayerMapping.tsx` at lines 724-756 — likely where confidence % suggestions render, above manual assignment (lines 780-809)
- `Hopper.tsx` signals analysis-complete via `onComplete(jobId, file)` when WebSocket step === `'Completed'`
- `page.tsx` switches views via `currentView`: `'match_setup'` renders `<MatchSetup />`, `'player_mapping'` renders `<PlayerMapping job={activeJob} />`
- Backend already has a `matches` router (`/api/v1/matches`, `{match_id}`, `{match_id}/reanalyze`) — those are **saved analyzed-match records**, unrelated to fixture search. New fixture-search routes must not collide with `/api/v1/matches/{match_id}` — put them under a distinct path, e.g. `/api/v1/fixtures/search` (see Phase 1 note)
- **Auth is required on `/api/v1/*` calls.** Frontend convention: `lib/apiBase.ts` exports `getApiBaseUrl()` and `getAuthHeaders()`. Every existing caller (`lib/api/jobs.ts`, `coach.ts`, `chunkedUpload.ts`, `reports.ts`) passes `headers: getAuthHeaders()` manually — there is no global fetch interceptor. `getAuthHeaders()` reads a Supabase JWT from `localStorage['gaffer-supabase-token']` and attaches it as `Authorization: Bearer <token>`. **`footballApi.ts` (Phase 3/4) must follow this same pattern** or every fixture search will silently 401 in the real app.

---

## Phase 0 — Setup & Types
Lay the groundwork before writing any UI or API code.
- Create `types/match.ts`, `types/lineup.ts`, `types/player.ts` (Match, Player, Lineup shapes) — model these on `MatchSetupData` (lines 24-34 of `MatchSetup.tsx`) so the new data slots into the existing localStorage handoff without reshaping it
- Use API-Football (via RapidAPI) as the provider. Add `FOOTBALL_API_KEY` as a backend env var placeholder in `backend/.env` / `.env.example`
- Do NOT delete `MOCK_FIXTURES` or `REAL_LINEUPS` yet — just confirm every place they're referenced (staged for removal in Phase 5)

**Outcome:** shared types compatible with `MatchSetupData`, and provider config in place.

---

## Phase 1 — Backend API Endpoints
Build the three endpoints the frontend will call, wrapping the external football API.

**Naming note:** the backend already has `GET /api/v1/matches`, `DELETE /api/v1/matches/{match_id}`,
and `POST /api/v1/matches/{match_id}/reanalyze` for saved analyzed-match records — an unrelated
concept. To avoid path collisions and confusion, use a distinct prefix for fixture search:
- `GET /api/v1/fixtures/search?q=` — search fixtures by team name
- `GET /api/v1/fixtures/{id}` — fixture details (teams, competition, date, venue)
- `GET /api/v1/fixtures/{id}/lineup` — official starting lineup

Other tasks:
- Add these as a new FastAPI router (new file, e.g. `routers/fixtures.py`, following whatever pattern the existing `matches` router uses)
- Normalize API-Football responses into our internal `Match`/`Player` types from Phase 0
- Handle upstream errors (provider down, no results) with clean error responses

**Outcome:** three working, testable REST endpoints, visible in `/docs`, with no path collisions against existing routers.

---

## Phase 2 — Football API Integration Layer (backend)
Isolate all third-party API-Football logic on the backend so it's swappable and testable.
This is the backend service the Phase 1 endpoints call into — distinct from the frontend's
`footballApi.ts` (Phase 3/4), which calls *our own* `/api/v1/fixtures/*` endpoints, not API-Football directly.
- `services/football_provider.py` (backend) — fetch wrapper, auth headers, response mapping
- Add basic caching (avoid re-hitting provider for repeat searches)
- Map API-Football's response fields → our internal `Match`/`Player`/`Lineup` types from Phase 0
- Export functions: `search_fixtures(query)`, `get_fixture(id)`, `get_lineup(id)` — used by the Phase 1 router

**Outcome:** a clean internal service the Phase 1 FastAPI router calls, with API-Football fully abstracted away behind it.

---

## Phase 3 — Frontend: Match Search & Selection (inside `MatchSetup.tsx`)
Replace `MOCK_FIXTURES` search with a live API call, reusing `MatchSetup.tsx`'s existing
UI shell/styling rather than building a brand-new modal from scratch.
- Add `services/footballApi.ts` (or `lib/services/footballApi.ts`, matching `fixtureImportService.ts`'s location) with `searchFixtures(query)`, calling `GET /api/v1/fixtures/search`, using `getApiBaseUrl()` and `getAuthHeaders()` from `lib/apiBase.ts` — same pattern as `lib/api/jobs.ts` (`headers: getAuthHeaders()`), otherwise calls will 401 in the real (non-bypass-auth) app
- Add `hooks/useMatchSearch.ts` wrapping search state + debounce
- In `MatchSetup.tsx`: replace the `MOCK_FIXTURES` array and its filter logic (lines ~37-44, ~160-167) with the new hook's live results — keep the existing card/list rendering markup as-is, just swap the data source
- Loading state: "Searching fixtures..."
- Empty state: "No matches found. Try another team name."

**Outcome:** the existing fixture search UI in `MatchSetup.tsx` now hits a live API instead of `MOCK_FIXTURES`, with zero visual change.

---

## Phase 4 — Frontend: Match Confirmation & Lineup Fetch (inside `MatchSetup.tsx`)
Replace `getFixtureLineup()` (local `REAL_LINEUPS` lookup) with a real API call, keeping
`handleSelectFixture`'s shape and downstream state-setting (lines 169-188) intact.
- Add `getFixtureLineup(id)` replacement in `footballApi.ts` calling `GET /api/v1/fixtures/{id}/lineup`
- In `MatchSetup.tsx`, update `handleSelectFixture` to call the new async lineup fetch instead of the synchronous local lookup, but keep setting `teamAName`, `teamBName`, `teamAFormation`, `teamBFormation`, `teamAPlayers`, `teamBPlayers` exactly as before, so nothing downstream needs to change
- Add confirmation display if not already present: competition, date, home/away, venue (data now comes from `GET /api/v1/fixtures/{id}`)
- Loading state: "Loading official lineup..."
- Error state: "Football data service unavailable. Retry."

**Outcome:** selecting a fixture now pulls real official data, but `MatchSetupData` — and therefore what gets saved to `localStorage['gaffer-match-setups']` — is unchanged in shape.

---

## Phase 5 — Player Mapping Integration & Cutover
Wire confidence-based suggestions into the **existing** auto-suggestion UI block in
`PlayerMapping.tsx` (lines 724-756), and finish removing the hardcoded data.
- `PlayerMapping.tsx` already reads `localStorage['gaffer-match-setups']` (lines 212-223) — since Phase 4 keeps that shape identical, this file likely needs NO structural changes, only using the confidence/lineup data that's now real instead of mock
- In the auto-suggestion block (lines 724-756), render "Suggested: Vitinha (94%)" using the real lineup player data now flowing through, above the manual assignment dropdown (lines 780-809) — do not touch `handleMapChange` or the manual override logic
- Delete `MOCK_FIXTURES` from `MatchSetup.tsx` and `REAL_LINEUPS` + `getFixtureLineup` from `lib/services/fixtureImportService.ts` (or delete the file entirely if nothing else imports from it — confirm with a repo-wide search first)
- Handle case where lineup fetch failed in Phase 4 → `PlayerMapping.tsx` should fall back to today's manual-only behavior gracefully (no suggestion shown, not a crash)

**Outcome:** the full pipeline (Hopper upload → analysis complete → MatchSetup fixture search/confirm → PlayerMapping with real suggestions) works end to end with real data, and all hardcoded fixture/lineup data is gone.

---

## Phase 6 — Error Handling & Polish
Tie together all loading/error states defined in the spec.
- Consistent loading/error UI components (`LoadingOverlay`, `ErrorCard` if needed)
- Verify all 3 error cases: no matches found, API unavailable, lineup unavailable
- Confirm styling matches existing theme (colors, typography, spacing, cards) — no redesign
- Manual QA pass through the full new flow

**Outcome:** feature feels like a native, polished part of the product, not a bolted-on addition.

---

## Phase 7 — Acceptance Check
Validate against the spec's acceptance criteria before calling it done.
- [ ] `MOCK_FIXTURES` and `REAL_LINEUPS` fully removed, no leftover imports
- [ ] Dynamic fixture search works in `MatchSetup.tsx`
- [ ] Official lineup loads correctly via `/api/v1/fixtures/{id}/lineup`
- [ ] `localStorage['gaffer-match-setups']` shape unchanged — `PlayerMapping.tsx` needed no structural edits
- [ ] `handleMapChange` / manual override logic in `PlayerMapping.tsx` untouched
- [ ] Hopper → analysis pipeline untouched
- [ ] UI matches existing design language (dark room / emerald-400 accents per workspace README)

**Outcome:** feature is demo-ready and matches the spec's definition of done.

---

## Suggested Order Rationale
Backend/types first (Phases 0–2) so frontend has something real to call against — avoids
building UI on mocked data that later needs rework. Frontend built inside-out: search →
confirmation → mapping integration, since each step depends on the previous one's output
shape. Error handling is layered in last (Phase 6) once the happy path is proven, then
checked off against acceptance criteria (Phase 7).
