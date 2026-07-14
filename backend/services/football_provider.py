"""
API-Football integration layer for Gaffer's Guide.

Wraps all third-party HTTP calls, response normalisation, and in-memory caching
so routers and future callers never touch API-Football directly.

Requires FOOTBALL_API_KEY in the environment (RapidAPI key for api-football).
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Optional

import httpx
from fastapi import HTTPException

LOGGER = logging.getLogger(__name__)

# ── Provider config ──────────────────────────────────────────────────────────

_API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
_API_FOOTBALL_TIMEOUT = 10.0
_CACHE_TTL_SECONDS = 300  # 5 minutes

# ── In-memory cache ──────────────────────────────────────────────────────────

_cache: dict[str, tuple[float, Any]] = {}
_cache_lock = threading.Lock()


def _cache_get(key: str) -> Any | None:
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.time() >= expires_at:
            del _cache[key]
            return None
        return value


def _cache_set(key: str, value: Any) -> None:
    with _cache_lock:
        _cache[key] = (time.time() + _CACHE_TTL_SECONDS, value)


@lru_cache(maxsize=1)
def _models() -> tuple[type, ...]:
    """Lazy import to avoid circular dependency with routers.fixtures."""
    from routers.fixtures import (
        FixtureDetail,
        FixtureLineupResponse,
        FixtureSearchResult,
        LineupModel,
        PlayerModel,
    )
    return FixtureDetail, FixtureLineupResponse, FixtureSearchResult, LineupModel, PlayerModel


# ── Normalisation helpers ────────────────────────────────────────────────────

_POS_MAP: dict[str, str] = {"G": "GK", "D": "DF", "M": "MF", "F": "FW"}


def _normalise_position(api_pos: str) -> str:
    """Map API-Football single-letter positions to our four-char codes."""
    return _POS_MAP.get((api_pos or "").upper(), "MF")


def _fixture_to_search_result(raw: dict[str, Any]) -> Any:
    """Convert a single raw API-Football fixture object into FixtureSearchResult."""
    _, _, FixtureSearchResult, _, _ = _models()
    f = raw.get("fixture", {})
    league = raw.get("league", {})
    teams = raw.get("teams", {})
    home_name: str = teams.get("home", {}).get("name", "")
    away_name: str = teams.get("away", {}).get("name", "")
    competition: str = league.get("name", "Unknown Competition")
    league_round: str = league.get("round", "")
    if league_round:
        competition = f"{competition} — {league_round}"
    venue_obj = f.get("venue") or {}
    venue: Optional[str] = venue_obj.get("name") if isinstance(venue_obj, dict) else None
    return FixtureSearchResult(
        id=str(f.get("id", "")),
        name=f"{home_name} vs {away_name}",
        competition=competition,
        date=f.get("date", ""),
        venue=venue,
        homeTeam=home_name,
        awayTeam=away_name,
    )


def _normalise_lineup_team(raw_team: dict[str, Any]) -> Any:
    """Convert a single raw API-Football lineup object into LineupModel."""
    _, _, _, LineupModel, PlayerModel = _models()
    team: dict[str, Any] = raw_team.get("team", {})
    formation: str = raw_team.get("formation") or ""
    players: list[Any] = []

    for entry in raw_team.get("startXI", []):
        p = entry.get("player", {})
        players.append(PlayerModel(
            id=str(p.get("id", "")),
            name=p.get("name", ""),
            number=int(p.get("number") or 0),
            position=_normalise_position(p.get("pos", "")),
            isStarting=True,
        ))

    for entry in raw_team.get("substitutes", []):
        p = entry.get("player", {})
        players.append(PlayerModel(
            id=str(p.get("id", "")),
            name=p.get("name", ""),
            number=int(p.get("number") or 0),
            position=_normalise_position(p.get("pos", "")),
            isStarting=False,
        ))

    return LineupModel(
        name=team.get("name", ""),
        formation=formation,
        players=players,
    )


# ── API-Football HTTP layer ──────────────────────────────────────────────────


def _provider_headers(api_key: str) -> dict[str, str]:
    return {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "v3.football.api-sports.io",
    }


def _get_api_key() -> str:
    """Return FOOTBALL_API_KEY or raise 503 with a clear message."""
    key = (os.getenv("FOOTBALL_API_KEY") or "").strip()
    if not key:
        raise HTTPException(
            status_code=503,
            detail=(
                "Football data service is not configured. "
                "Set the FOOTBALL_API_KEY environment variable."
            ),
        )
    return key


async def _apifootball_get(
    path: str,
    params: dict[str, Any],
    api_key: str,
) -> dict[str, Any]:
    """
    Single async GET against API-Football v3.

    Translates ALL provider errors — both HTTP-level and application-level errors
    embedded in a 200 response body — into clean HTTPExceptions so nothing leaks
    to the client and no error result can be cached as "zero results".

    API-Football returns HTTP 200 even for quota/auth/plan errors; the actual
    error sits in the ``"errors"`` key of the JSON body:
      - errors.requests  → daily quota exhausted
      - errors.rateLimit → per-minute rate limit hit
      - errors.token     → invalid or missing API key
      - errors.plan      → endpoint not available on current plan
    ``errors`` is an empty array ``[]`` when the call succeeded (not a dict).
    """
    url = f"{_API_FOOTBALL_BASE}{path}"
    try:
        async with httpx.AsyncClient(timeout=_API_FOOTBALL_TIMEOUT) as client:
            resp = await client.get(url, params=params, headers=_provider_headers(api_key))

        # Set FOOTBALL_API_DEBUG=1 to log raw request/response (never enable in production)
        if os.getenv("FOOTBALL_API_DEBUG") == "1":
            LOGGER.debug(
                "API-Football → %s params=%s status=%s",
                resp.request.url,
                params,
                resp.status_code,
            )
            LOGGER.debug("API-Football raw response (%s): %s", path, resp.text)

    except httpx.TimeoutException:
        LOGGER.warning("API-Football timeout: %s params=%s", path, params)
        raise HTTPException(
            status_code=504,
            detail="Football data provider timed out. Please retry in a moment.",
        )
    except httpx.RequestError as exc:
        LOGGER.error("API-Football request error: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Football data provider is unreachable. Check your network or retry later.",
        )

    # ── HTTP-level errors ────────────────────────────────────────────────────
    if resp.status_code in (401, 403):
        LOGGER.error("API-Football auth failure: HTTP %s", resp.status_code)
        raise HTTPException(
            status_code=503,
            detail="Invalid or expired football API key. Verify FOOTBALL_API_KEY.",
        )

    if resp.status_code == 429:
        LOGGER.warning("API-Football rate limit hit (HTTP 429)")
        raise HTTPException(
            status_code=429,
            detail="Football data provider rate limit reached. Please wait before retrying.",
        )

    if resp.status_code >= 400:
        LOGGER.error("API-Football HTTP %s: %s params=%s", resp.status_code, path, params)
        raise HTTPException(
            status_code=502,
            detail=f"Football data provider returned an unexpected error (HTTP {resp.status_code}).",
        )

    try:
        body: dict[str, Any] = resp.json()
    except Exception:
        LOGGER.error("API-Football non-JSON response for %s", path)
        raise HTTPException(
            status_code=502,
            detail="Football data provider returned a malformed response.",
        )

    # ── Body-level errors (API-Football returns HTTP 200 for these) ──────────
    # errors is [] when fine, or a dict with error keys when something went wrong.
    #
    # IMPORTANT: "plan" errors intentionally pass through instead of raising here.
    # The season-fallback loop in _apifootball_search_raw must inspect the body and
    # retry with an older season; raising 402 here short-circuits that loop and
    # surfaces a confusing error for season=2026 even when 2024/2023/2022 work fine.
    # The code-lookup (/teams?code=) also returns plan errors on the free tier — that
    # block gracefully ignores an empty response instead of trying to catch an exception.
    # Final 402 escalation (all seasons exhausted) is handled in _apifootball_search_raw.
    errors = body.get("errors")
    if isinstance(errors, dict) and errors:
        err_keys = set(errors.keys())
        actionable = err_keys - {"plan"}
        if actionable:
            LOGGER.warning("API-Football body errors %s: %s  path=%s", actionable, errors, path)
        else:
            LOGGER.debug("API-Football plan restriction (pass-through): %s  path=%s", errors, path)

        if "requests" in err_keys:
            raise HTTPException(
                status_code=429,
                detail=(
                    "Daily API quota exhausted. "
                    "Football data will be available again after midnight UTC."
                ),
            )

        if "rateLimit" in err_keys:
            raise HTTPException(
                status_code=429,
                detail="Football data provider rate limit reached. Please wait before retrying.",
            )

        if "token" in err_keys:
            raise HTTPException(
                status_code=503,
                detail="Invalid or missing football API key. Verify FOOTBALL_API_KEY.",
            )

        # "plan" falls through to caller — do not raise here (see note above)

        # Any other unrecognised key that isn't plan — surface it
        unknown = actionable - {"requests", "rateLimit", "token"}
        if unknown:
            first_msg = next(
                (v for k, v in errors.items() if k not in {"plan"}),
                "Unknown provider error",
            )
            raise HTTPException(status_code=502, detail=str(first_msg))

    return body


# Reserve/youth marker tokens — names containing these as word-tokens are deprioritised
_RESERVE_MARKERS: frozenset[str] = frozenset(
    {"U17", "U18", "U19", "U20", "U21", "U22", "U23", "II", "B"}
)


def _pick_best_team(candidates: list[dict[str, Any]], query: str) -> dict[str, Any]:
    """
    Return the best-matching first-team entry from a /teams search response.

    Scoring (additive, higher is better):
      +100  exact case-insensitive full-name match
      + 80  query exactly matches the team's official abbreviation code
            (e.g. query "PSG" → code "PSG" on Paris Saint-Germain; prevents
            "PSG U19" from winning over Paris SG just because "PSG" is its
            first word-token)
      + 60  query is the first word-token in the name  (e.g. "PSG" in "PSG U19")
      + 50  query appears as any complete word-token in the name
      + 20  query is a substring of the name (but not a full token)
      + 10  name is a substring of the query
      - 30  per reserve/youth marker token found in the name (U17-U23, II, B)

    Falls back to candidates[0] when no candidate scores above 0.
    """
    q_lower = query.strip().lower()

    def _score(entry: dict[str, Any]) -> float:
        team_obj = entry.get("team", {})
        name: str = (team_obj.get("name") or "").strip()
        name_lower = name.lower()
        tokens = name_lower.split()
        s = 0.0

        if name_lower == q_lower:
            s += 100

        # Official club abbreviation/code (e.g. "PSG" → Paris Saint-Germain)
        code: str = (team_obj.get("code") or "").strip().lower()
        if code and code == q_lower:
            s += 80

        if name_lower != q_lower:
            if tokens and tokens[0] == q_lower:
                s += 60
            elif q_lower in tokens:
                s += 50
            elif q_lower in name_lower:
                s += 20
            elif name_lower and name_lower in q_lower:
                s += 10

        for tok in tokens:
            if tok.upper() in _RESERVE_MARKERS:
                s -= 30

        return s

    scored = [(c, _score(c)) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    best, best_score = scored[0]
    LOGGER.debug(
        "_pick_best_team: query=%r → name=%r id=%s score=%.1f",
        query,
        best.get("team", {}).get("name"),
        best.get("team", {}).get("id"),
        best_score,
    )
    return best


def _slice_fixtures(
    fixtures: list[dict[str, Any]],
    past_count: int = 6,
    future_count: int = 6,
) -> list[dict[str, Any]]:
    """
    Return up to past_count recent + future_count upcoming fixtures, sorted chronologically.
    Falls back to the last (past_count + future_count) fixtures if all are in the past.
    The API returns fixtures sorted ascending by date; we preserve that order.
    """
    now_str = datetime.now(timezone.utc).isoformat()
    past = [fx for fx in fixtures if (fx.get("fixture") or {}).get("date", "") < now_str]
    future = [fx for fx in fixtures if (fx.get("fixture") or {}).get("date", "") >= now_str]
    selected = past[-past_count:] + future[:future_count]
    if not selected:
        selected = fixtures[-(past_count + future_count):]
    return selected


async def _apifootball_search_raw(q: str, api_key: str) -> list[dict[str, Any]]:
    """
    Two-step fixture search:
      1. GET /teams?search={q}  → pick best-matching team via _pick_best_team
      2. GET /fixtures?team={id}&season={s}  → iterate season candidates until results found

    Note: the free plan blocks the `last` and `next` convenience params, so we fetch all
    fixtures for a season and slice client-side via _slice_fixtures.  A paid plan could
    restore last/next for a smaller payload — remove the note when upgrading.
    """
    teams_data = await _apifootball_get("/teams", {"search": q}, api_key)
    teams: list[dict[str, Any]] = teams_data.get("response", [])

    # When the query looks like a club abbreviation (short, all-caps, e.g. "PSG", "MCI"),
    # also fetch by code= so the parent club surfaces alongside the youth/reserve teams
    # that the name-search typically returns first (API-Football search is name-literal).
    # The code= parameter is not available on the free API-Football plan (returns plan
    # error), so we catch that gracefully and fall back to name-search results only.
    # On a paid plan this enhancement activates automatically with no code change needed.
    if q == q.upper() and len(q) <= 5:
        # _apifootball_get passes plan errors through (response: []) so this block
        # silently adds nothing when code= is plan-blocked — no try/except needed.
        code_data = await _apifootball_get("/teams", {"code": q}, api_key)
        code_teams: list[dict[str, Any]] = code_data.get("response", [])
        # Merge, deduplicating by team id so _pick_best_team sees a clean list
        existing_ids = {str(t["team"]["id"]) for t in teams}
        for ct in code_teams:
            if str(ct["team"]["id"]) not in existing_ids:
                teams.append(ct)

    if not teams:
        return []

    team = _pick_best_team(teams, q)
    team_id: str = str(team["team"]["id"])

    # Free-tier workaround: API-Football free plans only provide data for 2022–2024.
    # We try the current year first so a paid-plan upgrade works automatic,
    # then fall back through the free-plan window. Remove fallback when plan is upgraded.
    current_year = datetime.now(timezone.utc).year
    season_candidates = (current_year, 2024, 2023, 2022)

    plan_blocked_count = 0

    for season in season_candidates:
        fixtures_data = await _apifootball_get(
            "/fixtures",
            {"team": team_id, "season": season},
            api_key,
        )
        # Note: body-level plan errors are now caught and raised in _apifootball_get
        # before we reach this point. The is_plan_blocked check here is a belt-and-
        # suspenders fallback in case the error slips through (e.g. a new error key).
        errors = fixtures_data.get("errors") or {}
        is_plan_blocked = isinstance(errors, dict) and "plan" in errors

        fixtures: list[dict[str, Any]] = fixtures_data.get("response", [])
        if fixtures:
            sliced = _slice_fixtures(fixtures)
            LOGGER.debug(
                "_apifootball_search_raw: team_id=%s season=%s total=%d returning=%d",
                team_id, season, len(fixtures), len(sliced),
            )
            return sliced

        if is_plan_blocked:
            plan_blocked_count += 1
            LOGGER.debug(
                "_apifootball_search_raw: plan blocked for team_id=%s season=%s",
                team_id, season,
            )

    if plan_blocked_count == len(season_candidates):
        raise HTTPException(
            status_code=402,
            detail=(
                "No fixtures available for this team on the current API plan. "
                "The free tier only provides data for seasons 2022–2024."
            ),
        )

    return []


# ── Public API ───────────────────────────────────────────────────────────────


async def search_fixtures(query: str) -> list[Any]:
    """Search fixtures by team name. Results are cached for 5 minutes."""
    cache_key = f"search:{query.strip().lower()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    api_key = _get_api_key()
    raw = await _apifootball_search_raw(query, api_key)
    results = [_fixture_to_search_result(fx) for fx in raw]
    _cache_set(cache_key, results)
    return results


async def get_fixture(fixture_id: str) -> Any:
    """Return fixture metadata. Cached for 5 minutes."""
    FixtureDetail, _, _, _, _ = _models()
    cache_key = f"fixture:{fixture_id}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    api_key = _get_api_key()
    data = await _apifootball_get("/fixtures", {"id": fixture_id}, api_key)
    fixtures: list[dict[str, Any]] = data.get("response", [])
    if not fixtures:
        raise HTTPException(status_code=404, detail=f"Fixture '{fixture_id}' not found.")

    result = _fixture_to_search_result(fixtures[0])
    detail = FixtureDetail(
        id=result.id,
        name=result.name,
        competition=result.competition,
        date=result.date,
        venue=result.venue,
        homeTeam=result.homeTeam,
        awayTeam=result.awayTeam,
    )
    _cache_set(cache_key, detail)
    return detail


async def get_lineup(fixture_id: str) -> Any:
    """Return official starting lineup and bench for both teams."""
    _, FixtureLineupResponse, _, _, _ = _models()
    api_key = _get_api_key()

    fixture_data, lineup_data = await asyncio.gather(
        _apifootball_get("/fixtures", {"id": fixture_id}, api_key),
        _apifootball_get("/fixtures/lineups", {"fixture": fixture_id}, api_key),
    )

    lineup_list: list[dict[str, Any]] = lineup_data.get("response", [])
    if not lineup_list:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Lineup for fixture '{fixture_id}' is not yet available. "
                "Lineups are typically released 1 hour before kickoff."
            ),
        )

    fixture_list: list[dict[str, Any]] = fixture_data.get("response", [])
    home_team_id: str = ""
    competition: str = "Unknown Competition"
    fixture_name: str = "Unknown Fixture"

    if fixture_list:
        raw_fx = fixture_list[0]
        teams = raw_fx.get("teams", {})
        home_team_id = str(teams.get("home", {}).get("id", ""))
        home_name: str = teams.get("home", {}).get("name", "")
        away_name: str = teams.get("away", {}).get("name", "")
        fixture_name = f"{home_name} vs {away_name}"
        league = raw_fx.get("league", {})
        competition = league.get("name", "Unknown Competition")
        league_round: str = league.get("round", "")
        if league_round:
            competition = f"{competition} — {league_round}"

    team_a_raw: Optional[dict[str, Any]] = None
    team_b_raw: Optional[dict[str, Any]] = None

    for lu in lineup_list:
        tid = str(lu.get("team", {}).get("id", ""))
        if tid == home_team_id:
            team_a_raw = lu
        else:
            team_b_raw = lu

    if team_a_raw is None and len(lineup_list) >= 1:
        team_a_raw = lineup_list[0]
    if team_b_raw is None and len(lineup_list) >= 2:
        team_b_raw = lineup_list[1]

    if team_a_raw is None or team_b_raw is None:
        raise HTTPException(
            status_code=404,
            detail=f"Could not resolve both team lineups for fixture '{fixture_id}'.",
        )

    team_a_norm = _normalise_lineup_team(team_a_raw)
    team_b_norm = _normalise_lineup_team(team_b_raw)

    LOGGER.debug(
        "get_lineup %s: team_a=%s starters=%d  team_b=%s starters=%d",
        fixture_id,
        team_a_norm.name,
        sum(1 for p in team_a_norm.players if p.isStarting),
        team_b_norm.name,
        sum(1 for p in team_b_norm.players if p.isStarting),
    )

    return FixtureLineupResponse(
        fixture_id=fixture_id,
        fixture_name=fixture_name,
        competition=competition,
        team_a=team_a_norm,
        team_b=team_b_norm,
    )
