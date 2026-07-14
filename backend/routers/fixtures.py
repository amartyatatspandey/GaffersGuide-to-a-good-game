"""
Fixture search and lineup endpoints for Gaffer's Guide.

Routes
------
GET /api/v1/fixtures/search?q=    — search fixtures by team name
GET /api/v1/fixtures/{id}         — fixture details (teams, competition, date, venue)
GET /api/v1/fixtures/{id}/lineup  — official starting lineup + bench

Provider logic lives in services/football_provider.py.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

# ── Pydantic response models ─────────────────────────────────────────────────
# Field names intentionally mirror frontend_final/types/player.ts (Player) and
# frontend_final/types/lineup.ts (FixtureSearchResult, FixtureLineup, Lineup)
# so the JSON keys are identical on both sides of the API boundary.


class PlayerModel(BaseModel):
    """Mirrors frontend_final/types/player.ts → Player."""

    id: str
    name: str
    number: int
    position: str  # 'GK' | 'DF' | 'MF' | 'FW'
    isStarting: bool


class LineupModel(BaseModel):
    """Mirrors frontend_final/types/lineup.ts → Lineup."""

    name: str
    formation: str
    players: list[PlayerModel]


class FixtureSearchResult(BaseModel):
    """Mirrors frontend_final/types/lineup.ts → FixtureSearchResult."""

    id: str
    name: str
    competition: str
    date: str        # ISO 8601
    venue: Optional[str] = None
    homeTeam: str
    awayTeam: str


class FixtureDetail(BaseModel):
    """Full fixture metadata; superset of FixtureSearchResult."""

    id: str
    name: str
    competition: str
    date: str
    venue: Optional[str] = None
    homeTeam: str
    awayTeam: str


class FixtureLineupResponse(BaseModel):
    """Mirrors frontend_final/types/lineup.ts → FixtureLineup."""

    fixture_id: str
    fixture_name: str
    competition: str
    team_a: LineupModel
    team_b: LineupModel


from services.football_provider import (
    get_fixture as provider_get_fixture,
    get_lineup as provider_get_lineup,
    search_fixtures as provider_search_fixtures,
)


# ── Router ───────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/api/v1/fixtures", tags=["fixtures"])


@router.get(
    "/search",
    response_model=list[FixtureSearchResult],
    summary="Search fixtures by team name",
    description=(
        "Returns up to 12 fixtures for the best-matching team. "
        "Requires at least 2 characters. Uses API-Football via RapidAPI."
    ),
)
async def search_fixtures(
    request: Request,
    q: str = Query(..., min_length=2, description="Team name, e.g. 'PSG' or 'Real Madrid'"),
) -> list[FixtureSearchResult]:
    """Search for fixtures matching a team name."""
    return await provider_search_fixtures(q)


@router.get(
    "/{fixture_id}/lineup",
    response_model=FixtureLineupResponse,
    summary="Get official starting lineup for a fixture",
    description=(
        "Returns starting XI and bench for both teams. "
        "team_a = home, team_b = away. "
        "Returns 404 if the lineup has not yet been released."
    ),
)
async def get_fixture_lineup(
    request: Request,
    fixture_id: str,
) -> FixtureLineupResponse:
    """Retrieve official starting lineup and bench for a fixture."""
    return await provider_get_lineup(fixture_id)


@router.get(
    "/{fixture_id}",
    response_model=FixtureDetail,
    summary="Get fixture details",
    description="Returns teams, competition, date, and venue for a specific fixture ID.",
)
async def get_fixture(
    request: Request,
    fixture_id: str,
) -> FixtureDetail:
    """Retrieve metadata for a specific fixture."""
    return await provider_get_fixture(fixture_id)
