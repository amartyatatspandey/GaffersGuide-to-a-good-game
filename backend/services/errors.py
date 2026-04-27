from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EngineRoutingError(Exception):
    """Structured service error to map into API error responses."""

    status_code: int
    code: str
    message: str
    hint: str | None = None

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

    def to_detail(self) -> dict[str, str]:
        payload: dict[str, str] = {"code": self.code, "message": self.message}
        if self.hint:
            payload["hint"] = self.hint
        return payload
