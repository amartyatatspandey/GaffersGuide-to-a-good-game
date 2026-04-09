from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ExperimentError(Exception):
    status_code: int
    code: str
    message: str
    hint: str | None = None

    def to_detail(self) -> dict[str, str]:
        payload = {"code": self.code, "message": self.message}
        if self.hint:
            payload["hint"] = self.hint
        return payload
