"""Result of one environment check (``doctor``, ``serve`` start-up, backend health)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Level(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class Check:
    name: str
    level: Level
    detail: str

    @property
    def failed(self) -> bool:
        return self.level is Level.FAIL
