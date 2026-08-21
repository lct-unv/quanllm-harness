from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Claim:
    id: str
    quote: str
    kind: str
    importance: str = "major"


@dataclass(frozen=True)
class Requirement:
    id: str
    quote: str
