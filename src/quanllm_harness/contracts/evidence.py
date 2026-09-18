from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .common import IssueOrigin, Severity


@dataclass(frozen=True)
class Evidence:
    id: str
    tool: str
    operation: str
    arguments: Mapping[str, Any]
    ok: bool
    result: Any = None
    error: str = ""
    limitations: Sequence[str] = ()
    claim_ids: Sequence[str] = ()
    input_verified: bool = False
    supports_claim: bool | None = None
    review: str = ""
    plugin_name: str = ""
    plugin_version: str = ""
    plugin_digest: str = ""
    execution_mode: str = "builtin"


@dataclass(frozen=True)
class Issue:
    origin: IssueOrigin
    severity: Severity
    quote: str
    problem: str
    correction: str = ""
    evidence_ids: Sequence[str] = ()

    @property
    def fingerprint(self) -> str:
        return "|".join((self.origin.value, self.quote.strip(), self.problem.strip()))
