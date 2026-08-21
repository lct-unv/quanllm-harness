from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any, cast

from .claims import Claim, Requirement
from .common import IssueOrigin, RunStatus, Usage
from .events import HarnessEvent
from .evidence import Evidence, Issue
from .requests import RequestPolicy


@dataclass(frozen=True)
class VerificationReport:
    claims: Sequence[Claim] = ()
    requirements: Sequence[Requirement] = ()
    evidence: Sequence[Evidence] = ()
    issues: Sequence[Issue] = ()
    protocol_warnings: Sequence[str] = ()
    verifier_summaries: Sequence[str] = ()

    @property
    def model_issues(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.origin is IssueOrigin.MODEL]

    @property
    def input_issues(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.origin is IssueOrigin.INPUT]


@dataclass(frozen=True)
class HarnessResult:
    status: RunStatus
    answer: str
    policy: RequestPolicy
    verification: VerificationReport
    events: Sequence[HarnessEvent]
    usage: Usage
    repair_rounds: int = 0
    run_id: str = ""
    record_path: str = ""

    def to_dict(self, *, include_events: bool = True) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value
            if is_dataclass(value):
                return {key: convert(item) for key, item in asdict(cast(Any, value)).items()}
            if isinstance(value, Mapping):
                return {str(key): convert(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(item) for item in value]
            return value

        payload = convert(self)
        if not include_events:
            payload.pop("events", None)
        return payload
