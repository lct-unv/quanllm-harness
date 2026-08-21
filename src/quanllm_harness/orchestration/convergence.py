from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from ..contracts import Issue


class ConvergenceAction(str, Enum):
    PASS = "pass"
    REPAIR = "repair"
    STOP = "stop"


@dataclass(frozen=True)
class ConvergenceDecision:
    action: ConvergenceAction
    reason: str = ""


@dataclass
class ConvergencePolicy:
    max_repair_rounds: int
    duplicate_issue_limit: int
    recurrence: Counter[str] = field(default_factory=Counter)

    def evaluate(
        self,
        issues: Sequence[Issue],
        *,
        completed_repair_rounds: int,
    ) -> ConvergenceDecision:
        if not issues:
            return ConvergenceDecision(ConvergenceAction.PASS)
        for issue in issues:
            self.recurrence[issue.fingerprint] += 1
        if completed_repair_rounds >= self.max_repair_rounds:
            return ConvergenceDecision(
                ConvergenceAction.STOP,
                "达到修复轮数上限，仍存在模型问题",
            )
        if any(count > self.duplicate_issue_limit for count in self.recurrence.values()):
            return ConvergenceDecision(
                ConvergenceAction.STOP,
                "同一问题重复出现，修复未收敛",
            )
        return ConvergenceDecision(ConvergenceAction.REPAIR)
