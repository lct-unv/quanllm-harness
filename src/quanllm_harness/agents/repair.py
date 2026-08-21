from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..contracts import Candidate
from . import prompts
from .runtime import AgentRuntime


@dataclass
class RepairAgent:
    runtime: AgentRuntime

    def repair(
        self,
        question: str,
        candidate: str,
        issues: Sequence[Mapping[str, Any]],
    ) -> Candidate:
        user = (
            "【原始用户问题】\n"
            + question
            + "\n\n【待修订答案】\n"
            + candidate
            + "\n\n【已确认问题】\n"
            + json.dumps(list(issues), ensure_ascii=False)
            + "\n\n【工具证据】\n"
            + json.dumps(
                [item.__dict__ for item in self.runtime.evidence],
                ensure_ascii=False,
                default=list,
            )
        )
        return self.runtime.reason(
            prompts.REPAIR_PROMPT, user, stage="定向修复", require_tool=False
        )
