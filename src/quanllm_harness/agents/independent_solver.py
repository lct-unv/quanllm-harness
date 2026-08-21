from __future__ import annotations

from dataclasses import dataclass

from ..contracts import Candidate
from . import prompts
from .runtime import AgentRuntime


@dataclass
class IndependentSolverAgent:
    runtime: AgentRuntime

    def solve(self, question: str, *, require_tool: bool = False) -> Candidate:
        return self.runtime.reason(
            prompts.INDEPENDENT_SOLVER_PROMPT,
            "【原始用户问题】\n" + question,
            stage="独立求解",
            require_tool=require_tool,
        )
