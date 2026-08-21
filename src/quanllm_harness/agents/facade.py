from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..contracts import Candidate, RequestPolicy
from .independent_solver import IndependentSolverAgent
from .repair import RepairAgent
from .router import RouterAgent
from .runtime import AgentRuntime
from .solver import SolverAgent
from .synthesizer import SynthesizerAgent


@dataclass
class HarnessAgents:
    """Compatibility facade; orchestration uses the individual Agent classes."""

    runtime: AgentRuntime

    def route(self, question: str) -> RequestPolicy:
        return RouterAgent(self.runtime).route(question)

    def solve(
        self,
        question: str,
        *,
        independent: bool = False,
        require_tool: bool = False,
    ) -> Candidate:
        cls = IndependentSolverAgent if independent else SolverAgent
        return cls(self.runtime).solve(question, require_tool=require_tool)

    def synthesize(self, question: str, primary: Candidate, independent: Candidate) -> Candidate:
        return SynthesizerAgent(self.runtime).synthesize(question, primary, independent)

    def repair(
        self,
        question: str,
        candidate: str,
        issues: Sequence[Mapping[str, Any]],
    ) -> Candidate:
        return RepairAgent(self.runtime).repair(question, candidate, issues)
