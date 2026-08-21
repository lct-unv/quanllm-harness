from __future__ import annotations

from dataclasses import dataclass

from ..agents.runtime import AgentRuntime
from ..contracts import HarnessEvent, VerificationReport
from ..protocols.claim_extraction import ClaimExtractionProtocol
from .mathematical import MathematicalVerifier
from .semantic import SemanticVerifier
from .structural import StructuralSnapshot


@dataclass
class VerificationEngine:
    """Aggregate independent structural, mathematical and semantic verification."""

    runtime: AgentRuntime

    def _warning(self, stage: str, exc: Exception) -> str:
        message = f"{stage}未完成：{type(exc).__name__}: {exc}"
        if self.runtime.event_sink:
            self.runtime.event_sink(HarnessEvent("degraded", stage, {"reason": message}))
        return message

    def verify(self, question: str, candidate: str, *, reference: str = "") -> VerificationReport:
        warnings: list[str] = []
        try:
            claims, requirements = ClaimExtractionProtocol(self.runtime).extract(
                question, candidate
            )
        except Exception as exc:
            warnings.append(self._warning("断言提取", exc))
            claims, requirements = [], []
        if claims:
            try:
                _, tool_warnings = MathematicalVerifier(self.runtime).collect(
                    question, candidate, claims
                )
                warnings.extend(tool_warnings)
            except Exception as exc:
                warnings.append(self._warning("工具核验计划", exc))
        all_evidence = list(self.runtime.evidence)
        snapshot = StructuralSnapshot(claims, requirements, all_evidence)
        snapshot.validate_references()
        issues, summaries, semantic_warnings = SemanticVerifier(self.runtime).verify(
            question,
            candidate,
            claims,
            requirements,
            all_evidence,
            reference,
        )
        warnings.extend(semantic_warnings)
        return VerificationReport(
            claims=tuple(claims),
            requirements=tuple(requirements),
            evidence=tuple(all_evidence),
            issues=tuple(issues),
            protocol_warnings=tuple(warnings),
            verifier_summaries=tuple(summaries),
        )
