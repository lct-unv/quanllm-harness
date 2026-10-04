from __future__ import annotations

from dataclasses import dataclass, replace

from ...contracts import IssueOrigin, VerificationReport
from ...plugins import (
    DomainCandidateUpdate,
    DomainFinalization,
    DomainRepairInstruction,
    PluginManifest,
    PluginVerificationContext,
    PluginVerificationResult,
)
from .deterministic import (
    apply_deterministic_corrections,
    deterministic_candidate_issues,
    deterministic_fallback_answer,
)
from .operator_backend import operator_tools
from .prompts import contribute as contribute_prompt
from .quantum_backends import quantum_tools
from .standard_results import (
    PAULI_REPAIR_HINT,
    apply_canonical,
    canonical_corrections,
)
from .sympy_tools import qm_teaching_sympy_tools


@dataclass(frozen=True)
class QuantumMechanicsTeachingStrategy:
    """Quantum-mechanics teaching verification and recovery policy.

    Matching is unconditional once the plugin is enabled. Individual checks are
    conservative and return no result for questions they do not recognize. This
    keeps domain selection explicit in host policy instead of guessing from a
    model name inside the generic core.
    """

    def matches(self, question: str) -> bool:
        return bool(question.strip())

    def candidate_issues(self, question: str, candidate: str):
        return tuple(deterministic_candidate_issues(question, candidate))

    def fallback_answer(self, question: str) -> str:
        return deterministic_fallback_answer(question)

    def correct_candidate(self, question: str, candidate: str) -> DomainCandidateUpdate:
        corrected, notes = apply_deterministic_corrections(question, candidate)
        return DomainCandidateUpdate(corrected, tuple(notes))

    def repair_instructions(self, question: str, candidate: str):
        return tuple(
            DomainRepairInstruction(
                key=key,
                problem=f"标准结果核对：{key} 与标准结果不符",
                correction=correction,
            )
            for key, correction in canonical_corrections(question, candidate)
        )

    def repair_hint(self, question: str) -> str:
        return PAULI_REPAIR_HINT if "泡利" in question else ""

    def finalize(self, question: str, candidate: str) -> DomainFinalization:
        corrected, report, _ = apply_canonical(question, candidate, VerificationReport())
        return DomainFinalization(
            candidate=corrected,
            issues=tuple(
                issue for issue in report.issues if issue.origin is IssueOrigin.INFRASTRUCTURE
            ),
            warnings=report.protocol_warnings,
        )


class QuantumMechanicsTeachingPlugin:
    manifest = PluginManifest(
        name="quanllm-qm-teaching",
        version="0.1.0",
        requires_harness=">=0.1.2,<0.2",
        description="Official quantum-mechanics teaching verification and recovery profile",
        permissions=(
            "domains.register",
            "prompts.contribute",
            "tools.register",
            "verifiers.register",
        ),
    )

    def __init__(self) -> None:
        self.strategy = QuantumMechanicsTeachingStrategy()

    def setup(self, context) -> None:
        context.domains.register("teaching", self.strategy)
        context.prompts.register("teaching", contribute_prompt)
        for tool in (*qm_teaching_sympy_tools(), *operator_tools(), *quantum_tools()):
            context.tools.register(replace(tool, name=f"quanllm-qm-teaching.{tool.name}"))

        def verify(verification: PluginVerificationContext) -> PluginVerificationResult:
            if not self.strategy.matches(verification.question):
                return PluginVerificationResult()
            issues = self.strategy.candidate_issues(verification.question, verification.candidate)
            ok_tools_by_claim: dict[str, set[str]] = {}
            for evidence in verification.evidence:
                if not evidence.ok:
                    continue
                for claim_id in evidence.claim_ids:
                    ok_tools_by_claim.setdefault(claim_id, set()).add(evidence.tool)
            warnings: list[str] = []
            required_tools = (
                (
                    ("本征矢", "本征值", "本征态", "eigenvector", "eigenvalue"),
                    "quanllm-qm-teaching.matrix_eigenpair_check",
                ),
                (
                    ("对易子", "commutator", "[a,b]", "[a，b]", "[a,[a,"),
                    "quanllm-qm-teaching.operator_algebra",
                ),
            )
            for claim in verification.claims:
                quote = claim.quote.casefold()
                for markers, tool_name in required_tools:
                    if any(marker in quote for marker in markers) and tool_name not in (
                        ok_tools_by_claim.get(claim.id) or set()
                    ):
                        warnings.append(f"断言 {claim.id} 缺少领域必需工具证据 {tool_name}")
            return PluginVerificationResult(
                issues=issues,
                warnings=tuple(warnings),
                summary=f"确定性领域门禁发现 {len(issues)} 个问题",
            )

        context.verifiers.register("deterministic", verify)


plugin = QuantumMechanicsTeachingPlugin()

__all__ = [
    "QuantumMechanicsTeachingPlugin",
    "QuantumMechanicsTeachingStrategy",
    "plugin",
]
