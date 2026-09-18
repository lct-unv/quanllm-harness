from __future__ import annotations

from dataclasses import dataclass

from ..agents.runtime import AgentRuntime
from ..contracts import (
    HarnessEvent,
    Issue,
    IssueOrigin,
    Severity,
    VerificationReport,
)
from ..plugins import PluginVerificationContext, PluginVerificationResult
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
        not_checkable: set[str] = set()
        if claims:
            try:
                _, tool_warnings, not_checkable = MathematicalVerifier(self.runtime).collect(
                    question, candidate, claims
                )
                warnings.extend(tool_warnings)
            except Exception as exc:
                warnings.append(self._warning("工具核验计划", exc))
        all_evidence = list(self.runtime.evidence)
        # Structural evidence gate: math claims (equations/derivations) are only
        # "objectively verified" when at least one of their evidence calls
        # succeeded. Claims declared not_checkable by the planner are exempt, and
        # non-math claims (definition/interpretation/conclusion/condition) are
        # left to the semantic verifier so conceptual questions are not degraded
        # merely for lacking a tool call.
        supported_claims = {claim_id for ev in all_evidence if ev.ok for claim_id in ev.claim_ids}
        ok_tools_by_claim: dict[str, set[str]] = {}
        for ev in all_evidence:
            if not ev.ok:
                continue
            for claim_id in ev.claim_ids:
                ok_tools_by_claim.setdefault(claim_id, set()).add(ev.tool)
        # Claims that are math (equation/derivation) OR that assert eigenvectors /
        # eigenvalues / commutators must have a successful evidence from the
        # *appropriate* tool; a weak pass (e.g. compare_expressions on a vector)
        # does not count. This keeps wrong eigenvectors/commutators from passing.
        eigen_markers = ("本征矢", "本征值", "本征态", "eigenvector", "eigenvalue")
        comm_markers = ("对易子", "commutator", "[a,b]", "[a，b]", "[a,[a,")
        uncovered: list[str] = []
        for claim in claims:
            quote = (claim.quote or "").casefold()
            needs_tool = claim.kind in {"equation", "derivation"} or any(
                marker in quote for marker in eigen_markers
            )
            if not needs_tool or claim.id in not_checkable:
                continue
            if claim.id not in supported_claims:
                uncovered.append(claim.id)
                continue
            tools = ok_tools_by_claim.get(claim.id, set())
            if any(marker in quote for marker in eigen_markers) and (
                "matrix_eigenpair_check" not in tools
            ):
                uncovered.append(claim.id)
            elif any(marker in quote for marker in comm_markers) and (
                "operator_algebra" not in tools
            ):
                uncovered.append(claim.id)
        uncovered = sorted(set(uncovered))
        if uncovered:
            warnings.append(
                self._warning(
                    "工具证据",
                    RuntimeError("断言缺少成功工具证据：" + ", ".join(uncovered)),
                )
            )
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
        issue_list = list(issues)
        if self.runtime.plugin_manager:
            plugin_context = PluginVerificationContext(
                question=question,
                candidate=candidate,
                claims=tuple(claims),
                requirements=tuple(requirements),
                evidence=tuple(all_evidence),
            )
            for verifier_name, plugin_name, verifier in self.runtime.plugin_manager.verifiers:
                try:
                    plugin_result = verifier(plugin_context)
                    if not isinstance(plugin_result, PluginVerificationResult):
                        raise TypeError("插件核验器必须返回 PluginVerificationResult")
                    for issue in plugin_result.issues:
                        if not isinstance(issue, Issue):
                            raise TypeError("插件核验器 issues 只能包含 Issue")
                        issue_list.append(issue)
                    warnings.extend(
                        f"插件核验器 {verifier_name}：{warning}"
                        for warning in plugin_result.warnings
                    )
                    if plugin_result.summary:
                        summaries.append(f"[{plugin_name}] {plugin_result.summary}")
                except Exception as exc:
                    warnings.append(self._warning(f"插件核验器 {verifier_name}", exc))
        final_issues = tuple(issue_list)
        if warnings and not any(issue.origin is IssueOrigin.MODEL for issue in final_issues):
            # Surface an explicit, machine-readable signal that the verification
            # pipeline degraded and the conclusion was NOT mathematically verified,
            # instead of silently delivering it as if the math check had run.
            final_issues = (
                *final_issues,
                Issue(
                    origin=IssueOrigin.INFRASTRUCTURE,
                    severity=Severity.MAJOR,
                    quote="",
                    problem="数学核验未完成：结构化协议降级，结论未经完整数学核验（详见 protocol_warnings）",
                    correction="",
                    evidence_ids=(),
                ),
            )
        return VerificationReport(
            claims=tuple(claims),
            requirements=tuple(requirements),
            evidence=tuple(all_evidence),
            issues=final_issues,
            protocol_warnings=tuple(warnings),
            verifier_summaries=tuple(summaries),
        )
