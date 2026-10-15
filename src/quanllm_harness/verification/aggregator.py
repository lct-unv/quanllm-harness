from __future__ import annotations

from dataclasses import dataclass

from ..agents.runtime import AgentRuntime
from ..contracts import (
    HarnessEvent,
    Issue,
    IssueOrigin,
    Severity,
    VerificationReport,
    answer_digest,
)
from ..plugins import PluginVerificationContext, PluginVerificationResult
from ..protocols.claim_extraction import ClaimExtractionProtocol
from .mathematical import MathematicalVerifier
from .semantic import SemanticVerifier
from .structural import StructuralSnapshot
from .timeline import build_assertion_timeline, timeline_issues


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
        # Evidence is scoped to the candidate it was collected for. Claim IDs are
        # re-issued per candidate, so unscoped matching would let an earlier
        # round's contradiction or stale support leak into the current answer.
        candidate_digest = answer_digest(candidate)
        current_evidence = [
            item
            for item in all_evidence
            if not item.candidate_digest or item.candidate_digest == candidate_digest
        ]
        # Structural evidence gate: math claims (equations/derivations) are only
        # "objectively verified" when at least one of their evidence calls
        # succeeded. Claims declared not_checkable by the planner are exempt, and
        # non-math claims (definition/interpretation/conclusion/condition) are
        # left to the semantic verifier so conceptual questions are not degraded
        # merely for lacking a tool call.
        supported_claims = {
            claim_id for ev in current_evidence if ev.ok for claim_id in ev.claim_ids
        }
        # The core only enforces generic evidence coverage. Requiring a specific
        # tool for a domain concept belongs to an enabled domain verifier.
        uncovered: list[str] = []
        for claim in claims:
            needs_tool = claim.kind in {"equation", "derivation"}
            if not needs_tool or claim.id in not_checkable:
                continue
            if claim.id not in supported_claims:
                uncovered.append(claim.id)
        uncovered = sorted(set(uncovered))
        evidence_gate_issue: Issue | None = None
        if uncovered:
            warnings.append(
                self._warning(
                    "工具证据",
                    RuntimeError("断言缺少成功工具证据：" + ", ".join(uncovered)),
                )
            )
            # Issue 004: a failed or missing tool call is never evidence that a
            # claim is true. The candidate text and its verified state are left
            # untouched -- a tool failure must not rewrite reasoning -- but the
            # run can no longer claim objective verification of that step.
            evidence_gate_issue = Issue(
                origin=IssueOrigin.INFRASTRUCTURE,
                severity=Severity.MAJOR,
                quote="",
                problem="数学断言缺少成功的工具证据：" + ", ".join(uncovered),
                correction=(
                    "补充可执行的工具调用并重新核验；确实无法客观核验时，"
                    "由工具核验计划标注 not_checkable。"
                ),
                evidence_ids=(),
            )
        snapshot = StructuralSnapshot(claims, requirements, current_evidence)
        snapshot.validate_references()
        # Issue 002: a deterministic assertion timeline localizes the earliest
        # contradicted step before any model-written finding is considered.
        timeline = build_assertion_timeline(
            claims,
            current_evidence,
            sorted(not_checkable),
            candidate_digest=candidate_digest,
        )
        localization = timeline_issues(timeline)
        issues, summaries, semantic_warnings, successful_semantic_passes = SemanticVerifier(
            self.runtime
        ).verify(
            question,
            candidate,
            claims,
            requirements,
            current_evidence,
            reference,
        )
        warnings.extend(semantic_warnings)
        # Python-produced localization findings are authoritative and are never
        # adjudicated away; they are listed first so repair targets the earliest
        # incorrect step instead of compensating further downstream.
        issue_list = list(localization)
        if self.runtime.plugin_manager:
            plugin_context = PluginVerificationContext(
                question=question,
                candidate=candidate,
                claims=tuple(claims),
                requirements=tuple(requirements),
                evidence=tuple(all_evidence),
            )
            invariant_run = self.runtime.plugin_manager.run_invariants(plugin_context)
            for finding in invariant_run.findings:
                if finding.checked:
                    summaries.append(
                        f"[{finding.plugin}] 不变量 {finding.invariant} 检查 {finding.checked} 项"
                    )
            issue_list.extend(invariant_run.issues)
            for failure in invariant_run.failures:
                warnings.append(self._warning("插件不变量", RuntimeError(failure)))
                issue_list.append(
                    Issue(
                        origin=IssueOrigin.INFRASTRUCTURE,
                        severity=Severity.MAJOR,
                        quote="",
                        problem=failure,
                        correction="修复或禁用失败的领域不变量后重新运行。",
                        evidence_ids=(),
                    )
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
                    message = self._warning(f"插件核验器 {verifier_name}", exc)
                    warnings.append(message)
                    issue_list.append(
                        Issue(
                            origin=IssueOrigin.INFRASTRUCTURE,
                            severity=Severity.MAJOR,
                            quote="",
                            problem=message,
                            correction="修复或禁用失败的插件核验器后重新运行。",
                            evidence_ids=(),
                        )
                    )
        issue_list.extend(issues)
        if evidence_gate_issue is not None:
            issue_list.append(evidence_gate_issue)
        final_issues = tuple(issue_list)
        if (
            warnings
            and successful_semantic_passes == 0
            and not any(issue.origin is IssueOrigin.MODEL for issue in final_issues)
        ):
            # Optional checkpoints may fail without downgrading a candidate that
            # another verifier actually checked. Only total semantic-verification
            # loss is delivery-critical.
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
            assertion_timeline=timeline,
        )
