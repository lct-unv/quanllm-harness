from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from ..agents import prompts
from ..agents.runtime import AgentRuntime
from ..contracts import Claim, Evidence, HarnessEvent, Issue, Requirement, Severity
from ..protocols import StructuredResponseError
from ..protocols.verdict import parse_verifier_response


@dataclass
class SemanticVerifier:
    runtime: AgentRuntime

    def _warning(self, stage: str, exc: Exception) -> str:
        message = f"{stage}未完成：{type(exc).__name__}: {exc}"
        if self.runtime.event_sink:
            self.runtime.event_sink(HarnessEvent("degraded", stage, {"reason": message}))
        return message

    def verify(
        self,
        question: str,
        candidate: str,
        claims: Sequence[Claim],
        requirements: Sequence[Requirement],
        evidence: Sequence[Evidence],
        reference: str,
    ) -> tuple[list[Issue], list[str], list[str]]:
        common = (
            "【原始问题】\n"
            + question
            + "\n\n【独立候选】\n"
            + (reference or "未启用")
            + "\n\n【待核验候选】\n"
            + candidate
            + "\n\n【关键断言】\n"
            + json.dumps([claim.__dict__ for claim in claims], ensure_ascii=False)
            + "\n\n【用户要求】\n"
            + json.dumps([item.__dict__ for item in requirements], ensure_ascii=False)
            + "\n\n【工具证据】\n"
            + json.dumps([item.__dict__ for item in evidence], ensure_ascii=False, default=list)
        )

        def validator(data):
            return parse_verifier_response(
                data, question=question, candidate=candidate, evidence=evidence
            )

        passes: list[tuple[list[Issue], str]] = []
        warnings: list[str] = []
        verifier_specs = [(prompts.FORMAL_VERIFIER_PROMPT, "形式与学科核验")]
        if self.runtime.settings.semantic_verifier_count == 2:
            verifier_specs.append((prompts.REQUIREMENTS_VERIFIER_PROMPT, "要求与教学核验"))
        for prompt, stage in verifier_specs:
            try:
                passes.append(self.runtime.json(prompt, common, stage=stage, validator=validator))
            except Exception as exc:
                warnings.append(self._warning(stage, exc))

        reported: dict[tuple[str, str], list[Issue]] = {}
        summaries: list[str] = []
        for issues, summary in passes:
            summaries.append(summary)
            for issue in issues:
                reported.setdefault((issue.origin.value, issue.quote), []).append(issue)
        accepted: list[Issue] = []
        for same_quote in reported.values():
            reported_issue = (
                self._merge_issues(same_quote) if len(same_quote) >= 2 else same_quote[0]
            )
            try:
                adjudicated = self._adjudicate_issue(
                    question, candidate, reference, evidence, reported_issue
                )
            except Exception as exc:
                warnings.append(self._warning("问题裁决", exc))
                adjudicated = None
            if adjudicated is not None:
                accepted.append(adjudicated)
        return accepted, summaries, warnings

    @staticmethod
    def _merge_issues(issues: Sequence[Issue]) -> Issue:
        first = issues[0]
        severity = (
            Severity.MAJOR
            if any(item.severity is Severity.MAJOR for item in issues)
            else Severity.MINOR
        )
        evidence_ids = tuple(dict.fromkeys(value for item in issues for value in item.evidence_ids))
        correction = next((item.correction for item in issues if item.correction), "")
        problem = "；".join(dict.fromkeys(item.problem for item in issues))
        return Issue(first.origin, severity, first.quote, problem, correction, evidence_ids)

    def _adjudicate_issue(
        self,
        question: str,
        candidate: str,
        reference: str,
        evidence: Sequence[Evidence],
        issue: Issue,
    ) -> Issue | None:
        def validate(data):
            if data.get("decision") not in {"valid", "invalid"}:
                raise StructuredResponseError("问题裁决 decision 非法")
            if data.get("severity") not in {"major", "minor"}:
                raise StructuredResponseError("问题裁决 severity 非法")
            return data

        data = self.runtime.json(
            prompts.ISSUE_ADJUDICATOR_PROMPT,
            "【原始问题】\n"
            + question
            + "\n\n【候选答案】\n"
            + candidate
            + "\n\n【独立候选】\n"
            + (reference or "未启用")
            + "\n\n【工具证据】\n"
            + json.dumps([item.__dict__ for item in evidence], ensure_ascii=False, default=list)
            + "\n\n【待裁决问题】\n"
            + json.dumps(issue.__dict__, ensure_ascii=False, default=list),
            stage="问题裁决",
            validator=validate,
        )
        if data["decision"] == "invalid":
            return None
        return Issue(
            origin=issue.origin,
            severity=Severity(data["severity"]),
            quote=issue.quote,
            problem=str(data.get("problem") or issue.problem).strip(),
            correction=str(data.get("correction") or issue.correction).strip(),
            evidence_ids=issue.evidence_ids,
        )
