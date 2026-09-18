from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from ..agents import ToolCallReviewerAgent, prompts
from ..agents.runtime import AgentRuntime
from ..contracts import Claim, Evidence, HarnessEvent
from ..protocols import StructuredResponseError


@dataclass
class MathematicalVerifier:
    runtime: AgentRuntime

    def collect(
        self,
        question: str,
        candidate: str,
        claims: Sequence[Claim],
    ) -> tuple[list[Evidence], list[str], set[str]]:
        schemas = self.runtime.tools.schemas()
        claims_by_id = {claim.id: claim for claim in claims}
        not_checkable_ids: set[str] = set()

        def validate(data: dict[str, Any]):
            checks = data.get("checks")
            not_checkable = data.get("not_checkable")
            if not isinstance(checks, list) or not isinstance(not_checkable, list):
                raise StructuredResponseError("工具计划必须同时包含 checks 与 not_checkable 数组")
            validated: list[tuple[str, str, dict[str, Any], str]] = []
            covered: set[str] = set()
            signatures: set[tuple[str, str]] = set()
            for _index, check in enumerate(checks, 1):
                if not isinstance(check, dict):
                    continue
                claim_id = str(check.get("claim_id") or "")
                tool_name = str(check.get("tool") or "")
                arguments = check.get("arguments")
                purpose = str(check.get("purpose") or "").strip()
                if claim_id not in claims_by_id or not isinstance(arguments, dict):
                    continue
                if not purpose:
                    continue
                try:
                    self.runtime.tools.validate_call(
                        tool_name, arguments, claim_kind=claims_by_id[claim_id].kind
                    )
                except ValueError:
                    # A single invalid check must not invalidate the whole plan;
                    # the evidence gate later flags any claim left without a
                    # successful tool evidence, so nothing passes silently.
                    continue
                signature = (tool_name, json.dumps(arguments, ensure_ascii=False, sort_keys=True))
                if signature not in signatures:
                    validated.append((claim_id, tool_name, arguments, purpose))
                    signatures.add(signature)
                covered.add(claim_id)
            for _index, item in enumerate(not_checkable, 1):
                if not isinstance(item, dict):
                    continue
                claim_id = str(item.get("claim_id") or "")
                reason = str(item.get("reason") or "").strip()
                if claim_id not in claims_by_id or not reason:
                    continue
                covered.add(claim_id)
                not_checkable_ids.add(claim_id)
            return validated

        plans = self.runtime.json(
            prompts.TOOL_PLANNER_PROMPT,
            "【原始问题】\n"
            + question
            + "\n\n【候选答案】\n"
            + candidate
            + "\n\n【关键断言】\n"
            + json.dumps([claim.__dict__ for claim in claims], ensure_ascii=False)
            + "\n\n【可用工具 Schema】\n"
            + json.dumps(schemas, ensure_ascii=False),
            stage="工具核验计划",
            validator=validate,
        )
        evidence: list[Evidence] = []
        warnings: list[str] = []
        reviewer = ToolCallReviewerAgent(self.runtime)
        for claim_id, tool_name, arguments, purpose in plans:
            claim = claims_by_id[claim_id]
            try:
                review = reviewer.review(
                    question=question,
                    candidate=candidate,
                    claim=claim,
                    tool=tool_name,
                    arguments=arguments,
                    purpose=purpose,
                    schemas=schemas,
                )
            except Exception as exc:
                warning = f"{claim_id} 工具调用审查未完成：{type(exc).__name__}: {exc}"
                warnings.append(warning)
                if self.runtime.event_sink:
                    self.runtime.event_sink(
                        HarnessEvent("tool_review_failed", tool_name, {"reason": warning})
                    )
                continue
            if review.decision == "reject":
                warning = f"{claim_id} 工具调用被语义审查拒绝：{review.reason}"
                warnings.append(warning)
                if self.runtime.event_sink:
                    self.runtime.event_sink(
                        HarnessEvent("tool_review_rejected", tool_name, {"reason": warning})
                    )
                continue
            item = self.runtime.tools.execute(
                review.tool, review.arguments, event_sink=self.runtime.event_sink
            )
            existing = next((known for known in self.runtime.evidence if known.id == item.id), None)
            claim_ids = tuple(dict.fromkeys((*(existing.claim_ids if existing else ()), claim_id)))
            support: bool | None = None
            if item.ok and review.expected_boolean is not None and isinstance(item.result, dict):
                for key in ("equivalent", "all_match", "zero", "hermitian", "unitary"):
                    value = item.result.get(key)
                    if item.result.get("resolved") is not False and isinstance(value, bool):
                        support = value == review.expected_boolean
                        break
            item = replace(
                item,
                claim_ids=claim_ids,
                input_verified=True,
                supports_claim=support,
                review=review.reason + "；来源：" + " | ".join(review.source_anchors),
            )
            evidence.append(item)
            self.runtime.upsert_evidence(item)
        return evidence, warnings, not_checkable_ids
