from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..contracts import Claim
from ..protocols import StructuredResponseError
from ..protocols.claim_extraction import contains_quote
from . import prompts
from .runtime import AgentRuntime


@dataclass(frozen=True)
class ToolCallReview:
    decision: str
    tool: str
    arguments: dict[str, Any]
    source_anchors: tuple[str, ...]
    expected_boolean: bool | None
    reason: str


@dataclass
class ToolCallReviewerAgent:
    runtime: AgentRuntime

    def review(
        self,
        *,
        question: str,
        candidate: str,
        claim: Claim,
        tool: str,
        arguments: dict[str, Any],
        purpose: str,
        schemas: list[dict[str, Any]],
    ) -> ToolCallReview:
        def validate(data: dict[str, Any]) -> ToolCallReview:
            decision = str(data.get("decision") or "")
            reviewed_tool = str(data.get("tool") or "")
            reviewed_arguments = data.get("arguments")
            source_anchors = data.get("source_anchors")
            expected_boolean = data.get("expected_boolean")
            reason = str(data.get("reason") or "").strip()
            if decision not in {"approve", "correct", "reject"}:
                raise StructuredResponseError("工具调用审查 decision 非法")
            if not reason:
                raise StructuredResponseError("工具调用审查缺少语义依据")
            if expected_boolean is not None and not isinstance(expected_boolean, bool):
                raise StructuredResponseError("工具调用审查 expected_boolean 非法")
            if decision == "reject":
                return ToolCallReview(
                    decision, reviewed_tool or tool, {}, (), expected_boolean, reason
                )
            if not reviewed_tool or not isinstance(reviewed_arguments, dict):
                raise StructuredResponseError("工具调用审查的重建结果不完整")
            if not isinstance(source_anchors, list) or not source_anchors:
                raise StructuredResponseError("工具调用审查缺少逐字来源片段")
            anchors: list[str] = []
            sources = (question, candidate)
            for anchor in source_anchors:
                if not isinstance(anchor, str) or not anchor.strip():
                    raise StructuredResponseError("工具调用审查来源片段非法")
                if not any(contains_quote(anchor, source) for source in sources):
                    raise StructuredResponseError(
                        "工具调用审查来源片段无法逐字定位（片段前 80 字符）："
                        + (anchor[:80] if len(anchor) > 80 else anchor)
                    )
                anchors.append(anchor)
            self.runtime.tools.validate_call(
                reviewed_tool, reviewed_arguments, claim_kind=claim.kind
            )
            reconstructed = dict(reviewed_arguments)
            effective_decision = (
                "approve"
                if reviewed_tool == tool and reconstructed == dict(arguments)
                else "correct"
            )
            return ToolCallReview(
                effective_decision,
                reviewed_tool,
                reconstructed,
                tuple(anchors),
                expected_boolean,
                reason,
            )

        payload = {
            "question": question,
            "candidate": candidate,
            "claim": claim.__dict__,
            "planned_call": {
                "tool": tool,
                "arguments": arguments,
                "purpose": purpose,
            },
            "available_tool_schemas": schemas,
        }
        return self.runtime.json_once(
            prompts.TOOL_CALL_REVIEWER_PROMPT,
            json.dumps(payload, ensure_ascii=False),
            stage=f"工具调用审查·{tool}",
            validator=validate,
        )
