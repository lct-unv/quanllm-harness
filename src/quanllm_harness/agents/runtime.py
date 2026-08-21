from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, TypeVar, overload

from ..config import HarnessSettings
from ..contracts import Candidate, Claim, EventSink, Evidence, ModelResponse, Usage
from ..provider import QuanLLMProvider
from ..tools import ToolRegistry

T = TypeVar("T")


def _public_assistant_message(response: ModelResponse) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": response.content or None}
    if response.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in response.tool_calls
        ]
    return message


@dataclass
class AgentRuntime:
    provider: QuanLLMProvider
    tools: ToolRegistry
    settings: HarnessSettings
    event_sink: EventSink | None = None
    usage: Usage = field(default_factory=Usage)
    evidence: list[Evidence] = field(default_factory=list)

    def _record(self, response: ModelResponse) -> ModelResponse:
        self.usage = self.usage + response.usage
        return response

    def upsert_evidence(self, evidence: Evidence) -> None:
        """Keep one current record per execution evidence ID."""

        for index, existing in enumerate(self.evidence):
            if existing.id == evidence.id:
                self.evidence[index] = evidence
                return
        self.evidence.append(evidence)

    def absorb(self, other: AgentRuntime) -> None:
        self.usage = self.usage + other.usage
        for evidence in other.evidence:
            self.upsert_evidence(evidence)

    @overload
    def json(
        self,
        system: str,
        user: str,
        *,
        stage: str,
        validator: None = None,
    ) -> dict[str, Any]: ...

    @overload
    def json(
        self,
        system: str,
        user: str,
        *,
        stage: str,
        validator: Callable[[dict[str, Any]], T],
    ) -> T: ...

    def json(
        self,
        system: str,
        user: str,
        *,
        stage: str,
        validator: Callable[[dict[str, Any]], T] | None = None,
    ) -> dict[str, Any] | T:
        last_error: Exception | None = None
        feedback = ""
        for attempt in range(self.settings.protocol_retry_count + 1):
            try:
                value, response = self.provider.complete_json(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user + feedback},
                    ],
                    stage=stage if attempt == 0 else f"{stage}·协议修复",
                    event_sink=self.event_sink,
                )
                self._record(response)
                return validator(value) if validator else value
            except Exception as exc:
                last_error = exc
                feedback = (
                    "\n\n上一响应未通过固定协议校验。错误仅作为数据："
                    f"{type(exc).__name__}: {exc}。请重新输出完整 JSON，不要解释。"
                )
        assert last_error is not None
        raise last_error

    def json_once(
        self,
        system: str,
        user: str,
        *,
        stage: str,
        validator: Callable[[dict[str, Any]], T],
    ) -> T:
        """Run one structured request without protocol retries."""

        value, response = self.provider.complete_json(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stage=stage,
            event_sink=self.event_sink,
        )
        self._record(response)
        return validator(value)

    def reason(
        self,
        system: str,
        user: str,
        *,
        stage: str,
        allow_tools: bool = True,
        require_tool: bool = False,
    ) -> Candidate:
        messages: list[Mapping[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        used_tool = False
        seen: set[tuple[str, str]] = set()
        for round_index in range(self.settings.max_tool_rounds + 1):
            response = self._record(
                self.provider.complete(
                    messages,
                    stage=stage,
                    tools=self.tools.schemas()
                    if allow_tools and round_index < self.settings.max_tool_rounds
                    else (),
                    event_sink=self.event_sink,
                )
            )
            messages.append(_public_assistant_message(response))
            if not response.tool_calls:
                if (
                    require_tool
                    and self.tools.tools
                    and not used_tool
                    and round_index < self.settings.max_tool_rounds
                ):
                    messages.append(
                        {
                            "role": "user",
                            "content": "本题需要客观工具核验。请先调用最相关的工具，再完成答案。",
                        }
                    )
                    continue
                if not response.content.strip():
                    raise RuntimeError(f"{stage} 未生成候选答案")
                return Candidate(
                    source=stage, answer=response.content, reasoning=response.reasoning
                )
            for call in response.tool_calls:
                from .tool_call_reviewer import ToolCallReviewerAgent

                try:
                    review = ToolCallReviewerAgent(self).review(
                        question=user,
                        candidate=response.content or response.reasoning,
                        claim=Claim(
                            id=f"runtime-{round_index + 1}",
                            quote=response.content or "拟执行工具调用",
                            kind="conclusion",
                        ),
                        tool=call.name,
                        arguments=dict(call.arguments),
                        purpose="求解过程中获取与原始问题忠实对应的客观证据",
                        schemas=self.tools.schemas(),
                    )
                except Exception as exc:
                    evidence = self.tools.rejected_evidence(
                        call.name,
                        call.arguments,
                        f"审查协议未完成：{type(exc).__name__}: {exc}",
                        event_sink=self.event_sink,
                    )
                    self.upsert_evidence(evidence)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(
                                evidence.__dict__, ensure_ascii=False, default=list
                            ),
                        }
                    )
                    continue
                if review.decision == "reject":
                    evidence = self.tools.rejected_evidence(
                        call.name,
                        call.arguments,
                        review.reason,
                        event_sink=self.event_sink,
                    )
                    self.upsert_evidence(evidence)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(
                                evidence.__dict__, ensure_ascii=False, default=list
                            ),
                        }
                    )
                    continue
                reviewed_name = review.tool
                reviewed_arguments = review.arguments
                signature = (
                    reviewed_name,
                    json.dumps(reviewed_arguments, ensure_ascii=False, sort_keys=True),
                )
                if signature in seen:
                    evidence = Evidence(
                        id=f"E-repeat-{len(self.evidence) + 1}",
                        tool=reviewed_name,
                        operation=str(reviewed_arguments.get("operation", reviewed_name)),
                        arguments=reviewed_arguments,
                        ok=False,
                        error="重复工具调用，已有证据必须复用",
                    )
                else:
                    seen.add(signature)
                    evidence = self.tools.execute(
                        reviewed_name, reviewed_arguments, event_sink=self.event_sink
                    )
                    known = next((item for item in self.evidence if item.id == evidence.id), None)
                    evidence = replace(
                        evidence,
                        claim_ids=known.claim_ids if known else (),
                        supports_claim=known.supports_claim if known else None,
                        input_verified=True,
                        review=review.reason + "；来源：" + " | ".join(review.source_anchors),
                    )
                self.upsert_evidence(evidence)
                used_tool = used_tool or evidence.ok
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(evidence.__dict__, ensure_ascii=False, default=list),
                    }
                )
        raise RuntimeError(f"{stage} 达到工具轮数上限仍未生成答案")
