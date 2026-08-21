from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..contracts import RequestPolicy
from ..protocols import StructuredResponseError
from . import prompts
from .runtime import AgentRuntime


@dataclass
class RouterAgent:
    runtime: AgentRuntime

    def route(self, question: str) -> RequestPolicy:
        def validate(data: dict[str, Any]) -> RequestPolicy:
            depth = data.get("depth")
            if depth not in {"simple", "standard", "deep"}:
                raise StructuredResponseError("任务路由 depth 非法")
            bool_fields = ("suspicious_input", "requires_tools", "requires_independent_solver")
            if any(not isinstance(data.get(name), bool) for name in bool_fields):
                raise StructuredResponseError("任务路由布尔字段非法")
            domains = data.get("tool_domains") or []
            if not isinstance(domains, list) or not all(isinstance(item, str) for item in domains):
                raise StructuredResponseError("任务路由 tool_domains 非法")
            return RequestPolicy(
                depth=depth,
                suspicious_input=data["suspicious_input"],
                requires_tools=data["requires_tools"],
                requires_independent_solver=data["requires_independent_solver"],
                language=str(data.get("language") or "zh"),
                reason=str(data.get("reason") or ""),
                tool_domains=tuple(domains),
            )

        return self.runtime.json(
            prompts.ROUTER_PROMPT,
            "【用户问题】\n" + question,
            stage="任务路由",
            validator=validate,
        )
