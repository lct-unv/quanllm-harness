from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from threading import Lock
from typing import Any

from ..contracts import EventSink, Evidence, HarnessEvent

ToolHandler = Callable[[Mapping[str, Any]], Any]
ArgumentValidator = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: ToolHandler
    limitations: Sequence[str] = ()
    claim_kinds: Sequence[str] = ()
    argument_validator: ArgumentValidator | None = None
    planner_visible: bool = True
    plugin_name: str = ""
    plugin_version: str = ""
    plugin_digest: str = ""
    execution_mode: str = "builtin"

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)
    _evidence_counter: int = 0
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _cache: dict[tuple[str, str], Evidence] = field(default_factory=dict, init=False, repr=False)

    def register(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"工具重复注册：{tool.name}")
        self.tools[tool.name] = tool

    def register_many(self, tools: Sequence[Tool]) -> None:
        for tool in tools:
            if not isinstance(tool, Tool):
                raise TypeError("工具插件只能返回 Tool 实例")
            self.register(tool)

    def load_entrypoint_tools(self, group: str = "quanllm_harness.tools") -> None:
        """Load tools provided by trusted installed Python distributions."""

        discovered = entry_points()
        selector = getattr(discovered, "select", None)
        legacy_discovered: Any = discovered
        selected = (
            selector(group=group) if selector is not None else legacy_discovered.get(group, ())
        )
        for entrypoint in selected:
            value = entrypoint.load()()
            tools = (value,) if isinstance(value, Tool) else tuple(value)
            self.register_many(tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.openai_schema() for tool in self.tools.values() if tool.planner_visible]

    def capabilities(self) -> dict[str, Any]:
        return {
            name: {
                "description": tool.description,
                "claim_kinds": list(tool.claim_kinds),
                "limitations": list(tool.limitations),
                "planner_visible": tool.planner_visible,
                "plugin_name": tool.plugin_name,
                "plugin_version": tool.plugin_version,
                "plugin_digest": tool.plugin_digest,
                "execution_mode": tool.execution_mode,
            }
            for name, tool in self.tools.items()
        }

    def validate_call(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        claim_kind: str | None = None,
    ) -> None:
        tool = self.tools.get(name)
        if tool is None:
            raise ValueError(f"未知工具：{name}")
        if not isinstance(arguments, Mapping):
            raise ValueError("工具参数必须是对象")
        if claim_kind and tool.claim_kinds and claim_kind not in tool.claim_kinds:
            raise ValueError(f"工具 {name} 不接受 {claim_kind} 类型断言")
        try:
            from jsonschema import Draft202012Validator
        except ImportError:
            Draft202012Validator = None
        if Draft202012Validator is not None:
            errors = sorted(
                Draft202012Validator(tool.parameters).iter_errors(dict(arguments)),
                key=lambda error: list(error.path),
            )
            if errors:
                raise ValueError("工具参数不符合 Schema：" + errors[0].message)
        if tool.argument_validator:
            tool.argument_validator(arguments)

    def rejected_evidence(
        self,
        name: str,
        arguments: Mapping[str, Any],
        reason: str,
        *,
        event_sink: EventSink | None = None,
    ) -> Evidence:
        with self._lock:
            self._evidence_counter += 1
            evidence_id = f"E-{self._evidence_counter:04d}"
        evidence = Evidence(
            id=evidence_id,
            tool=name,
            operation=str(arguments.get("operation", name)),
            arguments=dict(arguments),
            ok=False,
            error=f"工具调用前语义审查未通过：{reason}",
            input_verified=False,
            review=reason,
        )
        if event_sink:
            event_sink(
                HarnessEvent(
                    "tool_finished",
                    name,
                    {"evidence_id": evidence.id, "ok": False, "error": evidence.error},
                )
            )
        return evidence

    def execute(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        event_sink: EventSink | None = None,
    ) -> Evidence:
        signature = (
            name,
            json.dumps(dict(arguments), ensure_ascii=False, sort_keys=True, default=str),
        )
        with self._lock:
            cached = self._cache.get(signature)
            if cached is not None:
                if event_sink:
                    event_sink(
                        HarnessEvent(
                            "tool_reused", name, {"evidence_id": cached.id, "ok": cached.ok}
                        )
                    )
                return cached
            self._evidence_counter += 1
            evidence_id = f"E-{self._evidence_counter:04d}"
        if event_sink:
            event_sink(HarnessEvent("tool_started", name, {"evidence_id": evidence_id}))
        tool = self.tools.get(name)
        if tool is None:
            evidence = Evidence(
                id=evidence_id,
                tool=name,
                operation="unknown",
                arguments=dict(arguments),
                ok=False,
                error="未知工具",
            )
        else:
            try:
                self.validate_call(name, arguments)
                result = tool.handler(arguments)
                evidence = Evidence(
                    id=evidence_id,
                    tool=name,
                    operation=str(arguments.get("operation", name)),
                    arguments=dict(arguments),
                    ok=True,
                    result=result,
                    limitations=tuple(tool.limitations),
                    plugin_name=tool.plugin_name,
                    plugin_version=tool.plugin_version,
                    plugin_digest=tool.plugin_digest,
                    execution_mode=tool.execution_mode,
                )
            except Exception as exc:
                evidence = Evidence(
                    id=evidence_id,
                    tool=name,
                    operation=str(arguments.get("operation", name)),
                    arguments=dict(arguments),
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                    limitations=tuple(tool.limitations),
                    plugin_name=tool.plugin_name,
                    plugin_version=tool.plugin_version,
                    plugin_digest=tool.plugin_digest,
                    execution_mode=tool.execution_mode,
                )
        if event_sink:
            event_sink(
                HarnessEvent(
                    "tool_finished",
                    name,
                    {"evidence_id": evidence.id, "ok": evidence.ok, "error": evidence.error},
                )
            )
        with self._lock:
            self._cache[signature] = evidence
        return evidence


def default_tool_registry(*, include_plugins: bool = True) -> ToolRegistry:
    from .numeric_backend import numeric_tools
    from .operator_backend import operator_tools
    from .quantum_backends import quantum_tools
    from .sympy_backend import sympy_tools

    registry = ToolRegistry()
    for tool in (*sympy_tools(), *operator_tools(), *numeric_tools(), *quantum_tools()):
        registry.register(tool)
    if include_plugins:
        registry.load_entrypoint_tools()
    return registry
