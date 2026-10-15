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

# Structured evidence must stay JSON-portable so it can be embedded verbatim in
# every protocol payload, run record and plugin result.
_JSON_LEAF = (bool, int, float, str)
_MAX_EVIDENCE_DEPTH = 12


def _validate_evidence_payload(value: Any, depth: int = 0) -> None:
    if depth > _MAX_EVIDENCE_DEPTH:
        raise ValueError("工具结果嵌套层级过深")
    if value is None or isinstance(value, _JSON_LEAF):
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_evidence_payload(item, depth + 1)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("工具结果对象的键必须是字符串")
            _validate_evidence_payload(item, depth + 1)
        return
    raise ValueError(f"工具结果包含不可序列化的类型：{type(value).__name__}")


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
    # Issue 004 - reliability layer. A handler failure is retried up to
    # ``retry_attempts`` times; if every attempt fails and ``fallback`` is set,
    # the fallback handler runs and its evidence records that it was used.
    retry_attempts: int = 1
    fallback: ToolHandler | None = None
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
        if tool.retry_attempts < 1:
            raise ValueError(f"工具 {tool.name} 的 retry_attempts 必须为正整数")
        if tool.fallback is not None and not callable(tool.fallback):
            raise TypeError(f"工具 {tool.name} 的 fallback 必须可调用")
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
                "retry_attempts": tool.retry_attempts,
                "has_fallback": tool.fallback is not None,
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
        error_kind: str = "review",
    ) -> Evidence:
        with self._lock:
            self._evidence_counter += 1
            evidence_id = f"E-{self._evidence_counter:04d}"
        prefix = (
            "工具调用前 Schema 预校验未通过"
            if error_kind == "schema"
            else "工具调用前语义审查未通过"
        )
        evidence = Evidence(
            id=evidence_id,
            tool=name,
            operation=str(arguments.get("operation", name)),
            arguments=dict(arguments),
            ok=False,
            error=f"{prefix}：{reason}",
            input_verified=False,
            review=reason,
            error_kind=error_kind,
            attempts=0 if error_kind == "schema" else 1,
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
                error_kind="unknown",
                attempts=0,
            )
        else:
            evidence = self._run_tool(tool, dict(arguments), evidence_id, event_sink)
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

    def _run_tool(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        evidence_id: str,
        event_sink: EventSink | None,
    ) -> Evidence:
        """Pre-validate, run with a bounded retry policy, then fall back."""

        def build(
            ok: bool,
            *,
            result: Any = None,
            error: str = "",
            error_kind: str = "",
            attempts: int = 1,
            fallback_used: bool = False,
            limitations: tuple[str, ...] | None = None,
        ) -> Evidence:
            return Evidence(
                id=evidence_id,
                tool=tool.name,
                operation=str(arguments.get("operation", tool.name)),
                arguments=dict(arguments),
                ok=ok,
                result=result,
                error=error,
                limitations=tuple(tool.limitations if limitations is None else limitations),
                plugin_name=tool.plugin_name,
                plugin_version=tool.plugin_version,
                plugin_digest=tool.plugin_digest,
                execution_mode=tool.execution_mode,
                error_kind=error_kind,
                attempts=attempts,
                fallback_used=fallback_used,
            )

        try:
            self.validate_call(tool.name, arguments)
        except Exception as exc:
            return build(
                False,
                error=f"Schema 预校验失败：{type(exc).__name__}: {exc}",
                error_kind="schema",
                attempts=0,
            )

        attempts = 0
        last_error = ""
        last_kind = "handler"
        for attempt in range(1, tool.retry_attempts + 1):
            attempts = attempt
            try:
                result = tool.handler(arguments)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                last_kind = "handler"
            else:
                try:
                    _validate_evidence_payload(result)
                except ValueError as exc:
                    last_error = f"证据校验失败：{exc}"
                    last_kind = "evidence"
                else:
                    return build(True, result=result, attempts=attempts)
            if attempt < tool.retry_attempts and event_sink:
                event_sink(
                    HarnessEvent(
                        "tool_retried",
                        tool.name,
                        {
                            "evidence_id": evidence_id,
                            "attempt": attempt,
                            "next_attempt": attempt + 1,
                            "error": last_error,
                        },
                    )
                )

        if tool.fallback is None:
            return build(False, error=last_error, error_kind=last_kind, attempts=attempts)

        try:
            fallback_result = tool.fallback(arguments)
        except Exception as exc:
            return build(
                False,
                error=f"{last_error}；回退实现同样失败：{type(exc).__name__}: {exc}",
                error_kind=last_kind,
                attempts=attempts,
                limitations=(*tool.limitations, "回退实现失败"),
            )
        try:
            _validate_evidence_payload(fallback_result)
        except ValueError as exc:
            return build(
                False,
                error=f"回退实现证据校验失败：{exc}",
                error_kind="evidence",
                attempts=attempts,
                limitations=(*tool.limitations, "回退实现失败"),
            )
        if event_sink:
            event_sink(
                HarnessEvent(
                    "tool_fallback_used",
                    tool.name,
                    {"evidence_id": evidence_id, "error": last_error},
                )
            )
        return build(
            True,
            result=fallback_result,
            error=last_error,
            attempts=attempts,
            fallback_used=True,
            limitations=(*tool.limitations, "首次实现失败，已使用回退实现"),
        )


def default_tool_registry(*, include_plugins: bool = True) -> ToolRegistry:
    from .numeric_backend import numeric_tools
    from .sympy_backend import sympy_tools

    registry = ToolRegistry()
    for tool in (*sympy_tools(), *numeric_tools()):
        registry.register(tool)
    if include_plugins:
        registry.load_entrypoint_tools()
    return registry
