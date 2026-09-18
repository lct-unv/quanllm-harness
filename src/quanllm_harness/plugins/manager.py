from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from importlib.metadata import EntryPoint, entry_points, version
from pathlib import Path
from threading import Lock
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from ..contracts import EventSink, HarnessEvent
from .api import (
    Cleanup,
    EventSubscriber,
    PluginContext,
    PluginManifest,
    PluginPolicy,
    PluginStatus,
    PluginVerifier,
    ProviderFactory,
    SubprocessToolSpec,
)
from .runner import run_subprocess_tool

EntryPointsFunction = Callable[[], Any]


@dataclass
class _LoadedPlugin:
    entrypoint: EntryPoint
    instance: Any
    manifest: PluginManifest
    digest: str
    cleanup: Cleanup | None = None
    tool_names: list[str] = field(default_factory=list)
    verifier_names: list[str] = field(default_factory=list)
    provider_names: list[str] = field(default_factory=list)


class _PermissionedRegistrar:
    def __init__(self, manager: PluginManager, plugin: _LoadedPlugin, permission: str):
        self.manager = manager
        self.plugin = plugin
        self.permission = permission

    def require(self) -> None:
        if self.permission not in self.plugin.manifest.permissions:
            raise PermissionError(f"插件 {self.plugin.manifest.name} 未声明权限 {self.permission}")


class _ToolRegistrar(_PermissionedRegistrar):
    def register(self, tool: Any) -> None:
        from ..tools import Tool

        self.require()
        if not isinstance(tool, Tool):
            raise TypeError("插件只能注册 Tool 实例")
        plugin_name = self.plugin.manifest.name
        if "." not in tool.name or not tool.name.startswith(plugin_name + "."):
            raise ValueError(f"插件工具必须使用 {plugin_name}. 前缀，实际为 {tool.name!r}")
        owned = replace(
            tool,
            plugin_name=plugin_name,
            plugin_version=self.plugin.manifest.version,
            plugin_digest=self.plugin.digest,
            execution_mode=self.plugin.manifest.execution_mode,
        )
        self.manager._register_tool(owned)
        self.plugin.tool_names.append(owned.name)


class _EventRegistrar(_PermissionedRegistrar):
    def subscribe(self, callback: EventSubscriber) -> None:
        self.require()
        if not callable(callback):
            raise TypeError("事件订阅器必须可调用")
        self.manager._event_subscribers.append((self.plugin.manifest.name, callback))


class _VerifierRegistrar(_PermissionedRegistrar):
    def register(self, name: str, verifier: PluginVerifier) -> None:
        self.require()
        full_name = self.manager._namespaced(self.plugin.manifest.name, name)
        if not callable(verifier):
            raise TypeError("插件核验器必须可调用")
        if any(existing == full_name for existing, _, _ in self.manager._verifiers):
            raise ValueError(f"插件核验器重复注册：{full_name}")
        self.manager._verifiers.append((full_name, self.plugin.manifest.name, verifier))
        self.plugin.verifier_names.append(full_name)


class _ProviderRegistrar(_PermissionedRegistrar):
    def register(self, name: str, factory: ProviderFactory) -> None:
        self.require()
        if "secrets.api_key" not in self.plugin.manifest.permissions:
            raise PermissionError(
                f"Provider 插件 {self.plugin.manifest.name} 必须声明 secrets.api_key 权限"
            )
        full_name = self.manager._namespaced(self.plugin.manifest.name, name)
        if not callable(factory):
            raise TypeError("Provider factory 必须可调用")
        if full_name in self.manager._providers:
            raise ValueError(f"插件 Provider 重复注册：{full_name}")
        self.manager._providers[full_name] = (self.plugin.manifest.name, factory)
        self.plugin.provider_names.append(full_name)


class _ServiceRegistrar(_PermissionedRegistrar):
    def provide(self, name: str, value: Any) -> None:
        if "services.provide" not in self.plugin.manifest.permissions:
            raise PermissionError(f"插件 {self.plugin.manifest.name} 未声明权限 services.provide")
        full_name = self.manager._namespaced(self.plugin.manifest.name, name)
        if full_name in self.manager._services:
            raise ValueError(f"插件服务重复注册：{full_name}")
        self.manager._services[full_name] = value

    def get(self, name: str) -> Any:
        if "services.consume" not in self.plugin.manifest.permissions:
            raise PermissionError(f"插件 {self.plugin.manifest.name} 未声明权限 services.consume")
        if name not in self.manager._services:
            raise KeyError(f"插件服务不存在：{name}")
        provider = name.split(".", 1)[0]
        if (
            provider != self.plugin.manifest.name
            and provider not in self.plugin.manifest.dependencies
        ):
            raise PermissionError(
                f"插件 {self.plugin.manifest.name} 未声明对服务提供方 {provider} 的依赖"
            )
        return self.manager._services[name]


class PluginManager:
    """Discover, validate and host plugins without creating a separate SDK package."""

    def __init__(
        self,
        policy: PluginPolicy | None = None,
        *,
        entry_points_function: EntryPointsFunction = entry_points,
    ):
        self.policy = policy or PluginPolicy()
        self.policy.validate()
        self._entry_points_function = entry_points_function
        self._plugins: dict[str, _LoadedPlugin] = {}
        self._statuses: dict[str, PluginStatus] = {}
        self._tools: dict[str, Any] = {}
        self._event_subscribers: list[tuple[str, EventSubscriber]] = []
        self._verifiers: list[tuple[str, str, PluginVerifier]] = []
        self._providers: dict[str, tuple[str, ProviderFactory]] = {}
        self._services: dict[str, Any] = {}
        self._diagnostics: list[str] = []
        self._started = False
        self._lock = Lock()

    @classmethod
    def discover(
        cls,
        policy: PluginPolicy | None = None,
        *,
        include_legacy_tools: bool = True,
        entry_points_function: EntryPointsFunction = entry_points,
    ) -> PluginManager:
        manager = cls(policy, entry_points_function=entry_points_function)
        manager._discover_modern()
        manager._start_plugins()
        if include_legacy_tools:
            manager._discover_legacy_tools()
        manager._started = True
        return manager

    @staticmethod
    def _namespaced(plugin_name: str, local_name: str) -> str:
        if not local_name or any(character.isspace() for character in local_name):
            raise ValueError("扩展名称不能为空或包含空白")
        return (
            local_name
            if local_name.startswith(plugin_name + ".")
            else f"{plugin_name}.{local_name}"
        )

    def _select(self, group: str) -> Sequence[EntryPoint]:
        discovered = self._entry_points_function()
        selector = getattr(discovered, "select", None)
        if selector is not None:
            return tuple(selector(group=group))
        return tuple(discovered.get(group, ()))

    @staticmethod
    def _distribution_digest(entrypoint: EntryPoint) -> str:
        distribution = getattr(entrypoint, "dist", None)
        files = getattr(distribution, "files", None)
        if not distribution or not files:
            return ""
        digest = hashlib.sha256()
        for item in sorted(files, key=str):
            path = Path(distribution.locate_file(item))
            if not path.is_file() or path.suffix in {".pyc", ".pyo"}:
                continue
            digest.update(str(item).encode("utf-8"))
            try:
                digest.update(path.read_bytes())
            except OSError:
                return ""
        return digest.hexdigest()

    @staticmethod
    def _host_version() -> Version:
        try:
            return Version(version("quanllm-harness"))
        except Exception:
            return Version("0.1.3")

    def _enabled(self, name: str) -> tuple[bool, str]:
        if name in self.policy.disabled:
            return False, "已被策略禁用"
        if self.policy.enabled:
            return (name in self.policy.enabled, "不在启用列表中")
        if not self.policy.allow_unlisted:
            return False, "策略禁止未列出的插件"
        return True, ""

    def _validate_manifest(self, manifest: PluginManifest, digest: str) -> None:
        manifest.validate()
        try:
            if self._host_version() not in SpecifierSet(manifest.requires_harness):
                raise ValueError(
                    f"插件要求 quanllm-harness{manifest.requires_harness}，"
                    f"当前为 {self._host_version()}"
                )
        except (InvalidSpecifier, InvalidVersion) as exc:
            raise ValueError(f"插件 {manifest.name} 版本约束非法") from exc
        denied = set(manifest.permissions) - set(self.policy.allowed_permissions)
        if denied:
            raise PermissionError(f"策略未授权插件权限：{sorted(denied)}")
        expected = self.policy.trusted_digests.get(manifest.name, "")
        if expected and digest != expected:
            raise PermissionError("插件发行内容摘要与 allowlist 不一致")
        if self.policy.require_trusted_digest and not expected:
            raise PermissionError("策略要求插件必须具有受信 SHA-256 摘要")

    @staticmethod
    def _plugin_instance(value: Any) -> Any:
        if hasattr(value, "manifest"):
            return value
        if callable(value):
            created = value()
            if hasattr(created, "manifest"):
                return created
        raise TypeError("插件入口必须返回具有 manifest 的插件对象")

    def _discover_modern(self) -> None:
        for entrypoint in self._select("quanllm_harness.plugins"):
            display_name = entrypoint.name
            digest = self._distribution_digest(entrypoint)
            enabled, disabled_reason = self._enabled(display_name)
            if not enabled:
                # Entry points execute arbitrary module-level Python while loading.
                # Apply the host allowlist before import so a disabled plugin is
                # discoverable without executing any of its code.
                self._statuses[display_name] = PluginStatus(
                    name=display_name,
                    state="disabled",
                    reason=disabled_reason,
                    digest=digest,
                    entry_point=entrypoint.value,
                )
                continue
            try:
                expected = self.policy.trusted_digests.get(display_name, "")
                if expected and digest != expected:
                    raise PermissionError("插件发行内容摘要与 allowlist 不一致")
                if self.policy.require_trusted_digest and not expected:
                    raise PermissionError("策略要求插件必须具有受信 SHA-256 摘要")
                instance = self._plugin_instance(entrypoint.load())
                manifest = instance.manifest
                if not isinstance(manifest, PluginManifest):
                    raise TypeError("manifest 必须是 PluginManifest")
                display_name = manifest.name
                if manifest.name != entrypoint.name:
                    raise ValueError(
                        f"插件 manifest.name 必须与入口名一致："
                        f"{manifest.name!r} != {entrypoint.name!r}"
                    )
                if display_name in self._plugins or display_name in self._statuses:
                    raise ValueError(f"插件名称重复：{display_name}")
                self._statuses[display_name] = PluginStatus(
                    name=display_name,
                    version=manifest.version,
                    state="discovered",
                    permissions=tuple(manifest.permissions),
                    dependencies=tuple(manifest.dependencies),
                    execution_mode=manifest.execution_mode,
                    digest=digest,
                    entry_point=entrypoint.value,
                )
                self._validate_manifest(manifest, digest)
                self._plugins[display_name] = _LoadedPlugin(
                    entrypoint=entrypoint,
                    instance=instance,
                    manifest=manifest,
                    digest=digest,
                )
            except Exception as exc:
                self._statuses[display_name] = PluginStatus(
                    name=display_name,
                    state="error",
                    reason=f"{type(exc).__name__}: {exc}",
                    entry_point=getattr(entrypoint, "value", ""),
                )

    def _dependency_order(self) -> list[str]:
        result: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            if name in visiting:
                raise ValueError(f"插件依赖存在环：{name}")
            visiting.add(name)
            plugin = self._plugins[name]
            for dependency in plugin.manifest.dependencies:
                if dependency not in self._plugins:
                    raise ValueError(f"插件 {name} 缺少依赖：{dependency}")
                visit(dependency)
            visiting.remove(name)
            visited.add(name)
            result.append(name)

        for plugin_name in sorted(self._plugins):
            visit(plugin_name)
        return result

    def _context(self, plugin: _LoadedPlugin) -> PluginContext:
        return PluginContext(
            manifest=plugin.manifest,
            config=dict(self.policy.config.get(plugin.manifest.name, {})),
            tools=_ToolRegistrar(self, plugin, "tools.register"),
            events=_EventRegistrar(self, plugin, "events.subscribe"),
            verifiers=_VerifierRegistrar(self, plugin, "verifiers.register"),
            providers=_ProviderRegistrar(self, plugin, "providers.register"),
            services=_ServiceRegistrar(self, plugin, "services.provide"),
        )

    def _register_subprocess_tools(self, plugin: _LoadedPlugin) -> None:
        from ..tools import Tool

        if "tools.register" not in plugin.manifest.permissions:
            raise PermissionError(f"插件 {plugin.manifest.name} 未声明权限 tools.register")
        specs = tuple(getattr(plugin.instance, "subprocess_tools", ()))
        if plugin.manifest.execution_mode == "subprocess" and not specs:
            raise ValueError("subprocess 插件必须声明 subprocess_tools")
        for spec in specs:
            if not isinstance(spec, SubprocessToolSpec):
                raise TypeError("subprocess_tools 只能包含 SubprocessToolSpec")
            spec.validate()

            def handler(arguments: Mapping[str, Any], *, current: SubprocessToolSpec = spec) -> Any:
                return run_subprocess_tool(
                    current.command,
                    plugin=plugin.manifest.name,
                    tool=current.name,
                    arguments=arguments,
                    timeout_seconds=current.timeout_seconds or self.policy.call_timeout_seconds,
                    max_output_bytes=self.policy.max_output_bytes,
                )

            tool = Tool(
                name=spec.name,
                description=spec.description,
                parameters=spec.parameters,
                handler=handler,
                limitations=tuple(spec.limitations),
                claim_kinds=tuple(spec.claim_kinds),
                planner_visible=spec.planner_visible,
                plugin_name=plugin.manifest.name,
                plugin_version=plugin.manifest.version,
                plugin_digest=plugin.digest,
                execution_mode="subprocess",
            )
            if not tool.name.startswith(plugin.manifest.name + "."):
                raise ValueError(f"子进程工具必须使用 {plugin.manifest.name}. 前缀：{tool.name}")
            self._register_tool(tool)
            plugin.tool_names.append(tool.name)

    def _start_plugins(self) -> None:
        try:
            order = self._dependency_order()
        except Exception as exc:
            for name in self._plugins:
                self._statuses[name] = replace(
                    self._statuses[name], state="error", reason=f"{type(exc).__name__}: {exc}"
                )
            self._plugins.clear()
            return
        for name in order:
            plugin = self._plugins[name]
            if any(
                self._statuses[dependency].state != "active"
                for dependency in plugin.manifest.dependencies
            ):
                self._statuses[name] = replace(
                    self._statuses[name], state="error", reason="依赖插件未成功启动"
                )
                continue
            try:
                if plugin.manifest.execution_mode == "subprocess":
                    self._register_subprocess_tools(plugin)
                else:
                    setup = getattr(plugin.instance, "setup", None)
                    if not callable(setup):
                        raise TypeError("in_process 插件必须提供 setup(context)")
                    cleanup = setup(self._context(plugin))
                    if cleanup is not None and not callable(cleanup):
                        raise TypeError("插件 setup 返回值必须是清理函数或 None")
                    plugin.cleanup = cleanup
                self._statuses[name] = replace(
                    self._statuses[name],
                    state="active",
                    tools=tuple(plugin.tool_names),
                    verifiers=tuple(plugin.verifier_names),
                    providers=tuple(plugin.provider_names),
                )
            except Exception as exc:
                self._remove_extensions(name)
                self._statuses[name] = replace(
                    self._statuses[name], state="error", reason=f"{type(exc).__name__}: {exc}"
                )

    def _discover_legacy_tools(self) -> None:
        from ..tools import Tool

        for entrypoint in self._select("quanllm_harness.tools"):
            name = f"legacy:{entrypoint.name}"
            enabled, disabled_reason = self._enabled(name)
            if not enabled:
                self._statuses[name] = PluginStatus(
                    name=name,
                    version="legacy",
                    state="disabled",
                    reason=disabled_reason,
                    permissions=("tools.register",),
                    execution_mode="in_process",
                    entry_point=entrypoint.value,
                )
                continue
            try:
                value = entrypoint.load()()
                tools = (value,) if isinstance(value, Tool) else tuple(value)
                names: list[str] = []
                for tool in tools:
                    if not isinstance(tool, Tool):
                        raise TypeError("旧式工具插件只能返回 Tool 实例")
                    self._register_tool(
                        replace(
                            tool,
                            plugin_name=name,
                            plugin_version="legacy",
                            plugin_digest=self._distribution_digest(entrypoint),
                            execution_mode="in_process",
                        )
                    )
                    names.append(tool.name)
                self._statuses[name] = PluginStatus(
                    name=name,
                    version="legacy",
                    state="active",
                    reason="兼容入口；建议迁移到 quanllm_harness.plugins",
                    permissions=("tools.register",),
                    execution_mode="in_process",
                    digest=self._distribution_digest(entrypoint),
                    entry_point=entrypoint.value,
                    tools=tuple(names),
                )
            except Exception as exc:
                self._statuses[name] = PluginStatus(
                    name=name,
                    state="error",
                    reason=f"{type(exc).__name__}: {exc}",
                    entry_point=getattr(entrypoint, "value", ""),
                )

    def _register_tool(self, tool: Any) -> None:
        if tool.name in self._tools:
            raise ValueError(f"插件工具重复注册：{tool.name}")
        self._tools[tool.name] = tool

    def _remove_extensions(self, plugin_name: str) -> None:
        self._tools = {
            name: tool
            for name, tool in self._tools.items()
            if getattr(tool, "plugin_name", "") != plugin_name
        }
        self._event_subscribers = [
            item for item in self._event_subscribers if item[0] != plugin_name
        ]
        self._verifiers = [item for item in self._verifiers if item[1] != plugin_name]
        self._providers = {
            name: value for name, value in self._providers.items() if value[0] != plugin_name
        }
        self._services = {
            name: value
            for name, value in self._services.items()
            if not name.startswith(plugin_name + ".")
        }

    def build_tool_registry(self, *, include_builtin: bool = True):
        from ..tools import ToolRegistry, default_tool_registry

        registry = (
            default_tool_registry(include_plugins=False) if include_builtin else ToolRegistry()
        )
        registry.register_many(tuple(self._tools.values()))
        return registry

    @property
    def verifiers(self) -> Sequence[tuple[str, str, PluginVerifier]]:
        return tuple(self._verifiers)

    def create_provider(self, name: str, settings: Any):
        if name not in self._providers:
            raise ValueError(f"未知插件 Provider：{name}")
        _, factory = self._providers[name]
        provider = factory(settings)
        from ..providers import QuanLLMProvider

        if not isinstance(provider, QuanLLMProvider):
            raise TypeError(f"插件 Provider {name} 未返回 QuanLLMProvider")
        return provider

    def event_sink(self, external: EventSink | None = None) -> EventSink:
        def publish(event: HarnessEvent) -> None:
            if external:
                external(event)
            for plugin_name, callback in tuple(self._event_subscribers):
                try:
                    callback(event)
                except Exception as exc:
                    with self._lock:
                        self._diagnostics.append(
                            f"插件 {plugin_name} 事件订阅失败：{type(exc).__name__}: {exc}"
                        )

        return publish

    def notify_request_start(self, question: str) -> None:
        for name, plugin in self._plugins.items():
            if self._statuses[name].state != "active":
                continue
            hook = getattr(plugin.instance, "request_started", None)
            if callable(hook):
                try:
                    hook(question)
                except Exception as exc:
                    with self._lock:
                        self._diagnostics.append(
                            f"插件 {name} 请求初始化失败：{type(exc).__name__}: {exc}"
                        )

    def notify_request_end(self, result: Any | None, error: BaseException | None = None) -> None:
        for name, plugin in reversed(tuple(self._plugins.items())):
            if self._statuses[name].state != "active":
                continue
            hook = getattr(plugin.instance, "request_finished", None)
            if callable(hook):
                try:
                    hook(result, error)
                except Exception as exc:
                    with self._lock:
                        self._diagnostics.append(
                            f"插件 {name} 请求清理失败：{type(exc).__name__}: {exc}"
                        )

    def statuses(self) -> list[dict[str, Any]]:
        return [self._statuses[name].to_dict() for name in sorted(self._statuses)]

    def diagnostics(self) -> list[str]:
        return list(self._diagnostics)

    def doctor(self) -> dict[str, Any]:
        states = self.statuses()
        return {
            "ok": all(item["state"] in {"active", "disabled"} for item in states),
            "plugin_api_version": "1",
            "plugins": states,
            "diagnostics": self.diagnostics(),
        }

    def shutdown(self) -> None:
        if not self._started:
            return
        for name in reversed(self._dependency_order() if self._plugins else []):
            plugin = self._plugins[name]
            callbacks: list[Cleanup] = []
            if plugin.cleanup:
                callbacks.append(plugin.cleanup)
            shutdown = getattr(plugin.instance, "shutdown", None)
            if callable(shutdown):
                callbacks.append(shutdown)
            for callback in callbacks:
                try:
                    callback()
                except Exception as exc:
                    self._diagnostics.append(f"插件 {name} 关闭失败：{type(exc).__name__}: {exc}")
        self._started = False
