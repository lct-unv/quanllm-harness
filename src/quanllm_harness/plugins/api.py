from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from ..contracts import Claim, Evidence, HarnessEvent, Issue, Requirement

if TYPE_CHECKING:
    from ..config import HarnessSettings
    from ..providers import QuanLLMProvider
    from ..tools import Tool

PLUGIN_API_VERSION = "1"

SAFE_PLUGIN_PERMISSIONS = (
    "events.subscribe",
    "services.consume",
    "services.provide",
    "tools.register",
    "verifiers.register",
)


@dataclass(frozen=True)
class PluginManifest:
    """Stable metadata contract implemented inside the main distribution."""

    name: str
    version: str
    api_version: str = PLUGIN_API_VERSION
    requires_harness: str = ">=0.1.2"
    description: str = ""
    permissions: Sequence[str] = ()
    dependencies: Sequence[str] = ()
    execution_mode: str = "in_process"

    def validate(self) -> None:
        if not self.name or any(character.isspace() for character in self.name):
            raise ValueError("插件名称不能为空或包含空白")
        if not self.version:
            raise ValueError(f"插件 {self.name} 缺少版本")
        if self.api_version != PLUGIN_API_VERSION:
            raise ValueError(
                f"插件 {self.name} API {self.api_version!r} 与主程序 API "
                f"{PLUGIN_API_VERSION!r} 不兼容"
            )
        if self.execution_mode not in {"in_process", "subprocess"}:
            raise ValueError("execution_mode 只能是 in_process 或 subprocess")
        duplicates = [item for item in set(self.dependencies) if self.dependencies.count(item) > 1]
        if duplicates:
            raise ValueError(f"插件 {self.name} 重复声明依赖：{sorted(duplicates)}")
        if self.name in self.dependencies:
            raise ValueError(f"插件 {self.name} 不能依赖自身")


@dataclass(frozen=True)
class PluginPolicy:
    """Host-side plugin policy. Unknown permissions are denied by default."""

    enabled: Sequence[str] = ()
    disabled: Sequence[str] = ()
    allow_unlisted: bool = False
    allowed_permissions: Sequence[str] = SAFE_PLUGIN_PERMISSIONS
    trusted_digests: Mapping[str, str] = field(default_factory=dict)
    require_trusted_digest: bool = False
    call_timeout_seconds: float = 30.0
    max_output_bytes: int = 1_048_576
    config: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    config_path: Path | None = None

    def validate(self) -> None:
        if self.call_timeout_seconds <= 0:
            raise ValueError("插件调用超时必须为正数")
        if self.max_output_bytes <= 0:
            raise ValueError("插件输出上限必须为正数")
        overlap = set(self.enabled).intersection(self.disabled)
        if overlap:
            raise ValueError(f"插件不能同时启用和禁用：{sorted(overlap)}")


@dataclass(frozen=True)
class SubprocessToolSpec:
    name: str
    description: str
    parameters: Mapping[str, Any]
    command: Sequence[str]
    limitations: Sequence[str] = ()
    claim_kinds: Sequence[str] = ()
    planner_visible: bool = True
    timeout_seconds: float | None = None

    def validate(self) -> None:
        if not self.name or not self.command:
            raise ValueError("子进程工具必须声明 name 和 command")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("子进程工具 parameters 必须是 JSON Schema 对象")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("子进程工具 timeout_seconds 必须为正数")


@dataclass(frozen=True)
class PluginVerificationContext:
    question: str
    candidate: str
    claims: Sequence[Claim]
    requirements: Sequence[Requirement]
    evidence: Sequence[Evidence]


@dataclass(frozen=True)
class PluginVerificationResult:
    issues: Sequence[Issue] = ()
    warnings: Sequence[str] = ()
    summary: str = ""


PluginVerifier = Callable[[PluginVerificationContext], PluginVerificationResult]
ProviderFactory = Callable[["HarnessSettings"], "QuanLLMProvider"]
EventSubscriber = Callable[[HarnessEvent], None]
Cleanup = Callable[[], None]


class HarnessPlugin(Protocol):
    manifest: PluginManifest

    def setup(self, context: PluginContext) -> Cleanup | None: ...


class _ToolRegistrar(Protocol):
    def register(self, tool: Tool) -> None: ...


class _EventRegistrar(Protocol):
    def subscribe(self, callback: EventSubscriber) -> None: ...


class _VerifierRegistrar(Protocol):
    def register(self, name: str, verifier: PluginVerifier) -> None: ...


class _ProviderRegistrar(Protocol):
    def register(self, name: str, factory: ProviderFactory) -> None: ...


class _ServiceRegistrar(Protocol):
    def provide(self, name: str, value: Any) -> None: ...

    def get(self, name: str) -> Any: ...


@dataclass(frozen=True)
class PluginContext:
    manifest: PluginManifest
    config: Mapping[str, Any]
    tools: _ToolRegistrar
    events: _EventRegistrar
    verifiers: _VerifierRegistrar
    providers: _ProviderRegistrar
    services: _ServiceRegistrar


@dataclass(frozen=True)
class PluginStatus:
    name: str
    version: str = ""
    state: str = "discovered"
    reason: str = ""
    permissions: Sequence[str] = ()
    dependencies: Sequence[str] = ()
    execution_mode: str = ""
    digest: str = ""
    entry_point: str = ""
    tools: Sequence[str] = ()
    verifiers: Sequence[str] = ()
    providers: Sequence[str] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "state": self.state,
            "reason": self.reason,
            "permissions": list(self.permissions),
            "dependencies": list(self.dependencies),
            "execution_mode": self.execution_mode,
            "digest": self.digest,
            "entry_point": self.entry_point,
            "tools": list(self.tools),
            "verifiers": list(self.verifiers),
            "providers": list(self.providers),
        }
