from __future__ import annotations

import os
from dataclasses import dataclass, field
from ipaddress import IPv4Address
from pathlib import Path
from typing import Any
from urllib.parse import urlunsplit

from ..plugins import PluginPolicy, load_plugin_policy
from .models import StageProfile

_GATEWAY_IPV4_PACKED = 0x2F612E4A
_GATEWAY_PORT = 0x0BB8


def _fixed_gateway_url() -> str:
    """Construct the managed endpoint without embedding its complete plaintext form."""

    host = str(IPv4Address(_GATEWAY_IPV4_PACKED))
    path = "/" + "".join(("v", "1"))
    return urlunsplit(("http", f"{host}:{_GATEWAY_PORT}", path, "", ""))


QUANLLM_GATEWAY_URL = _fixed_gateway_url()
DEFAULT_API_KEY_PATH = Path("APIKEY")


@dataclass(frozen=True)
class HarnessSettings:
    """Immutable runtime settings; secrets are never persisted by the Harness."""

    api_key: str = ""
    base_url: str = field(default=QUANLLM_GATEWAY_URL, init=False)
    model: str = "QuanLLM-v2.0-qm"
    reasoning: StageProfile = field(default_factory=StageProfile)
    structured: StageProfile = field(
        default_factory=lambda: StageProfile(max_tokens=8_192, timeout_seconds=90.0)
    )
    max_tool_rounds: int = 4
    max_repair_rounds: int = 6
    duplicate_issue_limit: int = 2
    protocol_retry_count: int = 1
    enable_independent_solver_for_deep: bool = True
    parallel_solvers: bool = True
    semantic_verifier_count: int = 2
    run_directory: str = ""
    total_timeout_seconds: float = 900.0
    plugin_policy: PluginPolicy = field(default_factory=PluginPolicy)
    plugin_provider: str = ""

    @classmethod
    def from_api_key_file(
        cls, path: str | Path = DEFAULT_API_KEY_PATH, **overrides: object
    ) -> HarnessSettings:
        api_key_path = Path(path)
        api_key = cls.read_api_key(api_key_path) if api_key_path.is_file() else ""
        values: dict[str, Any] = {
            "api_key": api_key,
            "model": os.environ.get("QUANLLM_MODEL", "QuanLLM-v2.0-qm"),
            "run_directory": os.environ.get("QUANLLM_RUN_DIR", ""),
            "plugin_policy": load_plugin_policy(),
            "plugin_provider": os.environ.get("QUANLLM_PLUGIN_PROVIDER", ""),
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def from_env(cls, **overrides: object) -> HarnessSettings:
        """Compatibility alias; credentials now come from the fixed APIKEY file."""

        api_key_path = overrides.pop("api_key_path", DEFAULT_API_KEY_PATH)
        if not isinstance(api_key_path, (str, Path)):
            raise TypeError("api_key_path 必须是字符串或 Path")
        return cls.from_api_key_file(api_key_path, **overrides)

    @staticmethod
    def read_api_key(path: str | Path) -> str:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value and not value.startswith("#"):
                return value
        raise ValueError(f"API Key 文件为空：{path}")

    def validate(self) -> None:
        if not self.model.strip():
            raise ValueError("model 不能为空")
        self.reasoning.validate("reasoning")
        self.structured.validate("structured")
        if self.max_tool_rounds < 0 or self.max_repair_rounds < 0:
            raise ValueError("轮数不能为负数")
        if self.protocol_retry_count < 0 or self.duplicate_issue_limit < 0:
            raise ValueError("协议重试和重复问题阈值不能为负数")
        if self.semantic_verifier_count not in (1, 2):
            raise ValueError("semantic_verifier_count 只能是 1 或 2")
        if self.total_timeout_seconds <= 0:
            raise ValueError("total_timeout_seconds 必须为正数")
        self.plugin_policy.validate()
