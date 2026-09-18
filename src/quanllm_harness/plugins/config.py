from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from .api import SAFE_PLUGIN_PERMISSIONS, PluginPolicy


def default_plugin_config_path() -> Path:
    override = os.environ.get("QUANLLM_PLUGIN_CONFIG", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "quanllm-harness" / "plugins.json"


def load_plugin_policy(path: str | Path | None = None) -> PluginPolicy:
    config_path = Path(path).expanduser() if path else default_plugin_config_path()
    data: dict[str, Any] = {}
    if config_path.is_file():
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("插件配置根节点必须是 JSON 对象")
        data = loaded
    environment_enabled = tuple(
        item.strip()
        for item in os.environ.get("QUANLLM_PLUGINS_ENABLED", "").split(",")
        if item.strip()
    )
    environment_disabled = tuple(
        item.strip()
        for item in os.environ.get("QUANLLM_PLUGINS_DISABLED", "").split(",")
        if item.strip()
    )
    policy = PluginPolicy(
        enabled=environment_enabled or tuple(data.get("enabled") or ()),
        disabled=environment_disabled or tuple(data.get("disabled") or ()),
        allow_unlisted=bool(data.get("allow_unlisted", False)),
        allowed_permissions=tuple(data.get("allowed_permissions") or SAFE_PLUGIN_PERMISSIONS),
        trusted_digests=dict(data.get("trusted_digests") or {}),
        require_trusted_digest=bool(data.get("require_trusted_digest", False)),
        call_timeout_seconds=float(data.get("call_timeout_seconds", 30.0)),
        max_output_bytes=int(data.get("max_output_bytes", 1_048_576)),
        config=dict(data.get("config") or {}),
        config_path=config_path,
    )
    policy.validate()
    return policy


def save_plugin_policy(policy: PluginPolicy, path: str | Path | None = None) -> Path:
    config_path = (
        Path(path).expanduser() if path else policy.config_path or default_plugin_config_path()
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "enabled": list(policy.enabled),
        "disabled": list(policy.disabled),
        "allow_unlisted": policy.allow_unlisted,
        "allowed_permissions": list(policy.allowed_permissions),
        "trusted_digests": dict(policy.trusted_digests),
        "require_trusted_digest": policy.require_trusted_digest,
        "call_timeout_seconds": policy.call_timeout_seconds,
        "max_output_bytes": policy.max_output_bytes,
        "config": {name: dict(value) for name, value in policy.config.items()},
    }
    with NamedTemporaryFile("w", encoding="utf-8", dir=config_path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(config_path)
    return config_path


def set_plugin_enabled(name: str, enabled: bool, path: str | Path | None = None) -> Path:
    if not name or any(character.isspace() for character in name):
        raise ValueError("插件名称不能为空或包含空白")
    policy = load_plugin_policy(path)
    enabled_names = set(policy.enabled)
    disabled_names = set(policy.disabled)
    if enabled:
        enabled_names.add(name)
        disabled_names.discard(name)
    else:
        disabled_names.add(name)
        enabled_names.discard(name)
    updated = PluginPolicy(
        enabled=tuple(sorted(enabled_names)),
        disabled=tuple(sorted(disabled_names)),
        allow_unlisted=policy.allow_unlisted,
        allowed_permissions=policy.allowed_permissions,
        trusted_digests=policy.trusted_digests,
        require_trusted_digest=policy.require_trusted_digest,
        call_timeout_seconds=policy.call_timeout_seconds,
        max_output_bytes=policy.max_output_bytes,
        config=policy.config,
        config_path=policy.config_path,
    )
    return save_plugin_policy(updated, path)
