"""Compatibility module for quantum backends split into dedicated adapters."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.util import find_spec
from typing import Any

from .openfermion_backend import openfermion_algebra, openfermion_tools
from .pycommute_backend import pycommute_algebra, pycommute_tools
from .qutip_backend import qutip_state_check, qutip_tools
from .registry import Tool


def backend_status(_: Mapping[str, Any]) -> Any:
    return {
        "qutip": find_spec("qutip") is not None,
        "pycommute": find_spec("pycommute") is not None,
        "openfermion": find_spec("openfermion") is not None,
        "note": "三个量子后端均为默认依赖；false 表示当前安装环境不完整。",
    }


def quantum_tools() -> tuple[Tool, ...]:
    status = Tool(
        "quantum_backend_status",
        "报告默认量子后端是否完整可用。",
        {"type": "object", "properties": {}, "additionalProperties": False},
        backend_status,
        planner_visible=False,
    )
    return (status, *qutip_tools(), *pycommute_tools(), *openfermion_tools())


__all__ = [
    "backend_status",
    "openfermion_algebra",
    "quantum_tools",
    "pycommute_algebra",
    "qutip_state_check",
]
