"""Quantum backend adapters owned by the teaching domain plugin."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...tools._backend_availability import can_import
from ...tools.registry import Tool
from .openfermion_backend import openfermion_algebra, openfermion_tools
from .pycommute_backend import pycommute_algebra, pycommute_tools
from .qutip_backend import qutip_state_check, qutip_tools


def backend_status(_: Mapping[str, Any]) -> Any:
    return {
        "qutip": can_import("qutip"),
        "pycommute": can_import("pycommute"),
        "openfermion": can_import("openfermion"),
        "note": "三个量子后端来自 qm-teaching 可选依赖；false 表示未安装或无法导入。",
    }


def quantum_tools() -> tuple[Tool, ...]:
    status = Tool(
        "quantum_backend_status",
        "报告 qm-teaching 可选量子后端是否可用。",
        {"type": "object", "properties": {}, "additionalProperties": False},
        backend_status,
        planner_visible=False,
    )
    return (status, *qutip_tools(), *pycommute_tools(), *openfermion_tools())


__all__ = [
    "backend_status",
    "can_import",
    "openfermion_algebra",
    "quantum_tools",
    "pycommute_algebra",
    "qutip_state_check",
]
