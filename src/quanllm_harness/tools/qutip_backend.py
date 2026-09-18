from __future__ import annotations

from collections.abc import Mapping
from importlib.util import find_spec
from typing import Any

from ._backend_availability import can_import
from .registry import Tool


def qutip_state_check(args: Mapping[str, Any]) -> Any:
    try:
        import numpy as np
        import qutip
    except ImportError as exc:
        if find_spec("numpy") is None or find_spec("qutip") is None:
            raise RuntimeError("QuTiP 默认后端未安装，请重新安装 quanllm-harness") from exc
        raise RuntimeError(
            "QuTiP（或 numpy）已安装但无法导入，可能底层 DLL 加载失败；请修复安装后重试"
        ) from exc
    operation = str(args.get("operation", "normalize"))
    state = np.asarray(args.get("state"), dtype=complex)
    ket = qutip.Qobj(state.reshape((-1, 1)))
    if operation == "normalize":
        norm = float(ket.norm())
        return {"norm": norm, "normalized": bool(abs(norm - 1) < 1e-10)}
    operator = qutip.Qobj(np.asarray(args.get("operator"), dtype=complex))
    if operator.shape[0] != ket.shape[0] or operator.shape[0] != operator.shape[1]:
        raise ValueError("operator 必须是与 state 维数一致的方阵")
    if operation == "expectation":
        return {"expectation": repr(complex(qutip.expect(operator, ket)))}
    if operation == "variance":
        return {"variance": repr(complex(qutip.variance(operator, ket)))}
    raise ValueError("未知 QuTiP 操作")


def qutip_tools() -> tuple[Tool, ...]:
    if not can_import("qutip"):
        return ()
    return (
        Tool(
            "qutip_state_check",
            "使用 QuTiP 数值核验有限维态的归一化、期望值或方差。",
            {
                "type": "object",
                "properties": {
                    "operation": {"enum": ["normalize", "expectation", "variance"]},
                    "state": {"type": "array", "minItems": 1},
                    "operator": {"type": "array"},
                },
                "required": ["operation", "state"],
                "additionalProperties": False,
            },
            qutip_state_check,
            limitations=("有限维数值表示不能单独证明无限维算符恒等式",),
            claim_kinds=("equation", "derivation", "condition", "conclusion"),
        ),
    )
