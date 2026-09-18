from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._backend_availability import can_import
from .registry import Tool


def pycommute_algebra(args: Mapping[str, Any]) -> Any:
    try:
        from pycommute.expression import S_m, S_p, S_x, S_y, S_z, a, a_dag, c, c_dag, conj, n
    except ImportError as exc:
        if not can_import("pycommute"):
            raise RuntimeError("pycommute 默认后端未安装，请重新安装 quanllm-harness") from exc
        raise RuntimeError(
            "pycommute 已安装但无法导入，可能底层 DLL 加载失败；请修复安装后重试"
        ) from exc
    factories = {
        "fermion_create": c_dag,
        "fermion_annihilate": c,
        "fermion_number": n,
        "boson_create": a_dag,
        "boson_annihilate": a,
        "spin_plus": S_p,
        "spin_minus": S_m,
        "spin_x": S_x,
        "spin_y": S_y,
        "spin_z": S_z,
    }

    def monomial(items: Any) -> Any:
        if not isinstance(items, list) or not items:
            raise ValueError("算符单项式必须是非空数组")
        result = None
        for item in items:
            if not isinstance(item, dict) or item.get("operator") not in factories:
                raise ValueError("未知或非法算符描述")
            indices = item.get("indices") or []
            if not isinstance(indices, list) or not all(isinstance(v, (int, str)) for v in indices):
                raise ValueError("indices 只能包含整数或字符串")
            kwargs = {"spin": float(item["spin"])} if "spin" in item else {}
            factor = factories[item["operator"]](*indices, **kwargs)
            result = factor if result is None else result * factor
        return result

    left = monomial(args.get("left"))
    operation = str(args.get("operation", "commutator"))
    if operation == "adjoint":
        result = conj(left)
    else:
        right = monomial(args.get("right"))
        if operation == "commutator":
            result = left * right - right * left
        elif operation == "anticommutator":
            result = left * right + right * left
        elif operation == "product":
            result = left * right
        else:
            raise ValueError("未知 pycommute 操作")
    return {"expression": str(result), "zero": not bool(result)}


def pycommute_tools() -> tuple[Tool, ...]:
    if not can_import("pycommute"):
        return ()
    operator_schema = {
        "type": "object",
        "properties": {
            "operator": {
                "enum": [
                    "fermion_create",
                    "fermion_annihilate",
                    "fermion_number",
                    "boson_create",
                    "boson_annihilate",
                    "spin_plus",
                    "spin_minus",
                    "spin_x",
                    "spin_y",
                    "spin_z",
                ]
            },
            "indices": {
                "type": "array",
                "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
            },
            "spin": {"type": "number"},
        },
        "required": ["operator"],
        "additionalProperties": False,
    }
    return (
        Tool(
            "pycommute_algebra",
            "使用 pycommute 精确化简玻色、费米或自旋算符代数。",
            {
                "type": "object",
                "properties": {
                    "operation": {"enum": ["commutator", "anticommutator", "product", "adjoint"]},
                    "left": {"type": "array", "items": operator_schema, "minItems": 1},
                    "right": {"type": "array", "items": operator_schema, "minItems": 1},
                },
                "required": ["operation", "left"],
                "additionalProperties": False,
            },
            pycommute_algebra,
            claim_kinds=("equation", "derivation", "condition", "conclusion"),
        ),
    )
