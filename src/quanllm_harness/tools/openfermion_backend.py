from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._backend_availability import can_import
from .registry import Tool


def openfermion_algebra(args: Mapping[str, Any]) -> Any:
    try:
        from openfermion import BosonOperator, FermionOperator, hermitian_conjugated, normal_ordered
    except ImportError as exc:
        if not can_import("openfermion"):
            raise RuntimeError("OpenFermion 默认后端未安装，请重新安装 quanllm-harness") from exc
        raise RuntimeError(
            "OpenFermion 已安装但无法导入，可能底层 DLL 加载失败；请修复安装后重试"
        ) from exc
    cls = FermionOperator if str(args.get("statistics", "fermion")) == "fermion" else BosonOperator

    def expression(terms: Any) -> Any:
        if not isinstance(terms, list) or not terms:
            raise ValueError("terms 必须是非空数组")
        result = cls()
        for item in terms:
            if not isinstance(item, dict):
                raise ValueError("每个 term 必须是对象")
            parsed = []
            for action in item.get("actions") or []:
                if not isinstance(action, list) or len(action) != 2:
                    raise ValueError("action 必须是 [mode, 0或1]")
                parsed.append((int(action[0]), int(action[1])))
            result += cls(tuple(parsed), complex(item.get("coefficient", 1)))
        return result

    left = expression(args.get("left"))
    operation = str(args.get("operation", "normal_order"))
    if operation == "normal_order":
        result = normal_ordered(left)
    elif operation == "adjoint":
        result = hermitian_conjugated(left)
    else:
        right = expression(args.get("right"))
        if operation == "commutator":
            result = normal_ordered(left * right - right * left)
        elif operation == "anticommutator":
            result = normal_ordered(left * right + right * left)
        else:
            raise ValueError("未知 OpenFermion 操作")
    return {"expression": str(result), "zero": not bool(result.terms)}


def openfermion_tools() -> tuple[Tool, ...]:
    if not can_import("openfermion"):
        return ()
    term_schema = {
        "type": "object",
        "properties": {
            "coefficient": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            "actions": {
                "type": "array",
                "items": {
                    "type": "array",
                    "prefixItems": [{"type": "integer"}, {"enum": [0, 1]}],
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
        },
        "required": ["actions"],
        "additionalProperties": False,
    }
    return (
        Tool(
            "openfermion_algebra",
            "使用 OpenFermion 对费米或玻色算符做正规排序及代数运算。",
            {
                "type": "object",
                "properties": {
                    "statistics": {"enum": ["fermion", "boson"]},
                    "operation": {
                        "enum": ["normal_order", "adjoint", "commutator", "anticommutator"]
                    },
                    "left": {"type": "array", "items": term_schema, "minItems": 1},
                    "right": {"type": "array", "items": term_schema, "minItems": 1},
                },
                "required": ["statistics", "operation", "left"],
                "additionalProperties": False,
            },
            openfermion_algebra,
            claim_kinds=("equation", "derivation", "condition", "conclusion"),
        ),
    )
