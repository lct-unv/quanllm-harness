from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...tools.registry import Tool
from ...tools.sympy_backend import (
    _clean_symbol,
    _locals,
    _parse,
    _sympy,
    _validate_scalar_arguments,
    matrix_eigenpair_check,
)


def derive_boundary_equation(args: Mapping[str, Any]) -> Any:
    sp = _sympy()
    symbols = list(args.get("symbols") or [])
    variable = str(args.get("variable", "x"))
    left = _parse(args.get("left_expression"), sp, symbols)
    right = _parse(args.get("right_expression"), sp, symbols)
    point = _parse(args.get("point"), sp, symbols)
    x = _locals(sp, symbols + [variable])[_clean_symbol(variable)]

    def evaluate(expr):
        return sp.simplify(expr.subs({x: point}))

    left_value = evaluate(left)
    right_value = evaluate(right)
    left_derivative = evaluate(sp.diff(left, x))
    right_derivative = evaluate(sp.diff(right, x))
    if left_value == 0 or right_value == 0:
        raise ValueError("边界处波函数为零，无法用 log-derivative 匹配（请改用直接连续性方程）")
    lhs_log = sp.simplify(left_derivative / left_value)
    rhs_log = sp.simplify(right_derivative / right_value)
    difference = sp.simplify(lhs_log - rhs_log)
    return {
        "left_value": str(left_value),
        "right_value": str(right_value),
        "left_log_derivative": str(lhs_log),
        "right_log_derivative": str(rhs_log),
        "matching_difference": str(difference),
        "derived_matching_equation": str(sp.Eq(difference, 0)),
        "matched": difference == 0,
    }


def angular_momentum(args: Mapping[str, Any]) -> Any:
    sp = _sympy()
    from sympy.physics.wigner import clebsch_gordan, wigner_3j, wigner_6j

    operation = str(args.get("operation", ""))
    functions = {
        "clebsch_gordan": clebsch_gordan,
        "wigner_3j": wigner_3j,
        "wigner_6j": wigner_6j,
    }
    if operation not in functions:
        raise ValueError("未知角动量操作")
    values = [_parse(item, sp) for item in args.get("values") or []]
    if len(values) != 6:
        raise ValueError("角动量操作需要 6 个参数")
    return {"value": str(sp.simplify(functions[operation](*values)))}


def fock_ladder_expectation(args: Mapping[str, Any]) -> Any:
    """精确计算 <n|(a+a†)^p|n>，不采用有限维截断。"""

    sp = _sympy()
    power = int(args.get("power", 0))
    if power < 0 or power > 16:
        raise ValueError("power 必须位于 0 到 16")
    n = sp.Symbol("n", integer=True, nonnegative=True)
    amplitudes: dict[int, Any] = {0: sp.Integer(1)}
    for _ in range(power):
        updated: dict[int, Any] = {}
        for offset, coefficient in amplitudes.items():
            occupation = n + offset
            down = offset - 1
            up = offset + 1
            updated[down] = updated.get(down, 0) + coefficient * sp.sqrt(occupation)
            updated[up] = updated.get(up, 0) + coefficient * sp.sqrt(occupation + 1)
        amplitudes = updated
    value = sp.expand(sp.simplify(amplitudes.get(0, 0)))
    return {"expectation": str(value), "power": power, "state": "|n>"}


def qm_teaching_sympy_tools() -> tuple[Tool, ...]:
    matrix_schema = {
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "array",
            "minItems": 1,
            "items": {"type": ["string", "number"]},
        },
    }
    return (
        Tool(
            "derive_boundary_equation",
            "从两段分段波函数在边界点做 log-derivative 匹配并推导匹配方程。",
            {
                "type": "object",
                "properties": {
                    "left_expression": {"type": "string"},
                    "right_expression": {"type": "string"},
                    "variable": {"type": "string"},
                    "point": {},
                    "symbols": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["left_expression", "right_expression", "point"],
            },
            derive_boundary_equation,
            claim_kinds=("definition", "equation", "derivation", "condition", "conclusion"),
            argument_validator=_validate_scalar_arguments,
        ),
        Tool(
            "matrix_eigenpair_check",
            "确定性验证本征对 M·v=λ·v 及向量归一化。",
            {
                "type": "object",
                "properties": {
                    "matrix": matrix_schema,
                    "eigenvalue": {"type": "string"},
                    "eigenvector": {"type": "array", "items": {"type": "string"}},
                    "symbols": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["matrix", "eigenvalue", "eigenvector"],
            },
            matrix_eigenpair_check,
            claim_kinds=("definition", "equation", "derivation", "condition", "conclusion"),
            argument_validator=_validate_scalar_arguments,
        ),
        Tool(
            "angular_momentum",
            "精确计算 Clebsch-Gordan、Wigner 3j 或 Wigner 6j 系数。",
            {
                "type": "object",
                "properties": {
                    "operation": {"enum": ["clebsch_gordan", "wigner_3j", "wigner_6j"]},
                    "values": {"type": "array", "minItems": 6, "maxItems": 6},
                },
                "required": ["operation", "values"],
            },
            angular_momentum,
            claim_kinds=("equation", "derivation", "conclusion"),
        ),
        Tool(
            "fock_ladder_expectation",
            "精确计算数态中的 <n|(a+a†)^p|n>。",
            {
                "type": "object",
                "properties": {"power": {"type": "integer", "minimum": 0, "maximum": 16}},
                "required": ["power"],
            },
            fock_ladder_expectation,
            claim_kinds=("equation", "derivation", "conclusion"),
        ),
    )


__all__ = [
    "angular_momentum",
    "derive_boundary_equation",
    "fock_ladder_expectation",
    "qm_teaching_sympy_tools",
]
