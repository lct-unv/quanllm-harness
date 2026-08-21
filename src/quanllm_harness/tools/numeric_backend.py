from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .registry import Tool


def _context(args: Mapping[str, Any]):
    import mpmath as mp
    import sympy as sp

    dps = int(args.get("dps", 50))
    if not 15 <= dps <= 200:
        raise ValueError("dps 必须位于 15 到 200")
    mp.mp.dps = dps
    variable = str(args.get("variable", "x"))
    symbol = sp.Symbol(variable)
    values = args.get("parameters") or {}
    if not isinstance(values, Mapping):
        raise ValueError("parameters 必须是对象")
    local_symbols = {variable: symbol}
    local_symbols.update({str(name): sp.Symbol(str(name)) for name in values})
    expression = sp.sympify(str(args.get("expression", "")), locals=local_symbols)
    expression = expression.subs(
        {local_symbols[str(name)]: value for name, value in values.items()}
    )
    function = sp.lambdify(symbol, expression, modules="mpmath")
    return mp, expression, function


def numeric_integrate(args: Mapping[str, Any]) -> Any:
    mp, expression, function = _context(args)
    lower = mp.mpf(str(args["lower"])) if str(args["lower"]) not in {"-inf", "-oo"} else -mp.inf
    upper = (
        mp.mpf(str(args["upper"])) if str(args["upper"]) not in {"inf", "+inf", "oo"} else mp.inf
    )
    result = mp.quad(function, [lower, upper])
    return {"expression": str(expression), "value": mp.nstr(result, mp.mp.dps), "dps": mp.mp.dps}


def numeric_root(args: Mapping[str, Any]) -> Any:
    mp, expression, function = _context(args)
    initial = args.get("initial")
    if not isinstance(initial, list) or not 1 <= len(initial) <= 2:
        raise ValueError("initial 必须包含一个或两个初值")
    points = tuple(mp.mpf(str(value)) for value in initial)
    root = mp.findroot(function, points[0] if len(points) == 1 else points)
    residual = function(root)
    return {
        "expression": str(expression),
        "root": mp.nstr(root, mp.mp.dps),
        "residual": mp.nstr(residual, mp.mp.dps),
        "dps": mp.mp.dps,
    }


def truncation_convergence(args: Mapping[str, Any]) -> Any:
    values = args.get("values")
    if not isinstance(values, list) or len(values) < 3:
        raise ValueError("values 至少需要三个连续截断结果")
    numbers = [complex(value) for value in values]
    absolute_tolerance = float(args.get("absolute_tolerance", 1e-10))
    relative_tolerance = float(args.get("relative_tolerance", 1e-8))
    delta = abs(numbers[-1] - numbers[-2])
    scale = max(abs(numbers[-1]), abs(numbers[-2]), 1.0)
    return {
        "last_value": repr(numbers[-1]),
        "absolute_change": delta,
        "relative_change": delta / scale,
        "converged": delta <= absolute_tolerance + relative_tolerance * scale,
        "sample_count": len(numbers),
    }


def numeric_tools() -> tuple[Tool, ...]:
    common = {
        "expression": {"type": "string", "minLength": 1},
        "variable": {"type": "string", "minLength": 1},
        "parameters": {"type": "object", "additionalProperties": {"type": "number"}},
        "dps": {"type": "integer", "minimum": 15, "maximum": 200},
    }
    kinds = ("equation", "derivation", "condition", "conclusion")
    return (
        Tool(
            "numeric_integrate",
            "以可控精度计算定积分或无穷区间积分。",
            {
                "type": "object",
                "properties": {
                    **common,
                    "lower": {"anyOf": [{"type": "number"}, {"type": "string"}]},
                    "upper": {"anyOf": [{"type": "number"}, {"type": "string"}]},
                },
                "required": ["expression", "variable", "lower", "upper"],
                "additionalProperties": False,
            },
            numeric_integrate,
            limitations=("数值积分不是解析证明；奇点附近需结合误差和分段检查",),
            claim_kinds=kinds,
        ),
        Tool(
            "numeric_root",
            "以可控精度求单变量方程的局部数值根并返回残差。",
            {
                "type": "object",
                "properties": {
                    **common,
                    "initial": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 1,
                        "maxItems": 2,
                    },
                },
                "required": ["expression", "variable", "initial"],
                "additionalProperties": False,
            },
            numeric_root,
            limitations=("局部求根不能证明已找到全部根",),
            claim_kinds=kinds,
        ),
        Tool(
            "truncation_convergence",
            "检查连续有限维或截断结果是否达到给定绝对/相对容差。",
            {
                "type": "object",
                "properties": {
                    "values": {
                        "type": "array",
                        "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
                        "minItems": 3,
                    },
                    "absolute_tolerance": {"type": "number", "minimum": 0},
                    "relative_tolerance": {"type": "number", "minimum": 0},
                },
                "required": ["values"],
                "additionalProperties": False,
            },
            truncation_convergence,
            limitations=("仅衡量给定序列的末步变化，不替代理论误差界",),
            claim_kinds=kinds,
        ),
    )
