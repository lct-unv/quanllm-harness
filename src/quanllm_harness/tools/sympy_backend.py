from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .registry import Tool


def _validate_scalar_arguments(args: Mapping[str, Any]) -> None:
    operator_marks = ("|", "⟩", "⟨", "bra", "ket", "dagger", "†")
    for key in ("expression", "lhs", "rhs", "equation"):
        value = args.get(key)
        if isinstance(value, str) and any(mark in value for mark in operator_marks):
            raise ValueError("检测到态矢或抽象算符记号，不能交给可交换标量后端")


def _sympy():
    try:
        import sympy as sp
    except ImportError as exc:
        raise RuntimeError("SymPy 默认后端缺失，请重新安装 quanllm-harness") from exc
    return sp


def _locals(sp, symbols: list[str] | None = None) -> dict[str, Any]:
    values = {"i": sp.I, "I": sp.I, "pi": sp.pi, "E": sp.E, "oo": sp.oo}
    for name in symbols or ():
        if not name.isidentifier() or name.startswith("_"):
            raise ValueError(f"非法符号名：{name}")
        values[name] = sp.Symbol(name)
    return values


def _parse(expression: Any, sp, symbols: list[str] | None = None):
    if not isinstance(expression, (str, int, float)):
        raise ValueError("表达式必须是字符串或数值")
    text = str(expression)
    if len(text) > 10_000 or "__" in text:
        raise ValueError("表达式过长或包含禁止标识符")
    return sp.sympify(text.replace("^", "**"), locals=_locals(sp, symbols))


def symbolic_calculate(args: Mapping[str, Any]) -> Any:
    sp = _sympy()
    operation = str(args.get("operation", "simplify"))
    symbols = list(args.get("symbols") or [])
    expression = _parse(args.get("expression", ""), sp, symbols)
    variable = str(args.get("variable", "x"))
    variable_symbol = _locals(sp, symbols + [variable])[variable]
    if operation == "simplify":
        result = sp.simplify(expression)
    elif operation == "expand":
        result = sp.expand(expression)
    elif operation == "factor":
        result = sp.factor(expression)
    elif operation == "diff":
        result = sp.diff(expression, variable_symbol)
    elif operation == "integrate":
        bounds = args.get("bounds")
        if bounds is None:
            result = sp.integrate(expression, variable_symbol)
        else:
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValueError("bounds 必须是 [下限, 上限]")
            result = sp.integrate(
                expression,
                (variable_symbol, _parse(bounds[0], sp, symbols), _parse(bounds[1], sp, symbols)),
            )
    elif operation == "limit":
        result = sp.limit(expression, variable_symbol, _parse(args.get("point"), sp, symbols))
    elif operation == "substitute":
        substitutions = args.get("substitutions") or {}
        if not isinstance(substitutions, dict):
            raise ValueError("substitutions 必须是对象")
        result = expression.subs(
            {
                _parse(key, sp, symbols): _parse(value, sp, symbols)
                for key, value in substitutions.items()
            }
        )
        result = sp.simplify(result)
    else:
        raise ValueError(f"不支持的符号操作：{operation}")
    return {
        "expression": str(result),
        "unevaluated": bool(result.has(sp.Integral, sp.Sum, sp.Limit)),
    }


def compare_expressions(args: Mapping[str, Any]) -> Any:
    sp = _sympy()
    symbols = list(args.get("symbols") or [])
    lhs = _parse(args.get("lhs", ""), sp, symbols)
    rhs = _parse(args.get("rhs", ""), sp, symbols)
    difference = sp.simplify(lhs - rhs)
    return {"equivalent": difference == 0, "simplified_difference": str(difference)}


def solve_equation(args: Mapping[str, Any]) -> Any:
    sp = _sympy()
    variable = str(args.get("variable", "x"))
    symbols = list(dict.fromkeys([*(args.get("symbols") or []), variable]))
    x = _locals(sp, symbols)[variable]
    equation = str(args.get("equation", ""))
    if "=" in equation:
        lhs, rhs = equation.split("=", 1)
        expression = _parse(lhs, sp, symbols) - _parse(rhs, sp, symbols)
    else:
        expression = _parse(equation, sp, symbols)
    return {"solutions": [str(value) for value in sp.solve(expression, x)]}


def _parse_matrix(value: Any, sp):
    if not isinstance(value, list) or not value or not all(isinstance(row, list) for row in value):
        raise ValueError("矩阵必须是非空二维数组")
    width = len(value[0])
    if width == 0 or any(len(row) != width for row in value):
        raise ValueError("矩阵各行长度必须一致且非空")
    return sp.Matrix([[_parse(item, sp) for item in row] for row in value])


def _matrix_entries(value) -> list[list[str]]:
    return [[str(value[row, column]) for column in range(value.cols)] for row in range(value.rows)]


def compare_matrices(args: Mapping[str, Any]) -> Any:
    sp = _sympy()
    lhs = _parse_matrix(args.get("lhs"), sp)
    rhs = _parse_matrix(args.get("rhs"), sp)
    normalized = {"lhs": _matrix_entries(lhs), "rhs": _matrix_entries(rhs)}
    if lhs.shape != rhs.shape:
        return {
            "equivalent": False,
            "difference": None,
            "reason": f"shape mismatch: {lhs.shape} != {rhs.shape}",
            "normalized_inputs": normalized,
        }
    difference = lhs.applyfunc(lambda value: sp.simplify(value)) - rhs.applyfunc(
        lambda value: sp.simplify(value)
    )
    difference = difference.applyfunc(lambda value: sp.simplify(value))
    return {
        "equivalent": difference == sp.zeros(*lhs.shape),
        "difference": _matrix_entries(difference),
        "normalized_inputs": normalized,
    }


def matrix_calculate(args: Mapping[str, Any]) -> Any:
    sp = _sympy()

    first = _parse_matrix(args.get("matrix"), sp)
    normalized_inputs: dict[str, Any] = {"matrix": _matrix_entries(first)}
    operation = str(args.get("operation", "eigenvalues"))
    if operation == "eigenvalues":
        return {
            "eigenvalues": {str(key): value for key, value in first.eigenvals().items()},
            "normalized_inputs": normalized_inputs,
        }
    if operation == "eigenvectors":
        return {
            "eigenvectors": [
                {
                    "eigenvalue": str(value),
                    "multiplicity": multiplicity,
                    "vectors": [_matrix_entries(vector) for vector in vectors],
                }
                for value, multiplicity, vectors in first.eigenvects()
            ],
            "normalized_inputs": normalized_inputs,
        }
    if operation == "trace":
        return {"trace": str(sp.simplify(first.trace())), "normalized_inputs": normalized_inputs}
    if operation == "determinant":
        return {
            "determinant": str(sp.simplify(first.det())),
            "normalized_inputs": normalized_inputs,
        }
    if operation in {"commutator", "anticommutator", "multiply"}:
        second = _parse_matrix(args.get("other_matrix"), sp)
        normalized_inputs["other_matrix"] = _matrix_entries(second)
        if operation == "multiply":
            result = first * second
        elif operation == "commutator":
            result = first * second - second * first
        else:
            result = first * second + second * first
        result = result.applyfunc(lambda value: sp.simplify(value))
        return {
            "matrix": _matrix_entries(result),
            "expression": str(result),
            "normalized_inputs": normalized_inputs,
        }
    if operation == "hermitian":
        return {"hermitian": bool(first.is_hermitian), "normalized_inputs": normalized_inputs}
    if operation == "unitary":
        return {
            "unitary": first.rows == first.cols
            and sp.simplify(first.H * first) == sp.eye(first.rows),
            "normalized_inputs": normalized_inputs,
        }
    if operation in {"expectation", "variance"}:
        state = args.get("state")
        if not isinstance(state, list) or not state:
            raise ValueError("expectation/variance 需要非空 state")
        vector = sp.Matrix([_parse(item, sp) for item in state])
        normalized_inputs["state"] = [str(item) for item in vector]
        if first.rows != first.cols or first.cols != vector.rows:
            raise ValueError("算符与态向量维数不匹配")
        norm = sp.simplify((vector.H * vector)[0])
        if norm == 0:
            raise ValueError("态向量范数为零")
        mean = sp.simplify((vector.H * first * vector)[0] / norm)
        if operation == "expectation":
            return {"expectation": str(mean), "normalized_inputs": normalized_inputs}
        second = sp.simplify((vector.H * first**2 * vector)[0] / norm)
        return {
            "variance": str(sp.simplify(second - mean**2)),
            "normalized_inputs": normalized_inputs,
        }
    raise ValueError(f"不支持的矩阵操作：{operation}")


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


def dimension_check(args: Mapping[str, Any]) -> Any:
    sp = _sympy()
    import sympy.physics.units as units
    from sympy.physics.units.systems.si import SI
    from sympy.physics.units.util import check_dimensions

    aliases = {
        "m": units.meter,
        "s": units.second,
        "kg": units.kilogram,
        "K": units.kelvin,
        "Hz": units.hertz,
        "J": units.joule,
        "eV": units.electronvolt,
        "h": units.planck,
        "hbar": units.hbar,
        "c": units.speed_of_light,
        "k_B": units.boltzmann,
    }

    def dependencies(text):
        expr = sp.sympify(str(text).replace("^", "**"), locals=aliases)
        if expr.free_symbols:
            raise ValueError("量纲表达式含未映射符号：" + ", ".join(map(str, expr.free_symbols)))
        check_dimensions(expr)
        dim = SI.get_dimensional_expr(expr)
        return {
            str(key): str(value)
            for key, value in SI.get_dimension_system().get_dimensional_dependencies(dim).items()
        }

    left = dependencies(args.get("expression"))
    target = args.get("target_expression")
    result = {"dimensions": left}
    if target is not None:
        right = dependencies(target)
        result.update({"target_dimensions": right, "equivalent": left == right})
    return result


def boundary_match(args: Mapping[str, Any]) -> Any:
    sp = _sympy()
    variable = str(args.get("variable", "x"))
    symbols = list(dict.fromkeys([*(args.get("symbols") or []), variable]))
    x = _locals(sp, symbols)[variable]
    point = _parse(args.get("point"), sp, symbols)
    left = _parse(args.get("left_expression"), sp, symbols)
    right = _parse(args.get("right_expression"), sp, symbols)
    max_order = int(args.get("derivative_order", 1))
    if max_order < 0 or max_order > 4:
        raise ValueError("derivative_order 必须位于 0 到 4")
    checks = []
    for order in range(max_order + 1):
        lhs = sp.simplify(sp.diff(left, x, order).subs(x, point))
        rhs = sp.simplify(sp.diff(right, x, order).subs(x, point))
        difference = sp.simplify(lhs - rhs)
        checks.append(
            {
                "order": order,
                "left": str(lhs),
                "right": str(rhs),
                "difference": str(difference),
                "matches": difference == 0,
            }
        )
    return {"checks": checks, "all_match": all(item["matches"] for item in checks)}


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


def sympy_tools() -> tuple[Tool, ...]:
    object_schema = {"type": "object"}
    matrix_schema = {
        "type": "array",
        "description": (
            "二维矩阵；符号、表达式和虚数单位必须用字符串逐字保留。"
            '例如虚数单位 I 必须写成字符串 "I"，不得写成数值 1。'
        ),
        "minItems": 1,
        "items": {
            "type": "array",
            "minItems": 1,
            "items": {"type": ["string", "number"]},
        },
    }
    return (
        Tool(
            "symbolic_calculate",
            "通用标量符号化简、展开、因式分解、微积分、极限或代入检查；不得传入矩阵或二维数组。",
            {
                **object_schema,
                "properties": {
                    "operation": {
                        "enum": [
                            "simplify",
                            "expand",
                            "factor",
                            "diff",
                            "integrate",
                            "limit",
                            "substitute",
                        ]
                    },
                    "expression": {"type": "string"},
                    "variable": {"type": "string"},
                    "symbols": {"type": "array", "items": {"type": "string"}},
                    "bounds": {"type": "array"},
                    "point": {},
                    "substitutions": {"type": "object"},
                },
                "required": ["operation", "expression"],
            },
            symbolic_calculate,
            claim_kinds=("equation", "derivation", "condition", "conclusion"),
            argument_validator=_validate_scalar_arguments,
        ),
        Tool(
            "compare_expressions",
            "检查两个标量表达式是否符号等价；矩阵必须使用 compare_matrices。",
            {
                **object_schema,
                "properties": {
                    "lhs": {"type": "string"},
                    "rhs": {"type": "string"},
                    "symbols": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["lhs", "rhs"],
            },
            compare_expressions,
            claim_kinds=("equation", "derivation", "condition", "conclusion"),
            argument_validator=_validate_scalar_arguments,
        ),
        Tool(
            "compare_matrices",
            "逐元素精确检查两个矩阵是否符号等价，支持字符串形式的 I、pi 和符号表达式。",
            {
                **object_schema,
                "properties": {"lhs": matrix_schema, "rhs": matrix_schema},
                "required": ["lhs", "rhs"],
            },
            compare_matrices,
            claim_kinds=("equation", "derivation", "condition", "conclusion"),
        ),
        Tool(
            "solve_equation",
            "求解一个代数方程。",
            {
                **object_schema,
                "properties": {
                    "equation": {"type": "string"},
                    "variable": {"type": "string"},
                    "symbols": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["equation"],
            },
            solve_equation,
            claim_kinds=("equation", "derivation", "condition", "conclusion"),
            argument_validator=_validate_scalar_arguments,
        ),
        Tool(
            "matrix_calculate",
            "精确计算矩阵谱、乘积、对易子、反对易子、迹、行列式及厄米/幺正性；"
            "虚数单位 I 必须以字符串保留，不能改成数值 1。",
            {
                **object_schema,
                "properties": {
                    "matrix": matrix_schema,
                    "other_matrix": matrix_schema,
                    "state": {"type": "array"},
                    "operation": {
                        "enum": [
                            "eigenvalues",
                            "eigenvectors",
                            "trace",
                            "determinant",
                            "multiply",
                            "commutator",
                            "anticommutator",
                            "hermitian",
                            "unitary",
                            "expectation",
                            "variance",
                        ]
                    },
                },
                "required": ["matrix", "operation"],
            },
            matrix_calculate,
            claim_kinds=("equation", "derivation", "condition", "conclusion"),
        ),
        Tool(
            "angular_momentum",
            "精确计算 Clebsch-Gordan、Wigner 3j 或 Wigner 6j 系数。",
            {
                **object_schema,
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
            "dimension_check",
            "使用 SI 单位检查表达式量纲，并可与目标表达式比较。",
            {
                **object_schema,
                "properties": {
                    "expression": {"type": "string"},
                    "target_expression": {"type": "string"},
                },
                "required": ["expression"],
            },
            dimension_check,
            claim_kinds=("equation", "derivation", "condition", "conclusion"),
            argument_validator=_validate_scalar_arguments,
        ),
        Tool(
            "boundary_match",
            "精确检查两个分段表达式在指定边界处的函数值及若干阶导数是否连续。",
            {
                **object_schema,
                "properties": {
                    "left_expression": {"type": "string"},
                    "right_expression": {"type": "string"},
                    "variable": {"type": "string"},
                    "point": {},
                    "derivative_order": {"type": "integer", "minimum": 0, "maximum": 4},
                    "symbols": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["left_expression", "right_expression", "point"],
            },
            boundary_match,
            claim_kinds=("equation", "derivation", "condition"),
        ),
        Tool(
            "fock_ladder_expectation",
            "精确计算数态中的 <n|(a+a†)^p|n>，用于谐振子微扰与算符展开核验。",
            {
                **object_schema,
                "properties": {"power": {"type": "integer", "minimum": 0, "maximum": 16}},
                "required": ["power"],
            },
            fock_ladder_expectation,
            claim_kinds=("equation", "derivation", "conclusion"),
        ),
    )
