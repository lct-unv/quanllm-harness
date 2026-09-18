from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .registry import Tool

ALGEBRA_OPERATORS = {
    "canonical": frozenset({"x", "p"}),
    "angular_momentum": frozenset({"Jx", "Jy", "Jz", "Jplus", "Jminus", "Lx", "Ly", "Lz"}),
    "boson_single_mode": frozenset({"a", "adag"}),
    "fermion_single_mode": frozenset({"c", "cdag"}),
}


def _sympy_quantum():
    try:
        import sympy as sp
        from sympy.physics.quantum import AntiCommutator, Commutator, Dagger
        from sympy.physics.quantum.boson import BosonOp
        from sympy.physics.quantum.cartesian import PxOp, XOp
        from sympy.physics.quantum.constants import hbar
        from sympy.physics.quantum.fermion import FermionOp
        from sympy.physics.quantum.spin import Jminus, Jplus, Jx, Jy, Jz
    except ImportError as exc:
        raise RuntimeError("SymPy 默认后端缺失，请重新安装 quanllm-harness") from exc
    boson = BosonOp("a")
    fermion = FermionOp("c")
    return {
        "sp": sp,
        "AntiCommutator": AntiCommutator,
        "Commutator": Commutator,
        "Dagger": Dagger,
        "hbar": hbar,
        "operators": {
            "x": XOp(),
            "p": PxOp(),
            "Jx": Jx,
            "Jy": Jy,
            "Jz": Jz,
            "Jplus": Jplus,
            "Jminus": Jminus,
            "Lx": Jx,
            "Ly": Jy,
            "Lz": Jz,
            "a": boson,
            "adag": Dagger(boson),
            "c": fermion,
            "cdag": Dagger(fermion),
        },
    }


def _coefficient(value: Any, sp, scalar_symbols: list[str], hbar):
    if not isinstance(value, (str, int, float)):
        raise ValueError("算符项系数必须是字符串或数值")
    locals_: dict[str, Any] = {
        "I": sp.I,
        "i": sp.I,
        "pi": sp.pi,
        "hbar": hbar,
        "ℏ": hbar,
        "ħ": hbar,
    }
    for name in scalar_symbols:
        if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
            raise ValueError("scalar_symbols 包含非法符号名")
        if name not in locals_:
            locals_[name] = sp.Symbol(name)
    text = str(value)
    if len(text) > 2_000 or "__" in text:
        raise ValueError("算符项系数过长或包含禁止标识符")
    result = sp.sympify(text.replace("^", "**"), locals=locals_)
    if result.is_commutative is not True:
        raise ValueError("算符项系数必须是可交换标量")
    return result


def _expression(
    terms: Any,
    *,
    algebra: str,
    quantum: Mapping[str, Any],
    scalar_symbols: list[str],
):
    if not isinstance(terms, list) or not terms:
        raise ValueError("算符表达式必须是非空 term 数组")
    if len(terms) > 64:
        raise ValueError("算符表达式项数超过上限")
    sp = quantum["sp"]
    allowed = ALGEBRA_OPERATORS[algebra]
    result = sp.S.Zero
    components: list[Any] = []
    for term in terms:
        if not isinstance(term, dict):
            raise ValueError("每个算符项必须是对象")
        operators = term.get("operators")
        if not isinstance(operators, list):
            raise ValueError("算符项 operators 必须是数组")
        if len(operators) > 64:
            raise ValueError("单项式算符数超过上限")
        product = sp.S.One
        for token in operators:
            if token not in allowed:
                raise ValueError(f"算符 {token!r} 不属于 {algebra} 代数")
            product *= quantum["operators"][token]
        component = (
            _coefficient(term.get("coefficient", 1), sp, scalar_symbols, quantum["hbar"]) * product
        )
        components.append(component)
        result += component
    return result, components


def _binary_result(operation: str, left: Any, right: Any, quantum: Mapping[str, Any]):
    if operation == "commutator":
        return quantum["Commutator"](left, right).expand(commutator=True).doit()
    if operation == "anticommutator":
        return quantum["AntiCommutator"](left, right).expand(anticommutator=True).doit()
    if operation == "product":
        return left * right
    raise ValueError(f"未知算符代数操作：{operation}")


def operator_algebra(args: Mapping[str, Any]) -> Any:
    algebra = str(args.get("algebra") or "")
    if algebra not in ALGEBRA_OPERATORS:
        raise ValueError(f"未知算符代数：{algebra}")
    quantum = _sympy_quantum()
    scalar_symbols = list(args.get("scalar_symbols") or [])
    left, left_components = _expression(
        args.get("left_terms"),
        algebra=algebra,
        quantum=quantum,
        scalar_symbols=scalar_symbols,
    )
    operation = str(args.get("operation") or "commutator")
    right = None
    if operation == "adjoint":
        result = quantum["Dagger"](left).doit()
        derivation_terms = [
            {
                "left_term": str(component),
                "right_term": None,
                "result": str(quantum["Dagger"](component).doit()),
            }
            for component in left_components
        ]
    else:
        if "right_terms" not in args:
            raise ValueError(f"{operation} 运算缺少 right_terms")
        right, right_components = _expression(
            args.get("right_terms"),
            algebra=algebra,
            quantum=quantum,
            scalar_symbols=scalar_symbols,
        )
        result = _binary_result(operation, left, right, quantum)
        derivation_terms = [
            {
                "left_term": str(lhs),
                "right_term": str(rhs),
                "result": str(quantum["sp"].expand(_binary_result(operation, lhs, rhs, quantum))),
            }
            for lhs in left_components
            for rhs in right_components
        ]
    sp = quantum["sp"]
    result = sp.expand(result)
    unresolved = result.has(quantum["Commutator"], quantum["AntiCommutator"])
    return {
        "expression": str(result),
        "zero": bool(result == 0),
        "resolved": not unresolved,
        "algebra": algebra,
        "derivation_terms": derivation_terms,
        "normalized_inputs": {
            "left": str(left),
            "right": str(right) if right is not None else None,
        },
    }


def density_matrix_check(args: Mapping[str, Any]) -> Any:
    """Verify idempotency of a pure-state density matrix ρ=|ket⟩⟨bra| (ρ²=ρ)."""
    try:
        import sympy as sp
        from sympy.physics.quantum import Bra, Ket, OuterProduct
        from sympy.physics.quantum.innerproduct import InnerProduct
    except ImportError as exc:
        raise RuntimeError("SymPy 默认后端缺失，请重新安装 quanllm-harness") from exc
    ket_label = str(args.get("ket_label") or "").strip()
    bra_label = str(args.get("bra_label") or "").strip()
    if not ket_label or not bra_label:
        raise ValueError("ket_label 与 bra_label 必须是非空字符串")
    normalized = bool(args.get("normalized", True))
    ket = Ket(ket_label)
    bra = Bra(bra_label)
    rho = OuterProduct(ket, bra)
    inner = InnerProduct(bra, ket)
    rho2 = (ket * inner * bra).doit()
    difference = sp.expand(rho2 - rho)
    assumption = ""
    if normalized and ket_label == bra_label:
        difference = sp.expand(difference.subs(inner, 1))
        assumption = f"<{bra_label}|{ket_label}>=1"
    return {
        "density_matrix": str(rho),
        "rho_squared": str(rho2),
        "difference": str(difference),
        "idempotent": difference == 0,
        "assumption": assumption or "无（未做归一化假设）",
    }


def operator_tools() -> tuple[Tool, ...]:
    operator_tokens = sorted(set().union(*ALGEBRA_OPERATORS.values()))
    term_schema = {
        "type": "object",
        "properties": {
            "coefficient": {"type": ["string", "number"]},
            "operators": {
                "type": "array",
                "description": "按从左到右的乘法次序排列；空数组表示恒等算符。",
                "maxItems": 64,
                "items": {"enum": operator_tokens},
            },
        },
        "required": ["operators"],
        "additionalProperties": False,
    }
    terms_schema = {
        "type": "array",
        "description": "数组中的各个单项式彼此相加，不是相乘。",
        "minItems": 1,
        "maxItems": 64,
        "items": term_schema,
    }
    return (
        Tool(
            "operator_algebra",
            "精确计算结构化量子算符的对易子、反对易子、乘积或厄米共轭。"
            "先选择 algebra：canonical 仅用 x/p；angular_momentum 仅用 Jx/Jy/Jz/Jplus/Jminus；"
            "boson_single_mode 仅用 a/adag；fermion_single_mode 仅用 c/cdag。"
            "left_terms 与 right_terms 是求和项，每项 operators 是严格保序的乘法序列。"
            "不使用 label、power、type、properties、items、required 等 Schema 字段作为参数。",
            {
                "type": "object",
                "properties": {
                    "algebra": {"enum": sorted(ALGEBRA_OPERATORS)},
                    "operation": {"enum": ["commutator", "anticommutator", "product", "adjoint"]},
                    "left_terms": terms_schema,
                    "right_terms": terms_schema,
                    "scalar_symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "uniqueItems": True,
                    },
                },
                "required": ["algebra", "operation", "left_terms"],
                "additionalProperties": False,
            },
            operator_algebra,
            limitations=(
                "内置后端只处理单模玻色/费米代数；多模关系交由默认的专用多体后端",
                "一次调用只处理所选代数族，不擅自假定不同物理子系统互相对易",
            ),
            claim_kinds=("definition", "equation", "derivation", "condition", "conclusion"),
        ),
        Tool(
            "density_matrix_check",
            "核验纯态密度矩阵 ρ=|ket⟩⟨bra| 的幂等性 ρ²=ρ（在归一化 ⟨bra|ket⟩=1 假设下）。"
            "用于 ρ²=ρ、纯态密度矩阵等断言。",
            {
                "type": "object",
                "properties": {
                    "ket_label": {
                        "type": "string",
                        "description": "|ket⟩ 的标签，如 psi",
                    },
                    "bra_label": {
                        "type": "string",
                        "description": "⟨bra| 的标签，如 psi",
                    },
                    "normalized": {
                        "type": "boolean",
                        "description": "是否假设 ⟨bra|ket⟩=1，默认 true",
                    },
                },
                "required": ["ket_label", "bra_label"],
                "additionalProperties": False,
            },
            density_matrix_check,
            claim_kinds=("definition", "equation", "derivation", "condition", "conclusion"),
        ),
    )


__all__ = ["ALGEBRA_OPERATORS", "operator_algebra", "operator_tools"]
