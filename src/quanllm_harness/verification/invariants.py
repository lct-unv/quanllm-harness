from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import sympy as sp

from ..contracts import Evidence, Issue, IssueOrigin, Severity
from ..plugins import InvariantContext, InvariantResult

# Issue 003 - reusable domain invariants.
#
# An invariant is a deterministic, evidence-driven predicate attached to a
# domain property (here: projection, unitary, Hermitian and measurement
# properties of linear algebra). The host runs every registered invariant on
# every verification pass and folds the findings into the report, so important
# mathematical properties no longer depend on the model remembering to state
# them.
#
# Safety rule: an invariant only validates the matrices a tool call *declared*
# (its reviewed arguments, or the echo of those arguments in the tool result),
# and only for operations that explicitly ask for that property. A matrix that
# merely appears in some computed result is never judged, so the invariants
# cannot invent false positives from intermediate products.

_TOLERANCE = 1e-9

_PROJECTION_MARKERS = ("投影", "projection", "projector")
_UNITARY_MARKERS = ("幺正", "unitary")
_HERMITIAN_MARKERS = ("厄米", "自伴", "自共轭", "hermitian")
_MEASUREMENT_MARKERS = ("概率", "probability", "measurement", "测量")

_PROBABILITY_KEYS = ("probability", "probabilities", "probs", "概率")

_FRAC_RE = re.compile(r"\\d?frac\s*\{?([^{}\s]+)\}?\s*\{?([^{}\s]+)\}?")
_SQRT_RE = re.compile(r"\\sqrt\s*\{?([^{}\s]+)\}?")


def _parse_scalar(value: Any) -> sp.Expr | None:
    """Parse one JSON matrix cell or probability into a SymPy expression."""

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return sp.Integer(value)
    if isinstance(value, float):
        return sp.Rational(str(value))
    if not isinstance(value, str):
        return None
    text = value.strip().strip("$ ")
    if not text:
        return None
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = text.replace(r"\cdot", "*").replace(r"\times", "*")
    text = text.replace(r"\,", "").replace(r"\!", "")
    text = _SQRT_RE.sub(r"sqrt(\1)", text)
    text = _FRAC_RE.sub(r"((\1)/(\2))", text)
    text = text.replace("^", "**")
    text = re.sub(r"(?<=[0-9\)])i\b", "*I", text)
    text = re.sub(r"(?<![A-Za-z_])i(?![A-Za-z_])", "I", text)
    try:
        return sp.sympify(text, rational=True, locals={})
    except (SyntaxError, TypeError, ValueError, sp.SympifyError):
        return None


def _as_matrix(value: Any) -> sp.Matrix | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    if not all(isinstance(row, (list, tuple)) for row in value):
        return None
    if not all(len(row) == len(value[0]) for row in value):
        return None
    rows: list[list[sp.Expr]] = []
    for row in value:
        cells: list[sp.Expr] = []
        for cell in row:
            parsed = _parse_scalar(cell)
            if parsed is None:
                return None
            cells.append(parsed)
        rows.append(cells)
    matrix = sp.Matrix(rows)
    if matrix.rows != matrix.cols:
        return None
    return matrix


def _declared_matrices(evidence: Evidence) -> list[tuple[str, sp.Matrix]]:
    """Matrices the reviewed call declared, plus the tool's echo of them."""

    payloads: list[tuple[str, Any]] = [("调用输入", evidence.arguments)]
    if isinstance(evidence.result, Mapping):
        payloads.append(("工具回显输入", evidence.result.get("normalized_inputs")))
    found: list[tuple[str, sp.Matrix]] = []
    seen: set[tuple[int, str]] = set()
    for label, payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        for key, value in payload.items():
            matrix = _as_matrix(value)
            if matrix is None:
                continue
            signature = (matrix.rows, str(matrix))
            if signature in seen:
                continue
            seen.add(signature)
            found.append((f"{label}.{key}", matrix))
    return found


def _probability_vector(value: Any) -> tuple[list[sp.Expr], bool] | None:
    """Return the vector plus whether it claims to be a complete distribution.

    A positional list is treated as the full outcome vector, so its sum must be
    one. A named mapping may be a reported subset (for example only the
    probabilities the candidate chose to mention), so only the 0 ≤ p ≤ 1 range
    invariant is enforced for it.
    """

    if isinstance(value, Mapping):
        values = list(value.values())
        complete = False
    elif isinstance(value, (list, tuple)):
        values = list(value)
        complete = True
    else:
        return None
    if not values:
        return None
    parsed: list[sp.Expr] = []
    for item in values:
        expression = _parse_scalar(item)
        if expression is None:
            return None
        parsed.append(expression)
    return parsed, complete


def _probability_entries(payload: Any, prefix: str = "") -> list[tuple[str, list[sp.Expr], bool]]:
    found: list[tuple[str, list[sp.Expr], bool]] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            label = f"{prefix}{key}"
            lowered = str(key).casefold()
            if any(marker in lowered for marker in _PROBABILITY_KEYS):
                vector = _probability_vector(value)
                if vector is not None:
                    parsed, complete = vector
                    found.append((label, parsed, complete))
                    continue
            if isinstance(value, Mapping):
                found.extend(_probability_entries(value, f"{label}."))
    return found


def _equal(left: sp.Expr, right: sp.Expr) -> bool:
    try:
        difference = sp.simplify(left - right)
    except Exception:
        return False
    if difference == 0:
        return True
    if difference.is_number:
        try:
            return abs(complex(sp.N(difference, 15))) <= _TOLERANCE
        except (TypeError, ValueError):
            return False
    return False


def _entrywise(
    left: sp.Matrix, right: sp.Matrix, predicate: Callable[[sp.Expr, sp.Expr], bool]
) -> bool:
    if left.shape != right.shape:
        return False
    return all(
        predicate(left[row, col], right[row, col])
        for row in range(left.rows)
        for col in range(left.cols)
    )


def _is_hermitian(matrix: sp.Matrix) -> bool:
    return _entrywise(matrix, matrix.H, _equal)


def _is_idempotent(matrix: sp.Matrix) -> bool:
    return _entrywise(matrix * matrix, matrix, _equal)


def _is_unitary(matrix: sp.Matrix) -> bool:
    return _entrywise(matrix.H * matrix, sp.eye(matrix.rows), _equal)


def _numeric(value: sp.Expr) -> float | None:
    expression = sp.simplify(value)
    if not expression.is_real or expression.free_symbols:
        return None
    try:
        return float(sp.N(expression, 15))
    except (TypeError, ValueError):
        return None


def _triggered(context: InvariantContext, markers: Sequence[str]) -> bool:
    haystack = " ".join([context.question, *(claim.quote for claim in context.claims)]).casefold()
    return any(marker in haystack for marker in markers)


def _operation_matches(operation: str, markers: Sequence[str]) -> bool:
    lowered = (operation or "").casefold()
    return any(marker in lowered for marker in markers)


def _matrix_invariant(
    label: str,
    property_text: str,
    operation_markers: Sequence[str],
    trigger_markers: Sequence[str],
    predicate: Callable[[sp.Matrix], bool],
    advice: str,
):
    def invariant(context: InvariantContext) -> InvariantResult:
        if not _triggered(context, trigger_markers):
            return InvariantResult()
        issues: list[Issue] = []
        checked = 0
        seen: set[tuple[int, str]] = set()
        for evidence in context.evidence:
            if not evidence.ok or not evidence.input_verified:
                continue
            if not _operation_matches(evidence.operation, operation_markers):
                continue
            for path, matrix in _declared_matrices(evidence):
                signature = (matrix.rows, str(matrix))
                if signature in seen:
                    continue
                seen.add(signature)
                checked += 1
                if predicate(matrix):
                    continue
                issues.append(
                    Issue(
                        origin=IssueOrigin.MODEL,
                        severity=Severity.MAJOR,
                        quote="",
                        problem=(
                            f"领域不变量 {label}（{property_text}）未通过：证据 "
                            f"{evidence.id} 声明的矩阵 {path} 不满足该性质。"
                        ),
                        correction=advice,
                        evidence_ids=(evidence.id,),
                    )
                )
        return InvariantResult(checked=checked, issues=tuple(issues))

    return invariant


def hermitian_invariant(label: str = "hermitian"):
    """Independently check H†=H for matrices declared to a Hermiticity call."""

    return _matrix_invariant(
        label,
        "H†=H",
        ("hermitian", "adjoint", "conjugate_transpose"),
        _HERMITIAN_MARKERS,
        _is_hermitian,
        "按 H†=H 重算该矩阵并重新核验厄米性。",
    )


def unitary_invariant(label: str = "unitary"):
    """Independently check U†U=I for matrices declared to a unitarity call."""

    return _matrix_invariant(
        label,
        "U†U=I",
        ("unitary",),
        _UNITARY_MARKERS,
        _is_unitary,
        "按 U†U=I 重算该矩阵并重新核验幺正性。",
    )


def projection_invariant(label: str = "projection"):
    """Independently check P†=P and P²=P for declared projection matrices."""

    def predicate(matrix: sp.Matrix) -> bool:
        return _is_hermitian(matrix) and _is_idempotent(matrix)

    return _matrix_invariant(
        label,
        "P†=P 且 P²=P",
        ("projection", "projector", "idempotent", "project"),
        _PROJECTION_MARKERS,
        predicate,
        "按 P=(u u†)/(u†u) 重算投影矩阵，并重新核验 P†=P 与 P²=P。",
    )


def measurement_invariant(label: str = "measurement"):
    """Check 0 ≤ p ≤ 1 and Σp = 1 for probability vectors in reviewed payloads."""

    def invariant(context: InvariantContext) -> InvariantResult:
        if not _triggered(context, _MEASUREMENT_MARKERS):
            return InvariantResult()
        issues: list[Issue] = []
        checked = 0
        for evidence in context.evidence:
            if not evidence.ok or not evidence.input_verified:
                continue
            payloads: list[Any] = [evidence.arguments]
            if isinstance(evidence.result, Mapping):
                payloads.append(evidence.result)
                payloads.append(evidence.result.get("normalized_inputs"))
            for payload in payloads:
                for path, vector, complete in _probability_entries(payload):
                    numbers = [_numeric(item) for item in vector]
                    if any(number is None for number in numbers):
                        continue
                    checked += 1
                    values = [number for number in numbers if number is not None]
                    total = sum(values)
                    in_range = all(-_TOLERANCE <= value <= 1 + _TOLERANCE for value in values)
                    normalized = abs(total - 1.0) <= _TOLERANCE if complete else True
                    if in_range and normalized:
                        continue
                    issues.append(
                        Issue(
                            origin=IssueOrigin.MODEL,
                            severity=Severity.MAJOR,
                            quote="",
                            problem=(
                                f"领域不变量 {label}（0≤p≤1 且 Σp=1）未通过：证据 "
                                f"{evidence.id} 中的概率向量 {path}"
                                + (f" 之和为 {total:g}" if complete else " 存在越界分量")
                                + "。"
                            ),
                            correction="重新归一化概率向量并重新核验测量概率之和为 1。",
                            evidence_ids=(evidence.id,),
                        )
                    )
        return InvariantResult(checked=checked, issues=tuple(issues))

    return invariant
