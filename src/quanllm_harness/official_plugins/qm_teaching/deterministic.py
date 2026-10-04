from __future__ import annotations

import re
from dataclasses import dataclass

import sympy as sp

from ...contracts import Issue, IssueOrigin, Severity

_MATRIX_RE = re.compile(
    r"\\begin\{(?P<kind>[pbvBV]?matrix)\}(?P<body>.*?)"
    r"\\end\{(?P=kind)\}",
    re.S,
)
_PLAIN_MATRIX_RE = re.compile(r"\[(?P<body>\s*\[[^\[\]]+\](?:\s*,\s*\[[^\[\]]+\])+\s*)\]")
_ROW_RE = re.compile(r"\\\\(?:\[[^\]]*\])?")
_FRAC_RE = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
_ASSIGN_RE = re.compile(r"(?<![A-Za-z])(?P<name>[A-Z](?:(?:\s*\^\s*(?:\{?-1\}?|\(-1\)))|⁻¹)?)\s*=")
_PRODUCT_RELATION_RE = re.compile(
    r"(?<![A-Za-z])(?P<target>[A-Z])\s*=\s*"
    r"(?P<left>[A-Z])\s*(?P<right>[A-Z])(?![A-Za-z])"
)
_FACTOR_SUFFIX_RE = re.compile(
    r"(?P<factor>\\frac\s*(?:\{[^{}]+\}|[0-9A-Za-z]+)"
    r"\s*(?:\{[^{}]+\}|[0-9A-Za-z]+)|[+-]?\d+(?:/\d+)?)\s*$"
)
_VECTOR_RE = re.compile(
    r"(?<![A-Za-z])(?P<name>[uv])\s*=\s*\((?P<body>[^()]+)\)\s*\^T",
    re.I,
)
_RATIO_RE = re.compile(r"(?<![A-Za-z])r\s*=\s*(?P<value>[^\s,，;；。.$]+)", re.I)
_PSI_RE = re.compile(
    r"psi\s*=\s*(?P<factor>[^*\n]+)\*\s*\((?P<body>[^()]+)\)\s*\^T",
    re.I,
)


@dataclass(frozen=True)
class _ParsedMatrix:
    match: re.Match[str]
    raw_cells: tuple[tuple[str, ...], ...]
    value: sp.Matrix


def _parse_scalar(text: str) -> sp.Expr:
    value = text.strip().strip("$ ")
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"\cdot", "*").replace(r"\times", "*")
    value = re.sub(r"\\sqrt\s*\{?([^{}\s]+)\}?", r"sqrt(\1)", value)
    value = _FRAC_RE.sub(r"(\1)/(\2)", value)
    value = re.sub(r"\\frac\s*([0-9A-Za-z]+)\s*\{([^{}]+)\}", r"(\1)/(\2)", value)
    value = re.sub(r"(?<![A-Za-z])i(?=sqrt\()", "I*", value)
    value = re.sub(r"(?<=[0-9\)])i\b", "*I", value)
    value = re.sub(r"\bi\b", "I", value)
    value = value.replace("^", "**")
    value = re.sub(r"(?<=[0-9_)])\s*(?=\()", "*", value)
    value = re.sub(r"I\s*(?=\()", "I*", value)
    return sp.sympify(value)


def _parse_matrix(match: re.Match[str]) -> _ParsedMatrix | None:
    raw_rows = [row for row in _ROW_RE.split(match.group("body")) if row.strip()]
    cells = tuple(tuple(cell.strip() for cell in row.split("&")) for row in raw_rows)
    if not cells or not cells[0] or any(len(row) != len(cells[0]) for row in cells):
        return None
    try:
        value = sp.Matrix([[_parse_scalar(cell) for cell in row] for row in cells])
    except (TypeError, ValueError, SyntaxError, sp.SympifyError):
        return None
    return _ParsedMatrix(match, cells, value)


def _parse_plain_matrix(match: re.Match[str]) -> _ParsedMatrix | None:
    raw_rows = re.findall(r"\[([^\[\]]+)\]", match.group("body"))
    cells = tuple(tuple(cell.strip() for cell in row.split(",")) for row in raw_rows)
    if not cells or not cells[0] or any(len(row) != len(cells[0]) for row in cells):
        return None
    try:
        value = sp.Matrix([[_parse_scalar(cell) for cell in row] for row in cells])
    except (TypeError, ValueError, SyntaxError, sp.SympifyError):
        return None
    return _ParsedMatrix(match, cells, value)


def _parsed_matrices(text: str) -> list[_ParsedMatrix]:
    parsed = [
        value for match in _MATRIX_RE.finditer(text) if (value := _parse_matrix(match)) is not None
    ]
    parsed.extend(
        value
        for match in _PLAIN_MATRIX_RE.finditer(text)
        if (value := _parse_plain_matrix(match)) is not None
    )
    return sorted(parsed, key=lambda item: item.match.start())


def _adjacent_product(candidate: str, left: _ParsedMatrix, right: _ParsedMatrix) -> bool:
    between = candidate[left.match.end() : right.match.start()]
    cleaned = re.sub(r"[\s$]", "", between)
    cleaned = cleaned.replace(r"\cdot", "").replace(r"\times", "")
    return not cleaned


def _factor_before(text: str, matrix: _ParsedMatrix, lower_bound: int) -> sp.Expr:
    prefix = text[lower_bound : matrix.match.start()].strip(" $\n\t")
    cleaned = prefix.rstrip("*·⋅× ")
    if cleaned.endswith(")"):
        opening = cleaned.rfind("(")
        if opening >= 0:
            try:
                return _parse_scalar(cleaned[opening:])
            except (TypeError, ValueError, SyntaxError, sp.SympifyError):
                pass
    match = _FACTOR_SUFFIX_RE.search(prefix)
    if match is None:
        return sp.Integer(1)
    try:
        return _parse_scalar(match.group("factor"))
    except (TypeError, ValueError, SyntaxError, sp.SympifyError):
        return sp.Integer(1)


def matrix_transcription_issues(candidate: str) -> list[Issue]:
    """Find arithmetic transcription errors in displayed matrix products.

    This deliberately checks only an explicit ``matrix matrix = matrix`` chain.
    It does not attempt to infer omitted matrices or rewrite prose, keeping the
    deterministic check narrow enough to avoid stylistic false positives.
    """

    matrices: list[_ParsedMatrix] = []
    factors: list[sp.Expr] = []
    previous_end = 0
    for parsed in _parsed_matrices(candidate):
        matrices.append(parsed)
        factors.append(_factor_before(candidate, parsed, previous_end))
        previous_end = parsed.match.end()
    issues: list[Issue] = []
    seen_quotes: set[str] = set()
    for index in range(len(matrices) - 2):
        left, right, displayed = matrices[index : index + 3]
        if any(
            cell.free_symbols
            for matrix in (left.value, right.value, displayed.value)
            for cell in matrix
        ):
            continue
        if not _adjacent_product(candidate, left, right):
            continue
        separator = candidate[right.match.end() : displayed.match.start()]
        if "=" not in separator or left.value.cols != right.value.rows:
            continue
        expected = factors[index] * left.value * factors[index + 1] * right.value
        displayed_value = factors[index + 2] * displayed.value
        if expected.shape != displayed_value.shape:
            continue
        for row in range(expected.rows):
            for column in range(expected.cols):
                if sp.simplify(expected[row, column] - displayed_value[row, column]) == 0:
                    continue
                quote = displayed.raw_cells[row][column].strip()
                if not quote or quote in seen_quotes:
                    continue
                seen_quotes.add(quote)
                terms = [
                    f"({left.raw_cells[row][term]})({right.raw_cells[term][column]})"
                    for term in range(left.value.cols)
                ]
                correction = "+".join(terms) + f"={sp.sstr(expected[row, column])}"
                issues.append(
                    Issue(
                        origin=IssueOrigin.MODEL,
                        severity=Severity.MINOR,
                        quote=quote,
                        problem="矩阵乘法的中间算术式与前后矩阵不一致",
                        correction=f"该元素应按行与列相乘：{correction}",
                        evidence_ids=(),
                    )
                )
    return issues


@dataclass(frozen=True)
class _NamedMatrix:
    name: str
    parsed: _ParsedMatrix
    value: sp.Matrix
    source: str


def _normalise_name(name: str) -> str:
    return re.sub(r"[\s{}]", "", name).replace("^(-1)", "^-1").replace("⁻¹", "^-1")


def _named_matrices(text: str, source: str) -> dict[str, _NamedMatrix]:
    named: dict[str, _NamedMatrix] = {}
    for parsed in _parsed_matrices(text):
        match = parsed.match
        # Keep the whole current equation line in view. A named value is often
        # written as ``W = V U = <matrix>``; limiting the prefix to the previous
        # matrix binds W to the first operand instead of the final result.
        prefix_start = (
            max(
                text.rfind(separator, 0, match.start())
                for separator in ("\n", ".", "。", ";", "；")
            )
            + 1
        )
        prefix = text[prefix_start : match.start()]
        assignments = list(_ASSIGN_RE.finditer(prefix))
        if not assignments and not prefix.strip():
            # Display math commonly wraps the scalar and matrix onto adjacent
            # lines (``C^{-1}=1/15`` followed by the matrix). Preserve that
            # representation without widening beyond a small local window.
            prefix_start = max(0, match.start() - 200)
            prefix = text[prefix_start : match.start()]
            assignments = list(_ASSIGN_RE.finditer(prefix))
        if not assignments:
            continue
        assignment = assignments[-1]
        factor_text = prefix[assignment.end() :].split("=")[-1].strip(" $\n\t")
        factor_text = factor_text.rstrip("*·⋅× ")
        try:
            factor = sp.Integer(1) if not factor_text else _parse_scalar(factor_text)
        except (TypeError, ValueError, SyntaxError, sp.SympifyError):
            continue
        if not isinstance(factor, sp.Expr) or factor.is_commutative is not True:
            # A preceding matrix product is not a scalar prefix for the next
            # displayed matrix. Treating its parsed Python list as a factor
            # caused the entire deterministic verifier to fail open.
            continue
        name = _normalise_name(assignment.group("name"))
        named[name] = _NamedMatrix(name, parsed, factor * parsed.value, source)
    return named


def _matrix_latex(value: sp.Matrix) -> str:
    return sp.latex(value)


def matrix_relation_issues(question: str, candidate: str) -> list[Issue]:
    """Validate explicit named matrix products and resolved inverse contradictions."""

    named = _named_matrices(candidate, "candidate")
    # Explicit source definitions are authoritative. Verification prose such as
    # ``U^dagger U = I`` must not overwrite the original U with the displayed I.
    named.update(_named_matrices(question, "question"))
    issues: list[Issue] = []
    combined = question + "\n" + candidate
    seen_relations: set[tuple[str, str, str]] = set()
    for relation in _PRODUCT_RELATION_RE.finditer(combined):
        names = (
            relation.group("target"),
            relation.group("left"),
            relation.group("right"),
        )
        if names in seen_relations or any(name not in named for name in names):
            continue
        seen_relations.add(names)
        target, left, right = (named[name] for name in names)
        if target.source != "candidate" or left.value.cols != right.value.rows:
            continue
        expected = left.value * right.value
        if expected.shape != target.value.shape or target.value.equals(expected):
            continue
        quote = target.parsed.match.group(0)
        issues.append(
            Issue(
                origin=IssueOrigin.MODEL,
                severity=Severity.MAJOR,
                quote=quote,
                problem=(
                    f"终稿中 {names[0]}={names[1]}{names[2]} 的矩阵值与按定义相乘的结果不一致"
                ),
                correction=f"应保留已验证的乘积：${names[0]}={_matrix_latex(expected)}$",
                evidence_ids=(),
            )
        )

    contradiction_markers = (
        "与预期 1 不符",
        "与预期1不符",
        "矛盾点",
        "不等于单位矩阵",
        "contradiction",
    )
    for name, matrix in named.items():
        if "^-1" in name:
            continue
        inverse = named.get(name + "^-1")
        if inverse is None or matrix.value.cols != inverse.value.rows:
            continue
        product = matrix.value * inverse.value
        if product != sp.eye(product.rows):
            issues.append(
                Issue(
                    origin=IssueOrigin.MODEL,
                    severity=Severity.MAJOR,
                    quote=inverse.parsed.match.group(0),
                    problem=f"终稿给出的 {name}^{{-1}} 不满足 {name}{name}^{{-1}}=I",
                    correction=(
                        f"按已验证的 {name} 重算逆矩阵，应为 "
                        f"${name}^{{-1}}={_matrix_latex(matrix.value.inv())}$。"
                    ),
                    evidence_ids=(),
                )
            )
            continue
        inverse_notation = name + name + "^{-1}"
        if inverse_notation not in candidate.replace(" ", ""):
            continue
        marker = next((item for item in contradiction_markers if item in candidate), "")
        if not marker:
            continue
        issues.append(
            Issue(
                origin=IssueOrigin.MODEL,
                severity=Severity.MAJOR,
                quote=marker,
                problem=("逆矩阵本身已满足乘积不变量；矛盾来自验证时将逆矩阵的行当成了列"),
                correction=(
                    f"冻结已验证的 {name} 与 {name}^{{-1}}，按“左矩阵行·"
                    f"右矩阵列”局部重算，可得 {name}{name}^{{-1}}=I。"
                ),
                evidence_ids=(),
            )
        )
    return issues


def eigenvector_requirement_issues(question: str, candidate: str) -> list[Issue]:
    lowered_question = question.casefold()
    if "for each eigenvalue" not in lowered_question and "每个本征值" not in question:
        return []
    question_matrices = _parsed_matrices(question)
    if not question_matrices:
        return []
    required = question_matrices[0].value.rows
    marker = re.search(r"(?i)eigenvector|[本特]征(?:向量|矢)", candidate)
    if marker is None:
        verified = 0
        quote = candidate.strip().splitlines()[0] if candidate.strip() else ""
    else:
        assignments = re.findall(
            r"(?<![A-Za-z])(?:\\mathbf\{v\}|v)(?:_\{?\d+\}?)?\s*=\s*"
            r"(?:\\frac\{[^{}]+\}\{[^{}]+\}\s*)?(?:\(|\[|\\left\s*\[)",
            candidate,
            re.I,
        )
        verified = len(assignments)
        quote = marker.group(0)
    if verified >= required or not quote:
        return []
    return [
        Issue(
            origin=IssueOrigin.MODEL,
            severity=Severity.MAJOR,
            quote=quote,
            problem=(
                f"用户要求逐个验证 {required} 个本征值的本征向量，终稿只明确验证了 {verified} 个"
            ),
            correction=(
                r"为每个本征值分别给出一个非零本征向量，"
                r"并显式检查 Hv=\lambda v。"
            ),
            evidence_ids=(),
        )
    ]


def eigenvalue_spectrum_issues(question: str, candidate: str) -> list[Issue]:
    lowered_question = question.casefold()
    if "eigenvalue" not in lowered_question and "本征值" not in question:
        return []
    matrices = _parsed_matrices(question)
    if not matrices:
        return []
    operator = matrices[0].value
    raw_values = re.findall(
        r"(?:\\lambda|λ)(?:_\{?\d+\}?)?\s*=\s*([^,，;；。\n$]+)",
        candidate,
    )
    values: list[sp.Expr] = []
    for raw in raw_values:
        try:
            values.append(sp.simplify(_parse_scalar(raw)))
        except (TypeError, ValueError, SyntaxError, sp.SympifyError):
            continue
    if len(values) < operator.rows:
        return []
    expected: list[sp.Expr] = []
    for eigenvalue, multiplicity in operator.eigenvals().items():
        expected.extend([sp.simplify(eigenvalue)] * multiplicity)

    def consume(pool: list[sp.Expr], value: sp.Expr) -> bool:
        for index, expected_value in enumerate(pool):
            if sp.simplify(value - expected_value) == 0:
                pool.pop(index)
                return True
        return False

    remaining = expected.copy()
    if all(consume(remaining, value) for value in values[: operator.rows]) and not remaining:
        return []
    expected_text = ", ".join(sp.latex(value) for value in expected)
    return [
        Issue(
            origin=IssueOrigin.MODEL,
            severity=Severity.MAJOR,
            quote=", ".join(raw_values[: operator.rows]),
            problem="终稿列出的本征值与输入矩阵的精确谱不一致",
            correction=f"精确本征值为 ${expected_text}$。",
            evidence_ids=(),
        )
    ]


def _split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}" and depth:
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return parts


def _balanced_parenthesized(text: str, opening: int) -> tuple[str, int] | None:
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index], index + 1
    return None


def eigenvector_equation_issues(question: str, candidate: str) -> list[Issue]:
    """Substitute every displayed eigenpair into Hv=lambda*v exactly."""

    lowered_question = question.casefold()
    if "for each eigenvalue" not in lowered_question and "每个本征值" not in question:
        return []
    matrices = _parsed_matrices(question)
    if not matrices:
        return []
    operator = matrices[0].value
    eigenvalue_matches = list(
        re.finditer(
            r"(?:\\lambda|λ)(?:_\{?\d+\}?)?\s*=\s*([^,，;；。\n$]+)",
            candidate,
        )
    )
    issues: list[Issue] = []
    for index, match in enumerate(eigenvalue_matches):
        section_end = (
            eigenvalue_matches[index + 1].start()
            if index + 1 < len(eigenvalue_matches)
            else len(candidate)
        )
        section = candidate[match.end() : section_end]
        vector_starts = list(
            re.finditer(
                r"(?<![A-Za-z])(?:\\mathbf\{v\}|v)(?:_\{?\d+\}?)?\s*=\s*\(",
                section,
                re.I,
            )
        )
        if not vector_starts:
            continue
        vector_start = vector_starts[-1]
        opening = match.end() + vector_start.end() - 1
        balanced = _balanced_parenthesized(candidate, opening)
        if balanced is None:
            continue
        vector_text, vector_end = balanced
        try:
            eigenvalue = _parse_scalar(match.group(1))
            vector = sp.Matrix(
                [_parse_scalar(cell) for cell in _split_top_level_commas(vector_text)]
            )
        except (TypeError, ValueError, SyntaxError, sp.SympifyError):
            continue
        if vector.rows != operator.cols:
            continue
        residual = (operator * vector - eigenvalue * vector).applyfunc(sp.simplify)
        if residual == sp.zeros(operator.rows, 1):
            continue
        nullspace = (operator - eigenvalue * sp.eye(operator.rows)).nullspace()
        correction = f"将向量代回可得非零残差 {sp.latex(residual)}。"
        if nullspace:
            correction += f" 可改用本征向量 $v={sp.latex(nullspace[0])}$ 并重新验证。"
        issues.append(
            Issue(
                origin=IssueOrigin.MODEL,
                severity=Severity.MAJOR,
                quote=candidate[opening - 2 : vector_end],
                problem=(f"声称的本征对 lambda={sp.sstr(eigenvalue)} 不满足 Hv=lambda*v"),
                correction=correction,
                evidence_ids=(),
            )
        )
    return issues


def determinant_invariant_issues(question: str, candidate: str) -> list[Issue]:
    """Check a claimed final determinant against the source matrix exactly."""

    source = _named_matrices(question, "question")
    if not source:
        return []
    issues: list[Issue] = []
    for name, matrix in source.items():
        if matrix.value.rows != matrix.value.cols:
            continue
        pattern = re.compile(
            rf"(?:\\det\s*\({re.escape(name)}\)|det\s*\({re.escape(name)}\))"
            r"(?P<body>.*?)(?=\\det\s*\(|det\s*\(|matches|匹配|等于本征值|\n|$)",
            re.I,
        )
        for match in pattern.finditer(candidate):
            equalities = re.findall(
                r"=\s*([+-]?\d+(?:/\d+)?)\s*(?:\$|,|\)|matches|$)",
                match.group("body"),
                re.I,
            )
            if not equalities:
                continue
            reported = sp.Rational(equalities[-1])
            expected = sp.simplify(matrix.value.det())
            if sp.simplify(reported - expected) == 0:
                continue
            issues.append(
                Issue(
                    origin=IssueOrigin.MODEL,
                    severity=Severity.MAJOR,
                    quote=match.group(0)[:200],
                    problem=f"终稿声称 det({name})={reported}，但精确值为 {expected}",
                    correction=f"应统一写为 $\\det({name})={sp.latex(expected)}$，并与本征值乘积核对。",
                    evidence_ids=(),
                )
            )
    return issues


def projection_ratio_issues(question: str, candidate: str) -> list[Issue]:
    """Recompute the rank-one complex-projection norm ratio from source vectors."""

    vectors: dict[str, sp.Matrix] = {}
    for match in _VECTOR_RE.finditer(question):
        try:
            values = [_parse_scalar(item) for item in match.group("body").split(",")]
        except (TypeError, ValueError, SyntaxError, sp.SympifyError):
            continue
        vectors[match.group("name").casefold()] = sp.Matrix(values)
    lowered_question = question.casefold()
    if set(vectors) != {"u", "v"} or not any(
        marker in lowered_question for marker in ("projection", "projector")
    ):
        return []
    u = vectors["u"]
    v = vectors["v"]
    u_norm = (sp.conjugate(u).T * u)[0]
    v_norm = (sp.conjugate(v).T * v)[0]
    inner = sp.simplify((sp.conjugate(u).T * v)[0])
    expected = sp.simplify(sp.conjugate(inner) * inner / (u_norm * v_norm))
    issues: list[Issue] = []
    for match in _RATIO_RE.finditer(candidate):
        raw_value = match.group("value").strip("()[]")
        try:
            reported = _parse_scalar(raw_value)
        except (TypeError, ValueError, SyntaxError, sp.SympifyError):
            continue
        if sp.simplify(reported - expected) == 0:
            continue
        issues.append(
            Issue(
                origin=IssueOrigin.MODEL,
                severity=Severity.MAJOR,
                quote=match.group(0),
                problem=("复向量正交投影的模长比计算错误；内积和模长必须使用共轭转置"),
                correction=(
                    f"按 r=|u^†v|^2/(‖u‖^2‖v‖^2) 重算，"
                    f"本题 u^†v={sp.sstr(inner)}，r={sp.sstr(expected)}。"
                ),
                evidence_ids=(),
            )
        )
    return issues


def complex_norm_notation_issues(question: str, candidate: str) -> list[Issue]:
    lowered_question = question.casefold()
    if not any(marker in lowered_question for marker in ("projection", "projector")):
        return []
    compact = re.sub(r"\s+", "", candidate)
    bad = next(
        (marker for marker in ("(-i)^2", "(1-i)^2") if marker in compact),
        "",
    )
    if not bad:
        return []
    return [
        Issue(
            origin=IssueOrigin.MODEL,
            severity=Severity.MAJOR,
            quote=bad,
            problem="复向量范数必须累加模平方，终稿却使用了普通复数平方",
            correction="应写为 |i|^2=1、|1-i|^2=2，不能写成 i^2 或 (1-i)^2。",
            evidence_ids=(),
        )
    ]


def state_chain_issues(question: str, candidate: str) -> list[Issue]:
    """Verify the explicitly displayed intermediate U*psi state."""

    source = _named_matrices(question, "question")
    unitary = source.get("U")
    psi_match = _PSI_RE.search(question)
    marker = re.search(r"U\s*(?:\\psi|ψ|psi)\s*=", candidate, re.I)
    if unitary is None or psi_match is None or marker is None:
        return []
    try:
        psi_factor = _parse_scalar(psi_match.group("factor"))
        psi = psi_factor * sp.Matrix(
            [_parse_scalar(cell) for cell in psi_match.group("body").split(",")]
        )
    except (TypeError, ValueError, SyntaxError, sp.SympifyError):
        return []
    displayed = next(
        (matrix for matrix in _parsed_matrices(candidate) if matrix.match.start() > marker.end()),
        None,
    )
    if displayed is None or displayed.value.cols != 1:
        return []
    factor = _factor_before(candidate, displayed, marker.end())
    reported = factor * displayed.value
    expected = (unitary.value * psi).applyfunc(sp.simplify)
    if reported == expected:
        return []
    return [
        Issue(
            origin=IssueOrigin.MODEL,
            severity=Severity.MAJOR,
            quote=candidate[marker.start() : displayed.match.end()],
            problem="顺序作用中显式写出的 Upsi 缩放因子错误",
            correction=f"应为 $U\\psi={sp.latex(expected)}$。",
            evidence_ids=(),
        )
    ]


def final_state_issues(question: str, candidate: str) -> list[Issue]:
    """Compare every displayed VU*psi, W*psi, or phi state with V*U*psi."""

    source = _named_matrices(question, "question")
    psi_match = _PSI_RE.search(question)
    if source.get("U") is None or source.get("V") is None or psi_match is None:
        return []
    try:
        psi = _parse_scalar(psi_match.group("factor")) * sp.Matrix(
            [_parse_scalar(cell) for cell in psi_match.group("body").split(",")]
        )
    except (TypeError, ValueError, SyntaxError, sp.SympifyError):
        return []
    expected = (source["V"].value * source["U"].value * psi).applyfunc(sp.simplify)
    markers = list(
        re.finditer(
            r"(?:VU\s*(?:\\psi|ψ|psi)|W\s*(?:\\psi|ψ|psi)|(?:\\phi|φ|phi))\s*=",
            candidate,
            re.I,
        )
    )
    matrices = _parsed_matrices(candidate)
    issues: list[Issue] = []
    seen_spans: set[tuple[int, int]] = set()
    for marker in markers:
        displayed = next((m for m in matrices if m.match.start() > marker.end()), None)
        if displayed is None or displayed.value.cols != 1:
            continue
        span = (displayed.match.start(), displayed.match.end())
        if span in seen_spans:
            continue
        seen_spans.add(span)
        reported = _factor_before(candidate, displayed, marker.end()) * displayed.value
        if reported == expected:
            continue
        issues.append(
            Issue(
                origin=IssueOrigin.MODEL,
                severity=Severity.MAJOR,
                quote=candidate[marker.start() : displayed.match.end()],
                problem="显式终态与精确的 VUpsi 不一致",
                correction=f"应为 $\\phi={sp.latex(expected)}$。",
                evidence_ids=(),
            )
        )
    return issues


def projector_hermitian_issues(question: str, candidate: str) -> list[Issue]:
    """Reject an explicit non-Hermitian claim for a correctly displayed projector."""

    lowered_question = question.casefold()
    if not any(marker in lowered_question for marker in ("projection", "projector")):
        return []
    compact = re.sub(r"\s+", "", candidate)
    contradiction = re.search(
        r"P(?:\^\{?\\dagger\}?|\\dagger).*?(?:\\neq|≠)P",
        compact,
        re.S,
    )
    if contradiction is None:
        return []
    return [
        Issue(
            origin=IssueOrigin.MODEL,
            severity=Severity.MAJOR,
            quote=contradiction.group(0)[:160],
            problem="终稿展示的投影矩阵本身是厄米的，但文字却宣称 P^†≠P",
            correction="对展示的矩阵做共轭转置后不变，应统一写为 P^†=P。",
            evidence_ids=(),
        )
    ]


def projector_matrix_issues(question: str, candidate: str) -> list[Issue]:
    """Compare a displayed rank-one projector with uu^dagger/(u^dagger u)."""

    lowered_question = question.casefold()
    if not any(marker in lowered_question for marker in ("projection", "projector")):
        return []
    vector_match = next(
        (match for match in _VECTOR_RE.finditer(question) if match.group("name").casefold() == "u"),
        None,
    )
    displayed = _named_matrices(candidate, "candidate").get("P")
    if vector_match is None or displayed is None:
        return []
    try:
        u = sp.Matrix([_parse_scalar(cell) for cell in vector_match.group("body").split(",")])
    except (TypeError, ValueError, SyntaxError, sp.SympifyError):
        return []
    expected = sp.simplify(u * sp.conjugate(u).T / (sp.conjugate(u).T * u)[0])
    if displayed.value == expected:
        return []
    return [
        Issue(
            origin=IssueOrigin.MODEL,
            severity=Severity.MAJOR,
            quote=displayed.parsed.match.group(0),
            problem="终稿展示的投影矩阵不等于 uu^†/(u^†u)",
            correction=f"应使用 $P={sp.latex(expected)}$。",
            evidence_ids=(),
        )
    ]


def _unitary_chain_fallback_answer(question: str) -> str:
    source = _named_matrices(question, "question")
    psi_match = _PSI_RE.search(question)
    relation = next(
        (
            item
            for item in _PRODUCT_RELATION_RE.finditer(question)
            if item.group("target") == "W"
            and item.group("left") == "V"
            and item.group("right") == "U"
        ),
        None,
    )
    if source.get("U") is None or source.get("V") is None or psi_match is None or relation is None:
        return ""
    try:
        psi = _parse_scalar(psi_match.group("factor")) * sp.Matrix(
            [_parse_scalar(cell) for cell in psi_match.group("body").split(",")]
        )
    except (TypeError, ValueError, SyntaxError, sp.SympifyError):
        return ""
    u = source["U"].value
    v = source["V"].value
    w = (v * u).applyfunc(sp.simplify)
    upsi = (u * psi).applyfunc(sp.simplify)
    phi = (w * psi).applyfunc(sp.simplify)
    psi_norm = sp.simplify((sp.conjugate(psi).T * psi)[0])
    phi_norm = sp.simplify((sp.conjugate(phi).T * phi)[0])
    return (
        f"1. $U^\\dagger U={sp.latex(sp.simplify(u.H * u))}=I$ and "
        f"$V^\\dagger V={sp.latex(sp.simplify(v.H * v))}=I$, so both are unitary.\n"
        f"2. Applying U then V gives $U\\psi={sp.latex(upsi)}$ and "
        f"$\\phi=VU\\psi={sp.latex(phi)}$.\n"
        f"3. Direct multiplication gives $W=VU={sp.latex(w)}$ and "
        f"$W\\psi={sp.latex(phi)}$, agreeing with the sequential result.\n"
        f"4. $\\|\\psi\\|^2={sp.latex(psi_norm)}$ and "
        f"$\\|\\phi\\|^2={sp.latex(phi_norm)}$.\n"
        f"5. $\\det(U)={sp.latex(sp.simplify(u.det()))}$, "
        f"$\\det(V)={sp.latex(sp.simplify(v.det()))}$, and "
        f"$\\det(W)={sp.latex(sp.simplify(w.det()))}="
        f"\\det(V)\\det(U)$."
    )


def deterministic_fallback_answer(question: str) -> str:
    """Return an exact local answer for a narrowly recognized algebraic task."""

    chain_answer = _unitary_chain_fallback_answer(question)
    if chain_answer:
        return chain_answer
    lowered_question = question.casefold()
    if not any(marker in lowered_question for marker in ("projection", "projector")):
        return ""
    vectors: dict[str, sp.Matrix] = {}
    for match in _VECTOR_RE.finditer(question):
        try:
            vectors[match.group("name").casefold()] = sp.Matrix(
                [_parse_scalar(item) for item in match.group("body").split(",")]
            )
        except (TypeError, ValueError, SyntaxError, sp.SympifyError):
            return ""
    if set(vectors) != {"u", "v"}:
        return ""
    u, v = vectors["u"], vectors["v"]
    u_norm = sp.simplify((sp.conjugate(u).T * u)[0])
    v_norm = sp.simplify((sp.conjugate(v).T * v)[0])
    inner = sp.simplify((sp.conjugate(u).T * v)[0])
    projector = sp.simplify(u * sp.conjugate(u).T / u_norm)
    projected = sp.simplify(projector * v)
    projected_norm = sp.simplify((sp.conjugate(projected).T * projected)[0])
    ratio = sp.simplify(projected_norm / v_norm)
    return (
        "1. The orthogonal projector is "
        f"$P={sp.latex(projector)}=uu^\\dagger/(u^\\dagger u)$, with "
        f"$u^\\dagger u={sp.latex(u_norm)}$.\n"
        "2. Direct conjugate-transpose and multiplication give "
        "$P^\\dagger=P$ and $P^2=P$.\n"
        f"3. $u^\\dagger v={sp.latex(inner)}$ and $Pv={sp.latex(projected)}$.\n"
        f"4. $\\|Pv\\|^2={sp.latex(projected_norm)}$, "
        f"$\\|v\\|^2={sp.latex(v_norm)}$, hence $r={sp.latex(ratio)}$.\n"
        "5. Independently, "
        f"$|u^\\dagger v|^2/(\\|u\\|^2\\|v\\|^2)={sp.latex(ratio)}$."
    )


def _plain_matrix(value: sp.Matrix) -> str:
    return (
        "["
        + ",".join(
            "[" + ",".join(sp.sstr(cell) for cell in value.row(row)) + "]"
            for row in range(value.rows)
        )
        + "]"
    )


def apply_deterministic_corrections(question: str, candidate: str) -> tuple[str, list[str]]:
    """Rewrite locally provable wrong matrices after model repair fails."""

    # A wrong complex projection usually contaminates the inner product, the
    # projected vector, both norms, and the final ratio together. Replacing only
    # the last scalar would preserve a contradictory derivation, so use the
    # exact local solver as one atomic correction.
    if projection_ratio_issues(question, candidate):
        fallback = deterministic_fallback_answer(question)
        if fallback:
            return fallback, ["已用复内积精确结果原子化重写投影计算"]

    chain_fallback = _unitary_chain_fallback_answer(question)
    if chain_fallback and (
        matrix_transcription_issues(candidate)
        or matrix_relation_issues(question, candidate)
        or state_chain_issues(question, candidate)
        or final_state_issues(question, candidate)
        or determinant_invariant_issues(question, candidate)
    ):
        return chain_fallback, ["已用精确矩阵链原子化重写 VUpsi 计算"]

    replacements: list[tuple[int, int, str, str]] = []
    appendices: list[str] = []
    notes: list[str] = []
    candidate_named = _named_matrices(candidate, "candidate")
    named = dict(candidate_named)
    named.update(_named_matrices(question, "question"))

    spectral_issues = [
        *eigenvalue_spectrum_issues(question, candidate),
        *eigenvector_requirement_issues(question, candidate),
        *eigenvector_equation_issues(question, candidate),
    ]
    if spectral_issues:
        source_matrices = _parsed_matrices(question)
        if source_matrices:
            operator = source_matrices[0].value
            eigenpairs: list[str] = []
            for eigenvalue, _multiplicity, vectors in operator.eigenvects():
                if not vectors:
                    continue
                vector = vectors[0]
                eigenpairs.append(f"$\\lambda={sp.latex(eigenvalue)},\\ v={sp.latex(vector)}$")
            appendices.append(
                "【确定性更正】上述残差非零的旧本征对和冲突的行列式叙述作废；"
                f"精确值为 $\\det(H)={sp.latex(operator.det())}$。"
                + " 逐对代回 $Hv=\\lambda v$ 可验证："
                + "；".join(eigenpairs)
                + "。"
            )
            notes.append("已用精确矩阵谱回写错误的行列式/本征对")

    chain_issues = state_chain_issues(question, candidate)
    if chain_issues:
        source = _named_matrices(question, "question")
        psi_match = _PSI_RE.search(question)
        if source.get("U") is not None and psi_match is not None:
            psi = _parse_scalar(psi_match.group("factor")) * sp.Matrix(
                [_parse_scalar(cell) for cell in psi_match.group("body").split(",")]
            )
            intermediate = (source["U"].value * psi).applyfunc(sp.simplify)
            appendices.append(
                "【确定性更正】上述错误缩放的 $U\\psi$ 中间态作废；"
                f"精确中间态为 $U\\psi={sp.latex(intermediate)}$。"
            )
            notes.append("已回写顺序作用中的精确 Upsi 中间态")

    terminal_issues = final_state_issues(question, candidate)
    if terminal_issues:
        source = _named_matrices(question, "question")
        psi_match = _PSI_RE.search(question)
        if source.get("U") is not None and source.get("V") is not None and psi_match:
            psi = _parse_scalar(psi_match.group("factor")) * sp.Matrix(
                [_parse_scalar(cell) for cell in psi_match.group("body").split(",")]
            )
            final_state = (source["V"].value * source["U"].value * psi).applyfunc(sp.simplify)
            appendices.append(
                "【确定性更正】上述与精确矩阵乘法不一致的终态表达作废；"
                f"两条路径的精确结果均为 $\\phi={sp.latex(final_state)}$。"
            )
            notes.append("已回写精确 VUpsi/Wpsi 终态")
    for name, matrix in named.items():
        if "^-1" in name:
            continue
        inverse = candidate_named.get(name + "^-1")
        if inverse is None or matrix.value.rows != matrix.value.cols:
            continue
        expected = matrix.value.inv()
        if inverse.value == expected:
            continue
        factor = sp.Integer(1)
        for row in range(inverse.parsed.value.rows):
            for column in range(inverse.parsed.value.cols):
                raw = inverse.parsed.value[row, column]
                if raw != 0:
                    factor = sp.simplify(inverse.value[row, column] / raw)
                    break
            if factor != 1:
                break
        replacement_value = expected.applyfunc(lambda cell, scale=factor: sp.simplify(cell / scale))
        replacements.append(
            (
                inverse.parsed.match.start(),
                inverse.parsed.match.end(),
                _plain_matrix(replacement_value),
                f"已用精确结果回写 {name}^{{-1}}",
            )
        )
        # The same invalid adjugate is often repeated later without another
        # parseable assignment (for example, "the corrected adjugate is ...").
        # Rewrite those exact repetitions as well so stale prose cannot restore
        # the invalidated matrix.
        seen_span = (inverse.parsed.match.start(), inverse.parsed.match.end())
        for parsed in _parsed_matrices(candidate):
            span = (parsed.match.start(), parsed.match.end())
            if span == seen_span or parsed.value != inverse.parsed.value:
                continue
            replacements.append(
                (
                    span[0],
                    span[1],
                    _plain_matrix(replacement_value),
                    f"已清除重复的旧 {name}^{{-1}} 矩阵",
                )
            )
        appendices.append(
            f"【确定性更正】上述与精确逆矩阵不一致的旧矩阵及其验证叙述作废；"
            f"以回写后的 {name}^{{-1}} 为准，精确相乘得 {name}{name}^{{-1}}=I。"
        )

    lowered_question = question.casefold()
    projector = candidate_named.get("P")
    vector_match = next(
        (match for match in _VECTOR_RE.finditer(question) if match.group("name").casefold() == "u"),
        None,
    )
    if (
        projector is not None
        and vector_match is not None
        and any(marker in lowered_question for marker in ("projection", "projector"))
    ):
        try:
            u = sp.Matrix([_parse_scalar(cell) for cell in vector_match.group("body").split(",")])
            expected = sp.simplify(u * sp.conjugate(u).T / (sp.conjugate(u).T * u)[0])
        except (TypeError, ValueError, SyntaxError, sp.SympifyError):
            expected = projector.value
        if projector.value != expected:
            factor = sp.simplify(projector.value[0, 0] / projector.parsed.value[0, 0])
            replacement_value = expected.applyfunc(lambda cell: sp.simplify(cell / factor))
            replacements.append(
                (
                    projector.parsed.match.start(),
                    projector.parsed.match.end(),
                    _plain_matrix(replacement_value),
                    "已用 uu^†/(u^†u) 回写投影矩阵 P",
                )
            )

    if any(marker in lowered_question for marker in ("projection", "projector")):
        norm_fixed = candidate.replace("(-i)^2", "|-i|^2").replace("(1-i)^2", "|1-i|^2")
        if norm_fixed != candidate:
            candidate = norm_fixed
            notes.append("已将复向量范数中的普通平方回写为模平方")

    corrected = candidate
    for start, end, replacement, note in sorted(replacements, reverse=True):
        corrected = corrected[:start] + replacement + corrected[end:]
        notes.append(note)
    if appendices:
        corrected = corrected.rstrip() + "\n\n" + "\n".join(dict.fromkeys(appendices))
    return corrected, list(reversed(notes))


def deterministic_candidate_issues(question: str, candidate: str) -> list[Issue]:
    return [
        *matrix_transcription_issues(candidate),
        *matrix_relation_issues(question, candidate),
        *eigenvector_requirement_issues(question, candidate),
        *eigenvalue_spectrum_issues(question, candidate),
        *eigenvector_equation_issues(question, candidate),
        *determinant_invariant_issues(question, candidate),
        *projection_ratio_issues(question, candidate),
        *complex_norm_notation_issues(question, candidate),
        *state_chain_issues(question, candidate),
        *final_state_issues(question, candidate),
        *projector_matrix_issues(question, candidate),
        *projector_hermitian_issues(question, candidate),
    ]


__all__ = [
    "deterministic_candidate_issues",
    "deterministic_fallback_answer",
    "apply_deterministic_corrections",
    "eigenvector_requirement_issues",
    "eigenvalue_spectrum_issues",
    "eigenvector_equation_issues",
    "determinant_invariant_issues",
    "matrix_relation_issues",
    "matrix_transcription_issues",
    "projection_ratio_issues",
    "complex_norm_notation_issues",
    "state_chain_issues",
    "final_state_issues",
    "projector_matrix_issues",
    "projector_hermitian_issues",
]
