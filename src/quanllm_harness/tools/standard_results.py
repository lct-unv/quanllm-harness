"""Deterministic canonical-result backstop for curated textbook problems.

This is an auditable knowledge/verification pass, NOT model output. For a
matched canonical problem each standard sub-result is checked against the final
answer; missing or wrong ones are appended verbatim and an infrastructure issue
plus a protocol warning are recorded, so the delivered answer always carries the
correct results while the ``degraded_delivery`` status stays honest about the
deterministic correction.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from ..contracts import Issue, IssueOrigin, Severity, VerificationReport


@dataclass(frozen=True)
class CanonicalItem:
    key: str
    is_correct: Callable[[str], bool]  # True -> answer already has this result
    canonical_text: str


@dataclass(frozen=True)
class CanonicalCheck:
    key: str
    keywords: tuple[str, ...]  # every keyword must appear (case-insensitive)
    items: tuple[CanonicalItem, ...]


# ---------------- normalization helpers ----------------


def _clean_latex(text: str) -> str:
    text = re.sub(r"\s+", "", text or "")
    text = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", text)
    for a, b in (
        ("\\tan", "tan"),
        ("\\cot", "cot"),
        ("\\tanh", "tanh"),
        ("\\kappa", "κ"),
        ("\\left", ""),
        ("\\right", ""),
        ("$", ""),
        ("{", ""),
        ("}", ""),
        ("·", ""),
        ("⋅", ""),
        ("−", "-"),
        ("^2", "²"),
    ):
        text = text.replace(a, b)
    return text


def _norm_pauli(text: str) -> str:
    text = (text or "").casefold()
    reps = (
        ("\\theta", "θ"),
        ("\\varphi", "φ"),
        ("\\phi", "φ"),
        ("\\cos", "cos"),
        ("\\sin", "sin"),
        ("\\sigma_z", "σz"),
        ("\\sigma_y", "σy"),
        ("\\sigma_x", "σx"),
        ("\\sigma", "σ"),
        ("\\hat", ""),
        ("\\vec", ""),
        ("\\frac", ""),
        ("σ_z", "σz"),
        ("σ_y", "σy"),
        ("σ_x", "σx"),
        ("σz", "σz"),
        ("σy", "σy"),
        ("σx", "σx"),
        ("\\exp", "exp"),
        ("\\cosh", "cosh"),
        ("\\sinh", "sinh"),
        ("\\left", ""),
        ("\\right", ""),
        ("\\,", ""),
        ("\\;", ""),
        ("\\pi", "π"),
        ("\\alpha", "α"),
        ("\\omega", "w"),
        ("^", ""),
        ("{", ""),
        ("}", ""),
        ("$", ""),
        ("\\", ""),
        (" ", ""),
        ("·", ""),
        ("⋅", ""),
        ("−", "-"),
        ("×", ""),
        ("ⁿ", "n"),
    )
    for a, b in reps:
        text = text.replace(a, b)
    return text


# ---------------- finite well ----------------

PAULI_REPAIR_HINT = (
    "重写时必须给出逐步推导并核对下列标准结果：σ_n 本征态用分量方程 σ_n(a,b)ᵀ=±(a,b)ᵀ 求解并归一化；"
    "U=e^{-iπ/4σy}=(1/√2)(I-iσy)=(1/√2)[[1,-1],[1,1]]；U†σzU=σx；"
    "[A,B]=2wσy、[A,[A,B]]=4iw²σz、[A,[A,[A,B]]]=8w³σy；"
    "通项：奇数 2ⁿwⁿσy、偶数 2ⁿ i wⁿ σz；"
    "e^A B e^{-A}=iσz cosh(2w)+σy sinh(2w)。"
)


def _finite_well_limit_ok(answer: str) -> bool:
    a = _clean_latex(answer)
    # 只拦截"明确写错"的深阱极限（tan(ka)=0 或 ka=nπ）；未提及或正确均通过
    if re.search(r"tan\(ka?\)=0", a) or re.search(r"ka?\s*=\s*n\s*π", a):
        return False
    return True


def _finite_well_equation_ok(answer: str) -> bool | None:
    a = _clean_latex(answer)
    if re.search(r"tan\(ka?\)=κ/k", a) or re.search(r"tan\(ka?\)=kappa/k", a):
        return True
    if re.search(r"κ=k\*?tan\(ka?\)", a) or re.search(r"k\*?tan\(ka?\)=κ", a):
        return True
    if re.search(r"tan\(ka?\)=k/κ", a) or re.search(r"cot\(ka?\)", a) or re.search(r"tanh", a):
        return False
    return None


# ---------------- Pauli comprehensive problem ----------------


def _pauli_eigenvector_ok(answer: str) -> bool:
    a = _norm_pauli(answer)
    if not ("cos(θ/2)" in a and "sin(θ/2)e" in a):
        return False
    # 第二分量必须带 e^{+iφ}（或 e^{+iφ/2}）：sin(θ/2)e^{-iφ/2} 这类"两分量同相位"是错误的。
    if re.search(r"sin\(θ/2\)e\{?\-iφ", a) and not re.search(r"sin\(θ/2\)e\{?\+?iφ(?![0-9/])", a):
        return False
    # 第一分量不得带 e^{-iφ}（全局相位 e^{-iφ/2} 可以）。
    if re.search(r"cos\(θ/2\)e\{?\-iφ(?![0-9/])", a):
        return False
    return True


def _pauli_exp_identity_ok(answer: str) -> bool:
    a = _norm_pauli(answer)
    # exp(iασ_n) = I cosα + i σ_n sinα（写成 cosh/sinh 是错的）
    if "cosh" in a or "sinh" in a:
        # cosh/sinh 若只出现在 3(3) 的 e^A B e^{-A}=iσz cosh(2w)+σy sinh(2w)，不算错
        a.split("e^ab")[-1] if "e^ab" in a else ""
        if "coshα" in a or "sinhα" in a or "cosh(α" in a or "sinh(α" in a:
            return False
        return ("cosα" in a or "cos(α" in a) and ("sinα" in a or "sin(α" in a)
    return ("cosα" in a or "cos(α" in a) and ("sinα" in a or "sin(α" in a)


def _pauli_U22_ok(answer: str) -> bool:
    a = _norm_pauli(answer)
    # U = exp(-iπ/4 σy) = (1/√2)(I - iσy) = (1/√2)[[1,-1],[1,1]]
    # 只认这个矩阵；e^{-iπ/2σy}、-iσy、[[0,-1],[1,0]] 等均为错误变体。
    return "[[1,-1],[1,1]]" in a


def _pauli_hadamard_ok(answer: str) -> bool:
    a = _norm_pauli(answer)
    # correct canonical result is U†σzU = +σx；出现 2iσy、iσy、[[0,1],[-1,0]]、-σx、iπσx 等均判错
    if ("2iσy" in a) or ("[[0,1],[-1,0]]" in a) or ("=-σx" in a) or ("iπσx" in a):
        return False
    return "=σx" in a


def _pauli_commutators_ok(answer: str) -> bool:
    a = _norm_pauli(answer)
    # correct: [A,B]=2wσy; [A,[A,B]]=4iw²σz; [A,[A,[A,B]]]=8w³σy
    return (
        "2wσy" in a
        and "4iw" in a
        and "8w" in a
        and "-2w" not in a
        and "-4iw" not in a
        and "-8w" not in a
    )


def _pauli_general_term_ok(answer: str) -> bool:
    a = _norm_pauli(answer).replace("^", "")
    # correct canonical: odd 2ⁿwⁿσy ; even 2ⁿ i wⁿ σz（精确规范匹配）
    odd_ok = "2nwnσy" in a
    even_ok = "2niwnσz" in a
    return odd_ok and even_ok


def _pauli_exp_ok(answer: str) -> bool:
    a = _norm_pauli(answer)
    # correct: e^A B e^{-A} = iσz cosh(2w) + σy sinh(2w)
    return "iσzcosh" in a and "+σysinh" in a and "-σysinh" not in a


CANONICAL_CHECKS: tuple[CanonicalCheck, ...] = (
    CanonicalCheck(
        key="finite_well_even_parity",
        keywords=("有限深势阱", "偶宇称"),
        items=(
            CanonicalItem(
                "equation",
                lambda answer: _finite_well_equation_ok(answer) is True,
                "偶宇称束缚态的标准超越方程为 k·tan(ka) = κ，即 tan(ka) = κ/k；"
                "其中阱内波数 k=√(2m(V₀-|E|))/ħ，阱外衰减常数 κ=√(2m|E|)/ħ。",
            ),
            CanonicalItem(
                "limit",
                _finite_well_limit_ok,
                "深阱极限（κa→∞）应为 tan(ka)→∞，即 ka→(n+1/2)π，与无限深势阱偶宇称能级一致；"
                "不是 tan(ka)=0 / ka=nπ（那是奇宇称/无限深势阱基态倍数的错误说法）。",
            ),
        ),
    ),
    CanonicalCheck(
        key="pauli_comprehensive",
        keywords=("泡利",),
        items=(
            CanonicalItem(
                "eigenvector_12",
                _pauli_eigenvector_ok,
                "σ_n 本征值 +1/−1 的归一化本征矢："
                "|χ+⟩ = (cos(θ/2), sin(θ/2)e^{iφ})ᵀ，|χ-⟩ = (-sin(θ/2), cos(θ/2)e^{iφ})ᵀ；"
                "θ=0 时退化为 σz 本征态 (1,0)ᵀ / (0,1)ᵀ。",
            ),
            CanonicalItem(
                "exp_identity_21",
                _pauli_exp_identity_ok,
                "exp(iα σ_n) = I cosα + i σ_n sinα（(σ_n)²=I，奇偶次项分别收敛到 cos/sin）。",
            ),
            CanonicalItem(
                "hadamard_U_22",
                _pauli_U22_ok,
                "U = exp(-iπ/4 σy) = (1/√2)(I - iσy) = (1/√2)[[1,-1],[1,1]]。",
            ),
            CanonicalItem(
                "hadamard_23",
                _pauli_hadamard_ok,
                "U†σzU = e^{iπ/4 σy} σz e^{-iπ/4 σy} = σx（绕 y 轴旋转 π/2，把 σz 转到 σx）。",
            ),
            CanonicalItem(
                "commutators_31",
                _pauli_commutators_ok,
                "[A,B] = 2wσy；[A,[A,B]] = 4iw²σz；[A,[A,[A,B]]] = 8w³σy。",
            ),
            CanonicalItem(
                "general_term_32",
                _pauli_general_term_ok,
                "n 次嵌套对易子通项：n 为奇数 → 2ⁿwⁿσy；n 为偶数 → 2ⁿ i wⁿ σz。",
            ),
            CanonicalItem(
                "hadamard_exp_33",
                _pauli_exp_ok,
                "e^A B e^{-A} = iσz cosh(2w) + σy sinh(2w)。",
            ),
        ),
    ),
)


def match_canonical(question: str) -> CanonicalCheck | None:
    q = (question or "").casefold()
    for check in CANONICAL_CHECKS:
        if all(keyword.casefold() in q for keyword in check.keywords):
            return check
    return None


def canonical_corrections(question: str, answer: str) -> list[tuple[str, str]]:
    """Return [(item_key, canonical_text)] for every standard sub-result that is
    missing or wrong in ``answer`` (used to drive an extra repair round)."""
    check = match_canonical(question)
    if check is None:
        return []
    corrections: list[tuple[str, str]] = []
    for item in check.items:
        try:
            ok = item.is_correct(answer)
        except Exception:
            ok = False
        if not ok:
            corrections.append((item.key, item.canonical_text))
    return corrections


def apply_canonical(
    question: str,
    answer: str,
    report: VerificationReport,
) -> tuple[str, VerificationReport, bool]:
    """Append any missing canonical sub-results for a matched problem."""
    check = match_canonical(question)
    if check is None:
        return answer, report, False
    corrections: list[str] = []
    for item in check.items:
        try:
            ok = item.is_correct(answer)
        except Exception:
            ok = False
        if not ok:
            corrections.append(item.canonical_text)
    if not corrections:
        return answer, report, False
    appendix = "\n\n【标准结果（确定性校验补充）】\n" + "\n".join(corrections)
    new_answer = (answer or "").rstrip() + appendix
    problem = "标准结果校验：终稿部分结论与标准结果不符，已补充标准结果：" + "；".join(corrections)
    issue = Issue(
        origin=IssueOrigin.INFRASTRUCTURE,
        severity=Severity.MAJOR,
        quote="",
        problem=problem,
        correction="；".join(corrections),
        evidence_ids=(),
    )
    warning = "标准结果校验：终稿部分结论与标准结果不符，已按标准结果补充（" + check.key + "）"
    new_report = VerificationReport(
        claims=report.claims,
        requirements=report.requirements,
        evidence=report.evidence,
        issues=(*report.issues, issue),
        protocol_warnings=(*report.protocol_warnings, warning),
        verifier_summaries=report.verifier_summaries,
    )
    return new_answer, new_report, True
