from __future__ import annotations

from quanllm_harness.contracts import Claim, Evidence, IssueOrigin
from quanllm_harness.plugins import InvariantContext
from quanllm_harness.verification import (
    hermitian_invariant,
    measurement_invariant,
    projection_invariant,
    unitary_invariant,
)


def evidence(
    evidence_id: str,
    operation: str,
    arguments: dict,
    *,
    result: dict | None = None,
    ok: bool = True,
    input_verified: bool = True,
) -> Evidence:
    return Evidence(
        id=evidence_id,
        tool="matrix_calculate",
        operation=operation,
        arguments=arguments,
        ok=ok,
        result=result if result is not None else {},
        claim_ids=("C-001",),
        input_verified=input_verified,
        supports_claim=None,
    )


def context(question: str, claim_quote: str, evidence_items) -> InvariantContext:
    return InvariantContext(
        question=question,
        candidate=claim_quote,
        claims=(Claim("C-001", claim_quote, "equation", "major"),),
        requirements=(),
        evidence=tuple(evidence_items),
    )


NOT_UNITARY = [[1, 1], [0, 1]]
UNITARY = [[0, 1], [1, 0]]
NOT_HERMITIAN = [["0", "i"], ["0", "0"]]
HERMITIAN = [["0", "i"], ["-i", "0"]]
PROJECTOR = [[1, 0], [0, 0]]
NOT_IDEMPOTENT = [[1, 0], [0, "1/2"]]


def test_unitary_invariant_flags_declared_non_unitary_matrix():
    item = evidence(
        "E-0001",
        "unitary",
        {"matrix": NOT_UNITARY},
        result={"unitary": True, "normalized_inputs": {"matrix": NOT_UNITARY}},
    )
    result = unitary_invariant()(context("验证 U 是幺正矩阵。", "U 是幺正矩阵", [item]))
    assert result.checked == 1
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.origin is IssueOrigin.MODEL
    assert "U†U=I" in issue.problem
    assert issue.evidence_ids == ("E-0001",)


def test_unitary_invariant_accepts_unitary_matrix_and_parses_string_cells():
    item = evidence("E-0001", "unitary", {"matrix": UNITARY})
    result = unitary_invariant()(context("验证幺正性。", "该矩阵幺正", [item]))
    assert result.checked == 1
    assert result.issues == ()


def test_unitary_invariant_ignores_matrices_the_call_did_not_declare():
    item = evidence(
        "E-0001",
        "multiply",
        {"matrix": UNITARY, "other_matrix": UNITARY},
        result={"matrix": NOT_UNITARY},
    )
    result = unitary_invariant()(context("验证幺正性。", "W=VU 幺正", [item]))
    assert result.checked == 0
    assert result.issues == ()


def test_unitary_invariant_ignores_unreviewed_or_failed_evidence():
    failed = evidence("E-0001", "unitary", {"matrix": NOT_UNITARY}, ok=False)
    unreviewed = evidence("E-0002", "unitary", {"matrix": NOT_UNITARY}, input_verified=False)
    result = unitary_invariant()(context("验证幺正性。", "U 幺正", [failed, unreviewed]))
    assert result.checked == 0
    assert result.issues == ()


def test_unitary_invariant_is_inert_without_the_property_marker():
    item = evidence("E-0001", "unitary", {"matrix": NOT_UNITARY})
    result = unitary_invariant()(context("计算矩阵乘积。", "计算 W=VU", [item]))
    assert result.checked == 0
    assert result.issues == ()


def test_hermitian_invariant_handles_complex_entries():
    good = evidence("E-0001", "hermitian", {"matrix": HERMITIAN})
    bad = evidence("E-0002", "hermitian", {"matrix": NOT_HERMITIAN})
    result = hermitian_invariant()(context("检查 H 是否厄米。", "H 为厄米算符", [good, bad]))
    assert result.checked == 2
    assert len(result.issues) == 1
    assert result.issues[0].evidence_ids == ("E-0002",)


def test_projection_invariant_requires_hermitian_and_idempotent():
    good = evidence("E-0001", "projection", {"matrix": PROJECTOR})
    not_idempotent = evidence("E-0002", "projection", {"matrix": NOT_IDEMPOTENT})
    result = projection_invariant()(
        context("构造投影算符 P。", "P 是投影算符", [good, not_idempotent])
    )
    assert result.checked == 2
    assert len(result.issues) == 1
    assert "P†=P 且 P²=P" in result.issues[0].problem


def test_measurement_invariant_checks_complete_distributions():
    good = evidence("E-0001", "measurement", {"probabilities": [0.5, 0.5]})
    incomplete = evidence("E-0002", "measurement", {"probabilities": [0.5, 0.4]})
    out_of_range = evidence("E-0003", "measurement", {"probabilities": [1.2, -0.2]})
    result = measurement_invariant()(
        context("计算测量概率。", "各结果概率之和为 1", [good, incomplete, out_of_range])
    )
    assert result.checked == 3
    assert len(result.issues) == 2


def test_measurement_invariant_allows_named_subsets_but_not_out_of_range_values():
    subset = evidence("E-0001", "measurement", {"probabilities": {"lambda=1": "1/6"}})
    result = measurement_invariant()(context("计算测量概率。", "概率为 1/6", [subset]))
    assert result.checked == 1
    assert result.issues == ()


def test_measurement_invariant_tolerates_float_rounding():
    item = evidence("E-0001", "measurement", {"probabilities": [0.7, 0.30000000000000004]})
    result = measurement_invariant()(context("计算概率。", "概率之和为 1", [item]))
    assert result.checked == 1
    assert result.issues == ()
