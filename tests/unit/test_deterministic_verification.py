from quanllm_harness.contracts import IssueOrigin, Severity
from quanllm_harness.official_plugins.qm_teaching.deterministic import (
    apply_deterministic_corrections,
    determinant_invariant_issues,
    deterministic_candidate_issues,
    deterministic_fallback_answer,
    eigenvalue_spectrum_issues,
    eigenvector_equation_issues,
    eigenvector_requirement_issues,
    final_state_issues,
    matrix_relation_issues,
    matrix_transcription_issues,
    projection_ratio_issues,
    projector_hermitian_issues,
    projector_matrix_issues,
    state_chain_issues,
)


def test_case_5_matrix_transcription_error_is_detected():
    candidate = r"""
    AA^{-1}=
    \begin{pmatrix}2&1\\3&2\end{pmatrix}
    \begin{pmatrix}2&-1\\-3&2\end{pmatrix}
    =\begin{pmatrix}(4-3)&(2-2)\\(6-6)&(3+4)\end{pmatrix}
    =\begin{pmatrix}1&0\\0&1\end{pmatrix}.
    """

    issues = matrix_transcription_issues(candidate)

    assert len(issues) == 1
    assert issues[0].origin is IssueOrigin.MODEL
    assert issues[0].severity is Severity.MINOR
    assert issues[0].quote == "(3+4)"
    assert "(3)(-1)+(2)(2)=1" in issues[0].correction


def test_correct_matrix_product_has_no_transcription_issue():
    candidate = r"""
    \begin{pmatrix}2&1\\3&2\end{pmatrix}
    \begin{pmatrix}2&-1\\-3&2\end{pmatrix}
    =\begin{pmatrix}(4-3)&(-2+2)\\(6-6)&(-3+4)\end{pmatrix}.
    """

    assert matrix_transcription_issues(candidate) == []


def test_scaled_matrix_product_is_not_reported_as_transcription_error():
    candidate = r"""
    P=\frac{1}{2}\begin{bmatrix}1&1&0\\1&1&0\\0&0&0\end{bmatrix},
    Pv=\frac{1}{2}\begin{bmatrix}1&1&0\\1&1&0\\0&0&0\end{bmatrix}
    \begin{bmatrix}1\\2\\-1\end{bmatrix}
    =\begin{bmatrix}3/2\\3/2\\0\end{bmatrix}.
    """

    assert matrix_transcription_issues(candidate) == []


def test_case_8_localizes_row_column_error_instead_of_invalidating_inverse():
    question = r"C=\begin{bmatrix}3&2&1\\1&3&1\\2&2&3\end{bmatrix}."
    candidate = r"""
    C^{-1}=\frac{1}{15}
    \begin{bmatrix}7&-4&-1\\-1&7&-2\\-4&-2&7\end{bmatrix}.
    计算 CC^{-1} 的 (1,1) 元素为 4/5，与预期 1 不符。
    矛盾点：伴随矩阵有误。
    """

    issues = matrix_relation_issues(question, candidate)

    assert len(issues) == 1
    assert issues[0].quote == "与预期 1 不符"
    assert "行当成了列" in issues[0].problem
    assert "CC^{-1}=I" in issues[0].correction


def test_case_8_rejects_transposed_inverse_directly():
    question = "C=[[3,2,1],[1,3,1],[2,2,3]]."
    candidate = "C=[[3,2,1],[1,3,1],[2,2,3]]. C^(-1)=(1/15)*[[7,-1,-4],[-4,7,-2],[-1,-2,7]]."

    issues = matrix_relation_issues(question, candidate)

    assert len(issues) == 1
    assert "不满足" in issues[0].problem
    assert r"\frac{7}{15}" in issues[0].correction


def test_wrong_inverse_is_rewritten_exactly_after_model_repairs_fail():
    question = "C=[[3,2,1],[1,3,1],[2,2,3]]."
    candidate = "C=[[3,2,1],[1,3,1],[2,2,3]]. C^(-1)=(1/15)×[[7,-4,-1],[-4,7,-2],[-2,-2,7]]."

    corrected, notes = apply_deterministic_corrections(question, candidate)

    assert notes
    assert "[[7,-4,-1],[-1,7,-2],[-4,-2,7]]" in corrected
    assert matrix_relation_issues(question, corrected) == []


def test_wrong_complex_projection_is_atomically_rewritten():
    question = (
        "In C^3 let u=(1,i,1)^T and v=(2,1-i,i)^T. "
        "Construct the orthogonal projector and compute r."
    )
    candidate = "u^dagger v=3, Pv=u, and therefore r=3/7."

    corrected, notes = apply_deterministic_corrections(question, candidate)

    assert notes
    assert r"r=\frac{1}{21}" in corrected
    assert "r=3/7" not in corrected
    assert projection_ratio_issues(question, corrected) == []


def test_repeated_stale_inverse_is_rewritten_and_explicitly_invalidated():
    question = "C=[[3,2,1],[1,3,1],[2,2,3]]."
    wrong = "[[7,-1,-4],[-1,7,2],[-4,2,5]]"
    candidate = f"C⁻¹=(1/15){wrong}. Corrected adjugate: {wrong}."

    corrected, notes = apply_deterministic_corrections(question, candidate)

    assert wrong not in corrected
    assert "旧矩阵及其验证叙述作废" in corrected
    assert notes


def test_case_10_rejects_synthesized_matrix_that_regresses_from_vu():
    question = r"""
    U=\frac{1}{\sqrt{2}}\begin{bmatrix}1&i\\i&1\end{bmatrix},
    V=\begin{bmatrix}1&0\\0&i\end{bmatrix}. Compute W=VU.
    """
    candidate = r"""
    W=VU. W=\begin{bmatrix}
    1/\sqrt{2}&-i/\sqrt{2}\\i/\sqrt{2}&1/\sqrt{2}
    \end{bmatrix}.
    """

    issues = matrix_relation_issues(question, candidate)

    assert len(issues) == 1
    assert issues[0].severity is Severity.MAJOR
    assert "W=VU" in issues[0].problem


def test_plain_named_matrices_with_scalar_and_equality_chain_use_final_value():
    question = "Let U=(1/sqrt(2))*[[1,i],[i,1]], V=[[1,0],[0,i]]. Compute W=VU."
    candidate = "W = (1/sqrt(2))*[[1,0],[0,i]][[1,i],[i,1]] = (1/sqrt(2))*[[1,i],[-1,i]]."

    assert matrix_relation_issues(question, candidate) == []


def test_named_product_is_not_overwritten_by_later_action_on_same_line():
    question = "U=(1/sqrt(2))*[[1,i],[i,1]], V=[[1,0],[0,i]]. Compute W=VU."
    candidate = "W=(1/sqrt(2))*[[1,i],[-1,i]]. Then W psi=(1/2)*[[1+i],[-1+i]]."

    assert matrix_relation_issues(question, candidate) == []


def test_case_10_checks_intermediate_state_scaling():
    question = (
        "U=(1/sqrt(2))*[[1,i],[i,1]], V=[[1,0],[0,i]], and "
        "psi=(1/sqrt(2))*(1,1)^T. Compute phi=VU psi."
    )
    wrong = r"U\psi=(1/sqrt(2))*[[1+i],[1+i]]."
    correct = r"U\psi=(1/2)*[[1+i],[1+i]]."

    assert state_chain_issues(question, wrong)
    assert state_chain_issues(question, correct) == []


def test_case_10_wrong_matrix_and_determinant_are_atomically_rewritten():
    question = (
        "Let U=(1/sqrt(2))*[[1,i],[i,1]], V=[[1,0],[0,i]], and "
        "psi=(1/sqrt(2))*(1,1)^T. Compute W=VU and phi=VU psi."
    )
    candidate = "W=(1/sqrt(2))*[[1,i],[i,-1]]. phi=(1/2)*(1+i,i*(1+i))^T and det(W)=0."

    corrected, notes = apply_deterministic_corrections(question, candidate)

    assert notes
    assert r"\det(W)=i=\det(V)\det(U)" in corrected
    assert "det(W)=0" not in corrected
    assert deterministic_candidate_issues(question, corrected) == []


def test_case_10_rejects_invalid_factoring_of_final_complex_state():
    question = (
        "U=(1/sqrt(2))*[[1,i],[i,1]], V=[[1,0],[0,i]], and "
        "psi=(1/sqrt(2))*(1,1)^T. Compute phi=VU psi."
    )
    wrong = r"\phi=((1+i)/2)*[[1],[-1]]."
    correct = r"\phi=(1/2)*[[1+i],[-1+i]]."

    assert final_state_issues(question, wrong)
    assert final_state_issues(question, correct) == []


def test_plain_bracket_named_matrix_product_is_checked():
    question = "Let A=[[1,2],[0,1]] and B=[[1,0],[1,1]]. Compute C=AB."
    wrong = "C=AB. C=[[3,2],[2,1]]."
    correct = "C=AB. C=[[3,2],[1,1]]."

    issues = matrix_relation_issues(question, wrong)

    assert len(issues) == 1
    assert issues[0].quote == "[[3,2],[2,1]]"
    assert matrix_relation_issues(question, correct) == []


def test_plain_inverse_parentheses_does_not_overwrite_named_product():
    question = "A=[[1,2],[0,1]], B=[[1,0],[1,1]]. Compute C=AB."
    candidate = "C=[[3,2],[1,1]]. C^(-1)=(1/1)×[[1,-2],[-1,3]]."

    assert matrix_relation_issues(question, candidate) == []


def test_unicode_superscript_inverse_is_parsed_and_rewritten():
    question = "C=[[3,2,1],[1,3,1],[2,2,3]]."
    candidate = "C⁻¹=(1/15)[[7,-1,-4],[-1,7,2],[-4,2,5]]."

    corrected, notes = apply_deterministic_corrections(question, candidate)

    assert notes
    assert "[[7,-4,-1],[-1,7,-2],[-4,-2,7]]" in corrected


def test_adjacent_determinants_do_not_cross_contaminate_claims():
    question = "A=[[1,2],[0,1]], B=[[3,0],[0,1]]."
    candidate = "det(A)=1, calculation complete; det(B)=3, calculation complete."

    assert determinant_invariant_issues(question, candidate) == []


def test_case_6_detects_missing_per_eigenvalue_vector_verification():
    question = r"""
    Let H=\begin{bmatrix}2&1&i\\1&2&1\\-i&1&2\end{bmatrix}.
    For each eigenvalue, verify at least one corresponding eigenvector.
    """
    candidate = r"""
    The eigenvalues are $\lambda_1=2$, $\lambda_2=2+\sqrt{3}$, and
    $\lambda_3=2-\sqrt{3}$. Eigenvector Check for $\lambda=2$:
    $v=(1,i,-1)^T$ and $Hv=2v$.
    """

    issues = eigenvector_requirement_issues(question, candidate)

    assert len(issues) == 1
    assert "3 个本征值" in issues[0].problem
    assert "1 个" in issues[0].problem


def test_case_6_substitutes_claimed_eigenvector_instead_of_trusting_prose():
    question = "Let H=[[2,1,i],[1,2,1],[-i,1,2]]. For each eigenvalue, verify an eigenvector."
    candidate = r"For $\lambda_1=2$, take $v=(0,0,1)$ and verify $Hv=2v$."

    issues = eigenvector_equation_issues(question, candidate)

    assert len(issues) == 1
    assert "Hv=lambda*v" in issues[0].problem


def test_case_6_parses_bold_subscripted_vectors_and_wrong_determinant():
    question = "Let H=[[2,1,i],[1,2,1],[-i,1,2]]. For each eigenvalue, verify an eigenvector."
    candidate = (
        r"The determinant $\det(H)=6$ matches the spectrum. "
        r"For $\lambda_1=2$, $\mathbf{v}_1=(0,0,1)$ and $Hv=2v$."
    )

    assert determinant_invariant_issues(question, candidate)
    assert eigenvector_equation_issues(question, candidate)

    corrected, notes = apply_deterministic_corrections(question, candidate)
    assert "旧本征对" in corrected
    assert r"\det(H)=2" in corrected
    assert notes


def test_case_6_rejects_complex_spectrum_and_vague_eigenvectors():
    question = (
        "Let H=[[2,1,i],[1,2,1],[-i,1,2]]. Compute each eigenvalue and "
        "for each eigenvalue verify an eigenvector."
    )
    candidate = (
        r"$\lambda_1=2$, $\lambda_2=2+\sqrt{3+2i}$, "
        r"$\lambda_3=2-\sqrt{3+2i}$. "
        r"For the first, $v_1=[-1,-i,1]$. The others have non-zero vectors."
    )

    assert eigenvalue_spectrum_issues(question, candidate)
    assert eigenvector_requirement_issues(question, candidate)


def test_exact_spectral_appendix_counts_matrix_form_vectors():
    question = "H=[[2,1,i],[1,2,1],[-i,1,2]]. For each eigenvalue verify an eigenvector."
    candidate = "Eigenvectors: " + "; ".join(
        [
            r"$\lambda=2, v=\left[\begin{matrix}-1\\-i\\1\end{matrix}\right]$",
            r"$\lambda=2-\sqrt{3}, v=\left[\begin{matrix}1\\2\\3\end{matrix}\right]$",
            r"$\lambda=2+\sqrt{3}, v=\left[\begin{matrix}3\\2\\1\end{matrix}\right]$",
        ]
    )

    assert eigenvector_requirement_issues(question, candidate) == []


def test_symbolic_matrix_equations_are_not_treated_as_numeric_transcriptions():
    candidate = r"""
    \begin{bmatrix}1&0\\0&1\end{bmatrix}
    \begin{bmatrix}v_1\\v_2\end{bmatrix}
    =\begin{bmatrix}0\\0\end{bmatrix}
    """

    assert matrix_transcription_issues(candidate) == []


def test_case_7_recomputes_complex_projection_ratio_with_conjugation():
    question = (
        "In C^3 let u=(1,i,1)^T and v=(2,1-i,i)^T. "
        "Construct the orthogonal projection and compute r."
    )

    issues = projection_ratio_issues(question, "The result is r=3/7.")

    assert len(issues) == 1
    assert issues[0].quote == "r=3/7"
    assert "u^†v=1" in issues[0].correction
    assert "r=1/21" in issues[0].correction
    assert projection_ratio_issues(question, "The result is r=1/21.") == []


def test_case_7_rejects_false_nonhermitian_claim_for_hermitian_projector():
    question = "Construct the orthogonal projector P onto span(u)."
    candidate = (
        r"P=(1/3)*[[1,-i,1],[i,1,i],[1,-i,1]]. "
        r"P^\dagger=[[1,i,1],[-i,1,-i],[1,i,1]]/3 \neq P."
    )

    issues = projector_hermitian_issues(question, candidate)

    assert len(issues) == 1
    assert issues[0].severity is Severity.MAJOR


def test_case_7_checks_every_entry_of_displayed_projector():
    question = "Let u=(1,i,1)^T. Construct the orthogonal projector P."
    candidate = "P=(1/3)*[[1,-i,1],[i,1,-i],[1,-i,1]]."

    issues = projector_matrix_issues(question, candidate)

    assert len(issues) == 1
    assert "uu^†" in issues[0].problem


def test_case_7_has_exact_deterministic_fallback_when_model_solvers_fail():
    question = (
        "In C^3 let u=(1,i,1)^T and v=(2,1-i,i)^T. "
        "Construct the orthogonal projector and compute r."
    )

    answer = deterministic_fallback_answer(question)

    assert "u^\\dagger v=1" in answer
    assert "r=\\frac{1}{21}" in answer
    assert deterministic_candidate_issues(question, answer) == []


def test_complex_norm_plain_squares_are_rewritten_as_modulus_squares():
    question = "Construct the orthogonal projector for u=(1,i,1)^T."
    candidate = "||u||^2=1^2+(-i)^2+1^2 and ||v||^2=4+(1-i)^2+i^2."

    corrected, notes = apply_deterministic_corrections(question, candidate)

    assert "|-i|^2" in corrected
    assert "|1-i|^2" in corrected
    assert notes
