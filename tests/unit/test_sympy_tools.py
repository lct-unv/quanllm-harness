from quanllm_harness.tools.sympy_backend import (
    compare_expressions,
    compare_matrices,
    matrix_calculate,
    symbolic_calculate,
)


def test_substitution_detects_inconsistent_general_formula():
    result = symbolic_calculate(
        {
            "operation": "substitute",
            "expression": "6*(n+1)*(n+2)*c**2",
            "symbols": ["n", "c"],
            "substitutions": {"n": 0},
        }
    )
    assert result["expression"] == "12*c**2"
    comparison = compare_expressions(
        {"lhs": result["expression"], "rhs": "3*c**2", "symbols": ["c"]}
    )
    assert comparison == {"equivalent": False, "simplified_difference": "9*c**2"}


def test_pauli_commutator_is_exact():
    result = matrix_calculate(
        {
            "operation": "commutator",
            "matrix": [[0, 1], [1, 0]],
            "other_matrix": [[0, "-I"], ["I", 0]],
        }
    )
    assert result["matrix"] == [["2*I", "0"], ["0", "-2*I"]]
    assert result["normalized_inputs"]["other_matrix"] == [["0", "-I"], ["I", "0"]]


def test_matrix_comparison_is_exact_and_reports_normalized_inputs():
    result = compare_matrices(
        {
            "lhs": [["2*I", 0], [0, "-2*I"]],
            "rhs": [["2*I", 0], [0, "-2*I"]],
        }
    )
    assert result["equivalent"] is True
    assert result["difference"] == [["0", "0"], ["0", "0"]]
    assert result["normalized_inputs"]["lhs"][0][0] == "2*I"
