from __future__ import annotations

from math import isclose, pi, sqrt

from quanllm_harness.tools.numeric_backend import (
    numeric_integrate,
    numeric_root,
    truncation_convergence,
)


def test_gaussian_integral_at_high_precision():
    result = numeric_integrate(
        {
            "expression": "exp(-x**2)",
            "variable": "x",
            "lower": "-inf",
            "upper": "inf",
            "dps": 40,
        }
    )
    assert isclose(float(result["value"]), sqrt(pi), rel_tol=1e-14)


def test_root_returns_small_residual():
    result = numeric_root({"expression": "cos(x)-x", "variable": "x", "initial": [0.7], "dps": 40})
    assert isclose(float(result["root"]), 0.7390851332151607, rel_tol=1e-14)
    assert abs(float(result["residual"])) < 1e-30


def test_truncation_convergence_uses_absolute_and_relative_tolerances():
    result = truncation_convergence(
        {
            "values": [1.0, 1.00000001, 1.000000010001],
            "absolute_tolerance": 1e-10,
            "relative_tolerance": 1e-10,
        }
    )
    assert result["converged"] is True
