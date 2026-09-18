from __future__ import annotations

import pytest

from quanllm_harness.tools.quantum_backends import (
    can_import,
    openfermion_algebra,
    pycommute_algebra,
    qutip_state_check,
)


@pytest.mark.skipif(
    not can_import("qutip"), reason="QuTiP default dependency is unavailable or cannot be imported"
)
def test_qutip_expectation_value():
    result = qutip_state_check(
        {
            "operation": "expectation",
            "state": [1, 0],
            "operator": [[1, 0], [0, -1]],
        }
    )
    assert complex(result["expectation"]) == 1


@pytest.mark.skipif(
    not can_import("pycommute"),
    reason="pycommute default dependency is unavailable or cannot be imported",
)
def test_pycommute_boson_commutator():
    result = pycommute_algebra(
        {
            "operation": "commutator",
            "left": [{"operator": "boson_annihilate", "indices": [0]}],
            "right": [{"operator": "boson_create", "indices": [0]}],
        }
    )
    assert result == {"expression": "1", "zero": False}


@pytest.mark.skipif(
    not can_import("openfermion"),
    reason="OpenFermion default dependency is unavailable or cannot be imported",
)
def test_openfermion_fermion_anticommutator():
    result = openfermion_algebra(
        {
            "statistics": "fermion",
            "operation": "anticommutator",
            "left": [{"actions": [[0, 0]]}],
            "right": [{"actions": [[0, 1]]}],
        }
    )
    assert result == {"expression": "(1+0j) []", "zero": False}
