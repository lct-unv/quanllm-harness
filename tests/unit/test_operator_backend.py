from __future__ import annotations

import pytest

from quanllm_harness.tools import default_tool_registry
from quanllm_harness.tools.operator_backend import operator_algebra


def terms(*monomials: tuple[str, ...]) -> list[dict[str, object]]:
    return [{"operators": list(monomial)} for monomial in monomials]


def test_canonical_position_momentum_commutator():
    result = operator_algebra(
        {
            "algebra": "canonical",
            "operation": "commutator",
            "left_terms": terms(("x",)),
            "right_terms": terms(("p",)),
        }
    )
    assert result["expression"] == "hbar*I"
    assert result["resolved"] is True
    assert result["zero"] is False


def test_unicode_hbar_coefficient_uses_the_canonical_hbar_symbol():
    result = operator_algebra(
        {
            "algebra": "canonical",
            "operation": "commutator",
            "left_terms": [{"coefficient": "ℏ", "operators": ["x"]}],
            "right_terms": terms(("p",)),
            "scalar_symbols": ["ℏ"],
        }
    )
    assert result["expression"] == "hbar**2*I"


def test_total_angular_momentum_commutes_with_z_component():
    result = operator_algebra(
        {
            "algebra": "angular_momentum",
            "operation": "commutator",
            "left_terms": terms(("Jx", "Jx"), ("Jy", "Jy"), ("Jz", "Jz")),
            "right_terms": terms(("Jz",)),
        }
    )
    assert result["expression"] == "0"
    assert result["zero"] is True
    assert [item["result"] for item in result["derivation_terms"]] == [
        "-hbar*I*Jx*Jy - hbar*I*Jy*Jx",
        "hbar*I*Jx*Jy + hbar*I*Jy*Jx",
        "0",
    ]


@pytest.mark.parametrize(
    ("algebra", "left", "right", "operation", "expected"),
    [
        ("boson_single_mode", "a", "adag", "commutator", "1"),
        ("fermion_single_mode", "c", "cdag", "anticommutator", "1"),
    ],
)
def test_single_mode_creation_annihilation_algebra(algebra, left, right, operation, expected):
    result = operator_algebra(
        {
            "algebra": algebra,
            "operation": operation,
            "left_terms": terms((left,)),
            "right_terms": terms((right,)),
        }
    )
    assert result["expression"] == expected
    assert result["resolved"] is True


def test_operator_from_another_algebra_is_rejected():
    with pytest.raises(ValueError, match="不属于 canonical 代数"):
        operator_algebra(
            {
                "algebra": "canonical",
                "operation": "commutator",
                "left_terms": terms(("x",)),
                "right_terms": terms(("Jz",)),
            }
        )


def test_schema_has_no_labels_or_generic_operator_kinds():
    registry = default_tool_registry(include_plugins=False)
    schema = registry.tools["operator_algebra"].parameters
    serialized = str(schema)
    operator_enum = schema["properties"]["left_terms"]["items"]["properties"]["operators"]["items"][
        "enum"
    ]
    assert "label" not in serialized
    assert "position" not in operator_enum
    assert "momentum" not in operator_enum


def test_operator_tool_is_available_in_default_registry():
    registry = default_tool_registry(include_plugins=False)
    assert "operator_algebra" in registry.tools
    registry.validate_call(
        "operator_algebra",
        {
            "algebra": "angular_momentum",
            "operation": "commutator",
            "left_terms": terms(("Jx",)),
            "right_terms": terms(("Jy",)),
        },
        claim_kind="definition",
    )
