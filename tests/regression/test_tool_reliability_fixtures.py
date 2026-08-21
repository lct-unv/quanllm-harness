from __future__ import annotations

import json
from pathlib import Path

import mpmath as mp

from quanllm_harness.tools import default_tool_registry


def test_tool_reliability_cases():
    fixture = Path(__file__).parents[1] / "fixtures" / "tool_reliability_cases.json"
    cases = json.loads(fixture.read_text(encoding="utf-8"))
    registry = default_tool_registry(include_plugins=False)

    for case in cases:
        evidence = registry.execute(case["tool"], case["arguments"])
        assert evidence.ok, f"{case['id']}: {evidence.error}"
        if "expected_matrix" in case:
            assert evidence.result["matrix"] == case["expected_matrix"]
            normalized = evidence.result["normalized_inputs"]["other_matrix"]
            assert normalized[0][1] == "-I"
            assert normalized[1][0] == "I"
        else:
            difference = abs(mp.mpf(evidence.result["value"]) - mp.mpf(case["expected_value"]))
            assert difference <= mp.mpf(case["absolute_tolerance"]), case["id"]
