from __future__ import annotations

import json
from pathlib import Path

import pytest

from quanllm_harness.protocols.json_request import (
    decode_argument_objects,
    parse_json_object,
)

CASES = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "gateway_protocol_cases.json").read_text(
        encoding="utf-8"
    )
)


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_recorded_gateway_protocol_cases(case):
    parser = parse_json_object if case["parser"] == "json" else decode_argument_objects
    assert parser(case["input"]) == case["expected"]
