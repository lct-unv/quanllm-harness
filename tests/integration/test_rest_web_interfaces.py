from __future__ import annotations

import json

from fastapi.testclient import TestClient

from quanllm_harness.config import HarnessSettings
from quanllm_harness.contracts import (
    HarnessEvent,
    HarnessResult,
    RequestPolicy,
    RunStatus,
    Usage,
    VerificationReport,
)
from quanllm_harness.interfaces.rest_api import create_app


def _result() -> HarnessResult:
    return HarnessResult(
        status=RunStatus.VERIFIED,
        answer="测试答案",
        policy=RequestPolicy(depth="simple"),
        verification=VerificationReport(),
        events=(),
        usage=Usage(3, 5),
    )


class FakeService:
    settings = HarnessSettings(api_key="test-key")

    def answer(self, question, *, event_sink=None, cancellation=None):
        if event_sink:
            event_sink(HarnessEvent("agent_started", "主求解", {}))
            event_sink(HarnessEvent("reasoning_delta", "主求解", {"text": "思考"}))
        return _result()

    @staticmethod
    def capabilities():
        return {"test_tool": {"planner_visible": True}}

    @staticmethod
    def execution_graph():
        return [{"name": "route"}]


def test_web_health_and_metadata_endpoints():
    client = TestClient(create_app(service=FakeService()))
    assert client.get("/").status_code == 200
    page = client.get("/").text
    assert "QuanLLM" in page
    assert 'id="elapsed"' in page
    assert client.get("/healthz").json()["configured"] is True
    assert "test_tool" in client.get("/api/v1/capabilities").json()
    assert client.get("/api/v1/graph").json() == [{"name": "route"}]


def test_sync_answer_requires_configured_bearer_token():
    client = TestClient(create_app(service=FakeService(), server_token="secret"))
    assert client.post("/api/v1/answers", json={"question": "Q"}).status_code == 401
    response = client.post(
        "/api/v1/answers",
        json={"question": "Q"},
        headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 200
    assert response.json()["result"]["answer"] == "测试答案"


def test_streaming_answer_emits_events_and_terminal_result():
    client = TestClient(create_app(service=FakeService()))
    with client.stream("POST", "/api/v1/answers/stream", json={"question": "Q"}) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
    blocks = [block for block in body.split("\n\n") if block and not block.startswith(":")]
    event_names = [
        next(line[6:].strip() for line in block.splitlines() if line.startswith("event:"))
        for block in blocks
    ]
    assert event_names == ["request", "harness_event", "harness_event", "result"]
    for block in blocks:
        payload = json.loads(
            next(line[5:].strip() for line in block.splitlines() if line.startswith("data:"))
        )
        assert payload["elapsed_seconds"] >= 0
        assert payload["elapsed"].endswith("秒")
    result_block = blocks[-1]
    data = json.loads(
        next(line[5:].strip() for line in result_block.splitlines() if line.startswith("data:"))
    )
    assert data["result"]["usage"] == {"prompt_tokens": 3, "completion_tokens": 5}
