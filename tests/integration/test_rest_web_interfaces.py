from __future__ import annotations

import asyncio
import json

import httpx

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

    @staticmethod
    def plugins():
        return {"ok": True, "plugins": []}

    @staticmethod
    def close():
        return None


def test_web_health_and_metadata_endpoints():
    async def scenario():
        transport = httpx.ASGITransport(app=create_app(service=FakeService()))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/")
            assert response.status_code == 200
            assert "QuanLLM" in response.text
            assert 'id="elapsed"' in response.text
            assert (await client.get("/healthz")).json()["configured"] is True
            assert "test_tool" in (await client.get("/api/v1/capabilities")).json()
            assert (await client.get("/api/v1/graph")).json() == [{"name": "route"}]
            assert (await client.get("/api/v1/plugins")).json() == {
                "ok": True,
                "plugins": [],
            }

    asyncio.run(scenario())


def test_sync_answer_requires_configured_bearer_token():
    async def scenario():
        app = create_app(service=FakeService(), server_token="secret")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/api/v1/answers", json={"question": "Q"})
            assert response.status_code == 401
            response = await client.post(
                "/api/v1/answers",
                json={"question": "Q"},
                headers={"Authorization": "Bearer secret"},
            )
            assert response.status_code == 200
            assert response.json()["result"]["answer"] == "测试答案"

    asyncio.run(scenario())


def test_streaming_answer_emits_events_and_terminal_result():
    async def scenario():
        app = create_app(service=FakeService(), allow_no_auth=True)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/api/v1/answers/stream", json={"question": "Q"})
            assert response.status_code == 200
            return response.text

    body = asyncio.run(scenario())
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
