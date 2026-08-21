from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Thread
from time import monotonic
from typing import Any
from uuid import uuid4

from ... import __version__
from ...config import HarnessSettings
from ...contracts import HarnessEvent
from ...orchestration import CancellationToken, RunCancelled
from ..service import HarnessService
from ..timing import format_elapsed
from .schemas import AnswerRequest, AnswerResponse, HealthResponse

try:
    from fastapi import Depends, FastAPI, Header, HTTPException, Request
    from fastapi.responses import FileResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
except ImportError as exc:  # pragma: no cover - exercised by packaging smoke tests
    raise RuntimeError("REST API 默认依赖缺失，请重新安装 quanllm-harness") from exc

LOGGER = logging.getLogger(__name__)


def _event_payload(event: HarnessEvent, elapsed_seconds: float) -> dict[str, Any]:
    return {
        "kind": event.kind,
        "stage": event.stage,
        "payload": dict(event.payload),
        "elapsed_seconds": elapsed_seconds,
        "elapsed": format_elapsed(elapsed_seconds),
    }


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def create_app(
    *,
    settings: HarnessSettings | None = None,
    service: HarnessService | None = None,
    server_token: str | None = None,
):
    active_service = service or HarnessService(settings or HarnessSettings.from_api_key_file())
    expected_token = (
        server_token if server_token is not None else os.environ.get("QUANLLM_SERVER_TOKEN", "")
    )

    async def authorize(authorization: str | None = Header(default=None)) -> None:
        if not expected_token:
            return
        if authorization != f"Bearer {expected_token}":
            raise HTTPException(status_code=401, detail="Unauthorized")

    app = FastAPI(
        title="QuanLLM Harness API",
        version=__version__,
        description="Verified QuanLLM-v2.0 quantum-mechanics answer service",
    )
    static_root = Path(__file__).parents[1] / "web" / "static"
    app.mount("/assets", StaticFiles(directory=static_root), name="assets")

    @app.get("/", include_in_schema=False)
    async def web_ui():
        return FileResponse(static_root / "index.html")

    @app.get("/healthz", response_model=HealthResponse)
    async def health() -> HealthResponse:
        configured = bool(active_service.settings.api_key)
        return HealthResponse(
            status="ok" if configured else "unconfigured",
            model=active_service.settings.model,
            configured=configured,
            version=__version__,
        )

    @app.get("/api/v1/capabilities")
    async def capabilities() -> dict[str, object]:
        return active_service.capabilities()

    @app.get("/api/v1/graph")
    async def graph() -> list[dict[str, object]]:
        return active_service.execution_graph()

    def ensure_configured() -> None:
        if not active_service.settings.api_key:
            raise HTTPException(status_code=503, detail="APIKEY is not configured")

    @app.post(
        "/api/v1/answers",
        response_model=AnswerResponse,
        dependencies=[Depends(authorize)],
    )
    async def answer(payload: AnswerRequest) -> AnswerResponse:
        ensure_configured()
        request_id = uuid4().hex
        try:
            result = await asyncio.to_thread(active_service.answer, payload.question.strip())
        except Exception as exc:
            LOGGER.exception("Harness request %s failed", request_id)
            raise HTTPException(status_code=502, detail="Harness execution failed") from exc
        return AnswerResponse(request_id=request_id, result=result.to_dict())

    @app.post(
        "/api/v1/answers/stream",
        dependencies=[Depends(authorize)],
        response_class=StreamingResponse,
        responses={
            200: {
                "description": "Server-sent harness events followed by a terminal result.",
                "content": {"text/event-stream": {}},
            }
        },
    )
    async def answer_stream(payload: AnswerRequest, request: Request):
        ensure_configured()
        request_id = uuid4().hex
        started_at = monotonic()
        cancellation = CancellationToken()
        queue: Queue[tuple[str, dict[str, Any]]] = Queue(maxsize=512)

        def enqueue(event: str, data: dict[str, Any]) -> None:
            elapsed_seconds = monotonic() - started_at
            timed_data = {
                **data,
                "elapsed_seconds": elapsed_seconds,
                "elapsed": format_elapsed(elapsed_seconds),
            }
            while not cancellation.cancelled:
                try:
                    queue.put((event, timed_data), timeout=0.25)
                    return
                except Full:
                    continue

        def event_sink(event: HarnessEvent) -> None:
            elapsed_seconds = monotonic() - started_at
            enqueue("harness_event", _event_payload(event, elapsed_seconds))

        def worker() -> None:
            try:
                result = active_service.answer(
                    payload.question.strip(),
                    event_sink=event_sink,
                    cancellation=cancellation,
                )
                enqueue("result", {"request_id": request_id, "result": result.to_dict()})
            except RunCancelled:
                enqueue("cancelled", {"request_id": request_id})
            except Exception:
                LOGGER.exception("Streaming Harness request %s failed", request_id)
                enqueue(
                    "error",
                    {"request_id": request_id, "message": "Harness execution failed"},
                )

        Thread(target=worker, name=f"quanllm-request-{request_id[:8]}", daemon=True).start()

        def poll() -> tuple[str, dict[str, Any]] | None:
            try:
                return queue.get(timeout=0.25)
            except Empty:
                return None

        async def stream() -> AsyncIterator[str]:
            yield _sse(
                "request",
                {
                    "request_id": request_id,
                    "elapsed_seconds": 0.0,
                    "elapsed": format_elapsed(0),
                },
            )
            terminal = {"result", "error", "cancelled"}
            last_keepalive = monotonic()
            while True:
                if await request.is_disconnected():
                    cancellation.cancel()
                    return
                item = await asyncio.to_thread(poll)
                if item is None:
                    if monotonic() - last_keepalive >= 15:
                        yield ": keepalive\n\n"
                        last_keepalive = monotonic()
                    continue
                event, data = item
                yield _sse(event, data)
                if event in terminal:
                    return

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Request-ID": request_id,
            },
        )

    return app
