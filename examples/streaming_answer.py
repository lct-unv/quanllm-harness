from __future__ import annotations

import sys

from quanllm_harness import HarnessSettings, create_harness


def stream_event(event) -> None:
    if event.kind == "reasoning_delta":
        print(event.payload.get("text", ""), end="", flush=True)
    elif event.kind == "tool_finished":
        print(
            f"\n[tool:{event.stage} ok={event.payload.get('ok')}]",
            file=sys.stderr,
            flush=True,
        )
    elif event.kind in {"repair_started", "degraded"}:
        print(f"\n[{event.kind}:{dict(event.payload)}]", file=sys.stderr, flush=True)


settings = HarnessSettings.from_env()
result = create_harness(settings, event_sink=stream_event).answer(
    "推导一维有限深势阱偶宇称束缚态方程，并解释边界条件。"
)
print("\n\n" + result.answer)
print(f"status={result.status.value}", file=sys.stderr)
