from quanllm_harness import HarnessSettings, OpenAIQuanLLMProvider, QuanLLMHarness


def show_event(event):
    if event.kind in {"agent_started", "agent_finished", "tool_finished", "repair_started"}:
        print(f"[{event.stage}] {event.kind}: {dict(event.payload)}")


settings = HarnessSettings.from_api_key_file()
harness = QuanLLMHarness(
    provider=OpenAIQuanLLMProvider(settings),
    settings=settings,
    event_sink=show_event,
)
result = harness.answer("为什么纯态密度矩阵满足 rho^2=rho？")
print(result.status.value)
print(result.answer)
