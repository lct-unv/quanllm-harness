from types import SimpleNamespace

from quanllm_harness import HarnessSettings, OpenAIQuanLLMProvider
from quanllm_harness.provider import _decode_argument_objects


class FakeCompletions:
    def __init__(self, owner):
        self.owner = owner

    def create(self, **kwargs):
        self.owner.kwargs.append(kwargs)
        delta = SimpleNamespace(content='{"ok":true}', reasoning_content=None, tool_calls=None)
        choice = SimpleNamespace(delta=delta, finish_reason="stop")
        return [SimpleNamespace(choices=[choice], usage=None)]


class FakeClient:
    def __init__(self):
        self.kwargs = []
        self.options = []
        self.chat = SimpleNamespace(completions=FakeCompletions(self))

    def with_options(self, **kwargs):
        self.options.append(kwargs)
        return self


def test_v2_json_mode_forces_thinking_off():
    client = FakeClient()
    provider = OpenAIQuanLLMProvider(HarnessSettings(), client=client)
    provider.complete([{"role": "user", "content": "x"}], stage="json", structured=True)
    provider.complete([{"role": "user", "content": "x"}], stage="reason", structured=False)

    structured, reasoning = client.kwargs
    assert structured["extra_body"] == {"enable_thinking": False}
    assert structured["response_format"] == {"type": "json_object"}
    assert reasoning["extra_body"] == {"enable_thinking": True}
    assert "response_format" not in reasoning
    assert client.options == [
        {"timeout": 90.0, "max_retries": 0},
        {"timeout": 180.0, "max_retries": 0},
    ]


def test_gateway_concatenated_tool_arguments_are_split_safely():
    assert _decode_argument_objects('{"x":1}{"x":2}') == [{"x": 1}, {"x": 2}]
