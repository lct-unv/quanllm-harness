from __future__ import annotations

import sys
from types import SimpleNamespace

from quanllm_harness.interfaces.rest_api.__main__ import main


def test_server_defaults_to_port_3921(monkeypatch):
    calls = []
    monkeypatch.delenv("QUANLLM_HOST", raising=False)
    monkeypatch.delenv("QUANLLM_PORT", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(run=lambda *args, **kwargs: calls.append((args, kwargs))),
    )

    assert main([]) == 0
    assert calls[0][1]["host"] == "127.0.0.1"
    assert calls[0][1]["port"] == 3921


def test_server_port_can_still_be_overridden(monkeypatch):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(run=lambda *args, **kwargs: calls.append((args, kwargs))),
    )

    assert main(["--port", "8080"]) == 0
    assert calls[0][1]["port"] == 8080
