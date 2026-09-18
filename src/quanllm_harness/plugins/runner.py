from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from threading import Event, Thread
from typing import Any


class SubprocessPluginError(RuntimeError):
    pass


def run_subprocess_tool(
    command: Sequence[str],
    *,
    plugin: str,
    tool: str,
    arguments: Mapping[str, Any],
    timeout_seconds: float,
    max_output_bytes: int,
) -> Any:
    """Run one JSON request without a shell and enforce bounded output/time."""

    if not command or not all(isinstance(part, str) and part for part in command):
        raise SubprocessPluginError("插件子进程命令非法")
    request = json.dumps(
        {"protocol": 1, "plugin": plugin, "tool": tool, "arguments": dict(arguments)},
        ensure_ascii=False,
    ).encode("utf-8")
    if len(request) > max_output_bytes:
        raise SubprocessPluginError("插件请求超过配置的字节上限")
    allowed_environment = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "PATHEXT", "SYSTEMROOT", "TMP", "TEMP", "LANG", "LC_ALL"}
    }
    process = subprocess.Popen(
        list(command),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=allowed_environment,
    )
    stdout = bytearray()
    stderr_bytes = bytearray()
    output_exceeded = Event()

    def drain(stream: Any, destination: bytearray) -> None:
        while True:
            chunk = stream.read(65_536)
            if not chunk:
                return
            remaining = max_output_bytes + 1 - len(destination)
            if remaining > 0:
                destination.extend(chunk[:remaining])
            if len(destination) > max_output_bytes:
                output_exceeded.set()
                process.kill()
                return

    readers = (
        Thread(target=drain, args=(process.stdout, stdout), daemon=True),
        Thread(target=drain, args=(process.stderr, stderr_bytes), daemon=True),
    )
    for reader in readers:
        reader.start()
    process_stdin = process.stdin
    assert process_stdin is not None

    def send_request() -> None:
        try:
            process_stdin.write(request)
        except BrokenPipeError:
            pass
        finally:
            process_stdin.close()

    writer = Thread(target=send_request, daemon=True)
    writer.start()
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.wait()
        raise SubprocessPluginError(f"插件调用超过 {timeout_seconds:g} 秒") from exc
    finally:
        writer.join()
        for reader in readers:
            reader.join()
    if output_exceeded.is_set():
        raise SubprocessPluginError("插件输出超过配置的字节上限")
    stderr = bytes(stderr_bytes).decode("utf-8", errors="replace").strip()
    if process.returncode != 0:
        detail = stderr[:500] if stderr else f"exit={process.returncode}"
        raise SubprocessPluginError(f"插件子进程失败：{detail}")
    try:
        payload = json.loads(bytes(stdout).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SubprocessPluginError("插件子进程未返回有效 UTF-8 JSON") from exc
    if not isinstance(payload, dict) or payload.get("protocol") != 1:
        raise SubprocessPluginError("插件子进程响应协议不兼容")
    if payload.get("ok") is not True:
        raise SubprocessPluginError(str(payload.get("error") or "插件返回失败"))
    return payload.get("result")
