from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from threading import Lock
from time import monotonic

from ... import __version__
from ...config import DEFAULT_API_KEY_PATH, HarnessSettings, StageProfile
from ...contracts import HarnessEvent, HarnessResult, RunStatus
from ...plugins import (
    PluginManager,
    PluginManifest,
    load_plugin_policy,
    set_plugin_enabled,
)
from ..service import HarnessService
from ..timing import format_elapsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QuanLLM-v2.0-qm verified answer harness")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("question", nargs="*", help="待回答问题；省略时从标准输入读取")
    parser.add_argument("--model", default=os.environ.get("QUANLLM_MODEL", "QuanLLM-v2.0-qm"))
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=DEFAULT_API_KEY_PATH,
        help="API Key 文件（默认：当前目录/APIKEY）",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--structured-temperature", type=float, default=0.6)
    parser.add_argument("--run-dir", default=os.environ.get("QUANLLM_RUN_DIR", ""))
    parser.add_argument("--json", action="store_true", help="仅输出机器可读结果 JSON")
    parser.add_argument("--no-parallel", action="store_true", help="顺序运行双 Solver")
    parser.add_argument("--capabilities", action="store_true", help="打印工具能力并退出")
    parser.add_argument("--graph", action="store_true", help="打印默认执行图并退出")
    parser.add_argument(
        "--interactive", action="store_true", help="进入支持 :again/:paste 的会话模式"
    )
    parser.add_argument("--total-timeout", type=float, default=900.0, help="整次运行墙钟上限（秒）")
    return parser


def build_plugin_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quanllm-harness plugins")
    subparsers = parser.add_subparsers(dest="plugin_command", required=True)
    subparsers.add_parser("list", help="列出已发现插件")
    inspect_parser = subparsers.add_parser("inspect", help="查看一个插件")
    inspect_parser.add_argument("name")
    subparsers.add_parser("doctor", help="检查插件兼容性、权限和启动状态")
    validate_parser = subparsers.add_parser("validate", help="校验插件 manifest JSON")
    validate_parser.add_argument("manifest", type=Path)
    enable_parser = subparsers.add_parser("enable", help="将插件加入启用列表")
    enable_parser.add_argument("name")
    disable_parser = subparsers.add_parser("disable", help="禁用插件")
    disable_parser.add_argument("name")
    return parser


def _plugin_command(argv: list[str]) -> int:
    args = build_plugin_parser().parse_args(argv)
    if args.plugin_command in {"enable", "disable"}:
        path = set_plugin_enabled(args.name, args.plugin_command == "enable")
        print(f"插件配置已更新：{path}")
        return 0
    if args.plugin_command == "validate":
        try:
            payload = json.loads(args.manifest.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("manifest 根节点必须是对象")
            manifest = PluginManifest(**payload)
            manifest.validate()
        except Exception as exc:
            print(f"插件 manifest 无效：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(json.dumps({"valid": True, "name": manifest.name}, ensure_ascii=False))
        return 0
    manager = PluginManager.discover(load_plugin_policy())
    try:
        if args.plugin_command == "doctor":
            payload = manager.doctor()
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0 if payload["ok"] else 1
        statuses = manager.statuses()
        if args.plugin_command == "inspect":
            matches = [item for item in statuses if item["name"] == args.name]
            if not matches:
                print(f"未发现插件：{args.name}", file=sys.stderr)
                return 1
            print(json.dumps(matches[0], ensure_ascii=False, indent=2))
            return 0
        print(json.dumps(statuses, ensure_ascii=False, indent=2))
        return 0
    finally:
        manager.shutdown()


class TerminalEvents:
    def __init__(self, *, show_reasoning: bool):
        self.show_reasoning = show_reasoning
        self._reasoning_parts: dict[str, list[str]] = {}
        self._lock = Lock()
        self._started_at = monotonic()

    def _elapsed(self) -> str:
        return format_elapsed(monotonic() - self._started_at)

    def _flush_reasoning(self, stage: str) -> None:
        parts = self._reasoning_parts.pop(stage, None)
        if not parts:
            return
        reasoning = "".join(parts)
        print(f"\n──────── [{self._elapsed()}] {stage} · 原始思维链 ────────", flush=True)
        print(reasoning, end="" if reasoning.endswith("\n") else "\n", flush=True)

    def __call__(self, event: HarnessEvent) -> None:
        with self._lock:
            if event.kind == "reasoning_delta" and self.show_reasoning:
                text = str(event.payload.get("text", ""))
                self._reasoning_parts.setdefault(event.stage, []).append(text)
            elif event.kind in {"agent_finished", "provider_error"}:
                self._flush_reasoning(event.stage)
            elif event.kind == "agent_started":
                print(f"\n[{self._elapsed()}] [{event.stage}]", flush=True)
            elif event.kind == "tool_finished":
                state = "成功" if event.payload.get("ok") else "失败"
                print(f"[{self._elapsed()}] [工具 {event.stage}：{state}]", flush=True)
            elif event.kind == "repair_started":
                print(
                    f"[{self._elapsed()}] [开始第 {event.payload.get('round')} 轮定向修复]",
                    flush=True,
                )
            elif event.kind == "degraded":
                print(
                    f"[{self._elapsed()}] [降级] {event.payload.get('reason', '')}",
                    flush=True,
                )


def _settings_from_args(args: argparse.Namespace) -> HarnessSettings:
    if not args.api_key_file.is_file():
        raise ValueError(f"请将 API Key 写入：{args.api_key_file.resolve()}")
    api_key = HarnessSettings.read_api_key(args.api_key_file)
    return HarnessSettings(
        api_key=api_key,
        model=args.model,
        reasoning=StageProfile(temperature=args.temperature),
        structured=StageProfile(
            temperature=args.structured_temperature,
            max_tokens=8_192,
            timeout_seconds=90,
        ),
        parallel_solvers=not args.no_parallel,
        run_directory=args.run_dir,
        total_timeout_seconds=args.total_timeout,
        plugin_policy=load_plugin_policy(),
        plugin_provider=os.environ.get("QUANLLM_PLUGIN_PROVIDER", ""),
    )


def _render_result(result: HarnessResult, *, as_json: bool) -> int:
    if as_json:
        # Machine-readable output must always be valid UTF-8, independent of the
        # Windows console code page (GBK cannot encode e.g. '²', Greek, math
        # symbols and would crash mid-print). Write raw UTF-8 bytes.
        data = json.dumps(result.to_dict(), ensure_ascii=False, indent=2).encode("utf-8")
        try:
            sys.stdout.buffer.write(data + b"\n")
            sys.stdout.buffer.flush()
        except Exception:
            print(data.decode("utf-8", errors="replace"), flush=True)
    else:
        print(f"\n助手：{result.answer}")
        print(
            f"\n状态：{result.status.value} ｜ 修复 {result.repair_rounds} 轮"
            f" ｜ 输入 {result.usage.prompt_tokens} Token"
            f" ｜ 输出 {result.usage.completion_tokens} Token"
        )
        if result.record_path:
            print(f"运行记录：{result.record_path}")
    if result.status in {RunStatus.VERIFIED, RunStatus.VERIFIED_WITH_INPUT_AMBIGUITY}:
        return 0
    return 2 if result.answer else 1


def _read_paste() -> str:
    print("粘贴多行内容；单独输入 :end 结束：")
    lines: list[str] = []
    while True:
        line = input()
        if line.strip() == ":end":
            return "\n".join(lines).strip()
        lines.append(line)


def _interactive(service: HarnessService) -> int:
    print("QuanLLM Harness CLI；:again 重跑，:paste 多行，:quit 退出。")
    last_question = ""
    while True:
        try:
            value = input("你：").strip()
        except EOFError:
            return 0
        if value in {":quit", ":exit"}:
            return 0
        if value == ":paste":
            value = _read_paste()
        elif value == ":again":
            if not last_question:
                print("还没有可重新运行的问题。")
                continue
            value = last_question
        if not value:
            continue
        last_question = value
        result = service.answer(value, event_sink=TerminalEvents(show_reasoning=True))
        _render_result(result, as_json=False)


def _configure_windows_console_utf8() -> None:
    """Best-effort switch the Windows console to UTF-8 and make Python emit UTF-8.

    Fixes UnicodeEncodeError on the default GBK/936 code page (e.g. '†' / '⟨⟩'
    in tool capabilities) and mojibake for Chinese output. Safe no-op elsewhere.
    """
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            reconfigure = getattr(stream, "reconfigure", None)
            if callable(reconfigure):
                reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    _configure_windows_console_utf8()
    effective_argv = list(argv) if argv is not None else sys.argv[1:]
    if effective_argv[:1] == ["plugins"]:
        return _plugin_command(effective_argv[1:])
    args = build_parser().parse_args(effective_argv)
    if args.capabilities:
        service = HarnessService(HarnessSettings(plugin_policy=load_plugin_policy()))
        try:
            print(json.dumps(service.capabilities(), ensure_ascii=False, indent=2))
        finally:
            service.close()
        return 0
    if args.graph:
        print(json.dumps(HarnessService.execution_graph(), ensure_ascii=False, indent=2))
        return 0
    try:
        service = HarnessService(_settings_from_args(args))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    try:
        if args.interactive:
            return _interactive(service)
        if args.question:
            question = " ".join(args.question).strip()
        elif sys.stdin.isatty():
            question = input("你：").strip()
        else:
            question = sys.stdin.read().strip()
        if not question:
            print("问题不能为空", file=sys.stderr)
            return 1
        sink = None if args.json else TerminalEvents(show_reasoning=True)
        return _render_result(service.answer(question, event_sink=sink), as_json=args.json)
    finally:
        service.close()
