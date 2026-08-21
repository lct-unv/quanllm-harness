from __future__ import annotations

import argparse
import os


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve QuanLLM Harness REST API and Web UI")
    parser.add_argument("--host", default=os.environ.get("QUANLLM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("QUANLLM_PORT", "3921")))
    parser.add_argument("--reload", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("服务器默认依赖缺失，请重新安装 quanllm-harness") from exc
    uvicorn.run(
        "quanllm_harness.interfaces.rest_api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
