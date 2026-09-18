from __future__ import annotations

import argparse
import os


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve QuanLLM Harness REST API and Web UI")
    parser.add_argument("--host", default=os.environ.get("QUANLLM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("QUANLLM_PORT", "3921")))
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--insecure-no-auth",
        action="store_true",
        help=(
            "显式关闭回答接口的认证（不安全）。仅在内部/本机且端口未暴露公网时使用；"
            "未设置 QUANLLM_SERVER_TOKEN 且未加此标志时，回答接口默认返回 401。"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.insecure_no_auth:
        os.environ["QUANLLM_ALLOW_NO_AUTH"] = "1"
        print(
            "WARNING: --insecure-no-auth is set; answering endpoints run WITHOUT "
            "authentication. Only use on a trusted/internal network.",
            flush=True,
        )
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
