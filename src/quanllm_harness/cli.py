"""Backward-compatible CLI import."""

from .interfaces.cli.app import TerminalEvents, build_parser, main

__all__ = ["TerminalEvents", "build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
