from __future__ import annotations


def format_elapsed(seconds: float) -> str:
    """Format a non-negative duration for all user-facing interfaces."""

    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}小时{minutes:02d}分钟{seconds:02d}秒"


__all__ = ["format_elapsed"]
