from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageProfile:
    """Generation controls applied to one model-request family."""

    temperature: float = 0.6
    max_tokens: int = 16_384
    timeout_seconds: float = 180.0

    def validate(self, name: str) -> None:
        if not 0 <= self.temperature <= 2:
            raise ValueError(f"{name}.temperature 必须位于 [0, 2]")
        if self.max_tokens <= 0 or self.timeout_seconds <= 0:
            raise ValueError(f"{name} 的 token 和超时必须为正数")
