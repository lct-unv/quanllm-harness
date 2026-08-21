from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .config import HarnessSettings
from .contracts import HarnessResult


def save_run(
    directory: str | Path,
    *,
    question: str,
    result: HarnessResult,
    settings: HarnessSettings,
) -> HarnessResult:
    target_dir = Path(directory).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    run_id = result.run_id or uuid4().hex
    path = target_dir / f"{run_id}.json"
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "configuration": {
            "model": settings.model,
            "reasoning": {
                "temperature": settings.reasoning.temperature,
                "max_tokens": settings.reasoning.max_tokens,
                "timeout_seconds": settings.reasoning.timeout_seconds,
            },
            "structured": {
                "temperature": settings.structured.temperature,
                "max_tokens": settings.structured.max_tokens,
                "timeout_seconds": settings.structured.timeout_seconds,
            },
            "parallel_solvers": settings.parallel_solvers,
            "semantic_verifier_count": settings.semantic_verifier_count,
        },
        "result": result.to_dict(),
    }
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return replace(result, run_id=run_id, record_path=str(path))
