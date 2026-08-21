from __future__ import annotations

import json
from dataclasses import dataclass

from ..contracts import Candidate
from . import prompts
from .runtime import AgentRuntime


@dataclass
class SynthesizerAgent:
    runtime: AgentRuntime

    def synthesize(self, question: str, primary: Candidate, independent: Candidate) -> Candidate:
        evidence = [item.__dict__ for item in self.runtime.evidence]
        user = (
            "【原始用户问题】\n"
            + question
            + "\n\n【候选 A】\n"
            + primary.answer
            + "\n\n【候选 B】\n"
            + independent.answer
            + "\n\n【已有工具证据】\n"
            + json.dumps(evidence, ensure_ascii=False, default=list)
        )
        return self.runtime.reason(
            prompts.SYNTHESIZER_PROMPT, user, stage="候选综合", require_tool=False
        )
