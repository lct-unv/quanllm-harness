from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..agents import prompts
from ..agents.runtime import AgentRuntime
from ..contracts import Claim, Requirement
from .json_request import StructuredResponseError


def contains_quote(quote: str, source: str) -> bool:
    if not quote.strip():
        return False
    return quote.strip() in source or "".join(quote.split()) in "".join(source.split())


@dataclass
class ClaimExtractionProtocol:
    runtime: AgentRuntime

    def extract(
        self,
        question: str,
        candidate: str,
    ) -> tuple[list[Claim], list[Requirement]]:
        def validate(data: dict[str, Any]):
            raw_claims = data.get("claims")
            raw_requirements = data.get("requirements")
            if not isinstance(raw_claims, list) or not raw_claims:
                raise StructuredResponseError("断言提取没有返回非空 claims")
            if not isinstance(raw_requirements, list):
                raise StructuredResponseError("requirements 不是数组")
            claims: list[Claim] = []
            seen_ids: set[str] = set()
            allowed_kinds = {
                "definition",
                "equation",
                "derivation",
                "condition",
                "interpretation",
                "conclusion",
            }
            for index, item in enumerate(raw_claims, 1):
                if not isinstance(item, dict):
                    raise StructuredResponseError(f"第 {index} 个断言不是对象")
                claim_id = str(item.get("id") or f"C-{index:03d}")
                quote = str(item.get("quote") or "").strip()
                kind = str(item.get("kind") or "")
                importance = str(item.get("importance") or "")
                if kind not in allowed_kinds or importance not in {"major", "minor"}:
                    raise StructuredResponseError(f"断言 {claim_id} 枚举非法")
                if claim_id in seen_ids or not contains_quote(quote, candidate):
                    raise StructuredResponseError(f"断言 {claim_id} 重复或引文无法定位")
                seen_ids.add(claim_id)
                claims.append(Claim(claim_id, quote, kind, importance))
            requirements: list[Requirement] = []
            seen_ids.clear()
            for index, item in enumerate(raw_requirements, 1):
                if not isinstance(item, dict):
                    raise StructuredResponseError(f"第 {index} 个用户要求不是对象")
                requirement_id = str(item.get("id") or f"R-{index:03d}")
                quote = str(item.get("quote") or "").strip()
                if requirement_id in seen_ids or not contains_quote(quote, question):
                    raise StructuredResponseError(f"要求 {requirement_id} 重复或引文无法定位")
                seen_ids.add(requirement_id)
                requirements.append(Requirement(requirement_id, quote))
            return claims, requirements

        return self.runtime.json(
            prompts.CLAIM_EXTRACTOR_PROMPT,
            "【原始问题】\n" + question + "\n\n【候选答案】\n" + candidate,
            stage="断言提取",
            validator=validate,
        )
