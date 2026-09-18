from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from ..agents import prompts
from ..agents.runtime import AgentRuntime
from ..contracts import Claim, Requirement
from .json_request import StructuredResponseError

_WS_PATTERN = re.compile(r"\s+")
# Keep letters/digits, Latin Extended-A (e.g. hbar \u0127), Greek and math symbols.
_KEEP_PATTERN = re.compile(
    r"[^0-9A-Za-z\u0100-\u017f\u0370-\u03ff\u2190-\u21ff"
    r"\u2200-\u2219\u221b-\u22ff\u4e00-\u9fff]"
)
_LATEX_NOISE = re.compile(
    r"\\(?:left|right|text|mathrm|operatorname|displaystyle|quad|qquad|"
    r"frac|sqrt|hbar|kappa|alpha|beta|gamma|delta|lambda|mu|nu|omega|pi|"
    r"theta|rho|sigma|tau|phi|psi|partial|sum|int|infty|pm|cdot|times|approx|"
    r"neq|leq|geq|rightarrow|leftarrow|Big|big|Bigg|bigg|,|;|:|\!|\~|\s)"
)


# Conservative Greek-letter and physics-symbol transliteration used only by the
# fuzzy fallback, so "psi" vs "ψ" or "hbar" vs "ℏ" still count as locatable.
# ASCII "k" is intentionally NOT mapped to "kappa", keeping "κ" and "k" distinct.
_GREEK_TO_ASCII = str.maketrans(
    {
        "ψ": "psi",
        "Ψ": "Psi",
        "κ": "kappa",
        "Κ": "Kappa",
        "ħ": "hbar",
        "Ħ": "Hbar",
        "α": "alpha",
        "Α": "Alpha",
        "β": "beta",
        "Β": "Beta",
        "γ": "gamma",
        "Γ": "Gamma",
        "δ": "delta",
        "Δ": "Delta",
        "ε": "epsilon",
        "Ε": "Epsilon",
        "λ": "lambda",
        "Λ": "Lambda",
        "μ": "mu",
        "Μ": "Mu",
        "ν": "nu",
        "Ν": "Nu",
        "ξ": "xi",
        "Ξ": "Xi",
        "π": "pi",
        "Π": "Pi",
        "ρ": "rho",
        "Ρ": "Rho",
        "σ": "sigma",
        "Σ": "Sigma",
        "ς": "sigma",
        "τ": "tau",
        "Τ": "Tau",
        "φ": "phi",
        "Φ": "Phi",
        "χ": "chi",
        "Χ": "Chi",
        "ω": "omega",
        "Ω": "Omega",
        "η": "eta",
        "Η": "Eta",
        "θ": "theta",
        "Θ": "Theta",
        "ι": "iota",
        "Ι": "Iota",
        "ζ": "zeta",
        "Ζ": "Zeta",
        "ℏ": "hbar",
    }
)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.translate(_GREEK_TO_ASCII)
    text = _LATEX_NOISE.sub(" ", text)
    text = text.replace("\\", " ")
    return _WS_PATTERN.sub("", text)


def contains_quote(quote: str, source: str) -> bool:
    """Locate ``quote`` inside ``source`` with progressively tolerant matching.

    Exact and whitespace-collapsed matches are tried first; the final fallback
    strips LaTeX/markdown noise, punctuation, unicode variants and case, so a
    quote that differs only in formatting still counts as locatable. This keeps
    the structured protocol from failing merely because of cosmetic differences.
    """
    if not quote.strip():
        return False
    if quote.strip() in source:
        return True
    if "".join(quote.split()) in "".join(source.split()):
        return True
    needle = _KEEP_PATTERN.sub("", _normalize(quote)).casefold()
    haystack = _KEEP_PATTERN.sub("", _normalize(source)).casefold()
    return bool(needle and haystack and needle in haystack)


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
            allowed_kinds = {
                "definition",
                "equation",
                "derivation",
                "condition",
                "interpretation",
                "conclusion",
            }
            claims: list[Claim] = []
            seen_ids: set[str] = set()
            for index, item in enumerate(raw_claims, 1):
                if not isinstance(item, dict):
                    continue
                claim_id = str(item.get("id") or f"C-{index:03d}")
                quote = str(item.get("quote") or "").strip()
                kind = str(item.get("kind") or "")
                importance = str(item.get("importance") or "")
                if kind not in allowed_kinds or importance not in {"major", "minor"}:
                    continue
                # A single unlocatable or duplicate claim must not invalidate the
                # whole extraction: keep the claims that can be verified and let
                # the mathematical verifier run on them.
                if claim_id in seen_ids or not contains_quote(quote, candidate):
                    continue
                seen_ids.add(claim_id)
                claims.append(Claim(claim_id, quote, kind, importance))
            if not claims:
                missing = [
                    str(item.get("quote"))[:60]
                    for item in raw_claims
                    if isinstance(item, dict) and str(item.get("quote") or "").strip()
                ]
                raise StructuredResponseError(
                    "所有断言的引文均无法在候选答案中定位："
                    + (" | ".join(missing[:5]) if missing else "（claims 为空或全部非法）")
                )
            requirements: list[Requirement] = []
            seen_ids.clear()
            for index, item in enumerate(raw_requirements, 1):
                if not isinstance(item, dict):
                    continue
                requirement_id = str(item.get("id") or f"R-{index:03d}")
                quote = str(item.get("quote") or "").strip()
                if requirement_id in seen_ids or not contains_quote(quote, question):
                    continue
                seen_ids.add(requirement_id)
                requirements.append(Requirement(requirement_id, quote))
            if not requirements:
                # Best effort: keep the model's requirement texts when none can be
                # located verbatim so the requirements verifier still has content.
                requirements = [
                    Requirement(
                        str(item.get("id") or f"R-{index:03d}"),
                        str(item.get("quote") or "").strip(),
                    )
                    for index, item in enumerate(raw_requirements, 1)
                    if isinstance(item, dict) and str(item.get("quote") or "").strip()
                ]
            return claims, requirements

        return self.runtime.json(
            prompts.CLAIM_EXTRACTOR_PROMPT,
            "【原始问题】\n" + question + "\n\n【候选答案】\n" + candidate,
            stage="断言提取",
            validator=validate,
        )
