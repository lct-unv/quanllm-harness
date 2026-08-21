"""Backward-compatible imports; new code should use :mod:`quanllm_harness.providers`."""

from .protocols.json_request import (
    StructuredResponseError,
    parse_json_object,
)
from .protocols.json_request import (
    decode_argument_objects as _decode_argument_objects,
)
from .providers import OpenAIQuanLLMProvider, ProviderError, QuanLLMProvider

__all__ = [
    "OpenAIQuanLLMProvider",
    "ProviderError",
    "QuanLLMProvider",
    "StructuredResponseError",
    "_decode_argument_objects",
    "parse_json_object",
]
