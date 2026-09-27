"""
langchain_serpapi_provider
==========================

A LangChain-compatible provider for SerpApi.

Public API
----------
- ``SerpApiRetriever`` — LangChain BaseRetriever subclass
- ``SerpApiCache``     — disk-based response cache
- ``SerpApiResult``    — normalized single result Pydantic model
- ``NormalizedSearchResponse`` — normalized response container
- ``normalize_response`` — raw-dict → NormalizedSearchResponse
- ``select_engine``    — heuristic engine picker
- ``build_serpapi_tool``  — create a LangChain Tool (fixed engine)
- ``build_auto_engine_tool`` — create a LangChain Tool (auto engine)
"""

from .cache import SerpApiCache
from .retriever import SerpApiRetriever
from .schema import NormalizedSearchResponse, SerpApiResult, normalize_response
from .tool import build_auto_engine_tool, build_serpapi_tool, select_engine

__all__ = [
    "SerpApiRetriever",
    "SerpApiCache",
    "SerpApiResult",
    "NormalizedSearchResponse",
    "normalize_response",
    "select_engine",
    "build_serpapi_tool",
    "build_auto_engine_tool",
]

__version__ = "0.1.0"
