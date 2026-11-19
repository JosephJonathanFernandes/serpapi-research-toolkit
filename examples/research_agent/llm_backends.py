"""
Chat-model backends for the Tier 3 (LLM) stance classifier.

Anything with ``.invoke(prompt)`` returning an object with ``.content`` works,
so every LangChain chat model can be passed in directly. This module adds a
dependency-free adapter for a local `Ollama <https://ollama.com>`_ server, so
Tier 3 runs with no API key and no extra pip installs.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "mistral"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


@dataclass
class _Reply:
    content: str


class OllamaChat:
    """Minimal ``.invoke(prompt)`` client for Ollama's ``/api/chat`` endpoint.

    Uses temperature 0 and Ollama's JSON mode, which suits the stance prompt.
    """

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: float = 300.0,
        json_mode: bool = True,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.json_mode = json_mode

    def invoke(self, prompt: Any) -> _Reply:
        payload: dict = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0, "seed": 0},
            "messages": [{"role": "user", "content": str(prompt)}],
        }
        if self.json_mode:
            payload["format"] = "json"
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return _Reply(content=body.get("message", {}).get("content", ""))

    def __repr__(self) -> str:  # shown in eval output
        return f"OllamaChat(model={self.model!r})"


def ollama_available(base_url: str = DEFAULT_OLLAMA_URL, timeout: float = 2.0) -> bool:
    """True when an Ollama server answers at *base_url*."""
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=timeout):
            return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def default_chat_model(model_name: Optional[str] = None) -> Any:
    """Pick a chat model: OpenAI if ``OPENAI_API_KEY`` is set, else local Ollama.

    ``OLLAMA_MODEL`` / ``OLLAMA_URL`` override the Ollama defaults.
    Raises ``RuntimeError`` with setup instructions when neither is available.
    """
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError("OPENAI_API_KEY is set but langchain-openai is missing: pip install langchain-openai") from exc
        return ChatOpenAI(model=model_name or DEFAULT_OPENAI_MODEL, temperature=0)

    base_url = os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL)
    if ollama_available(base_url):
        return OllamaChat(model=model_name or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL), base_url=base_url)

    raise RuntimeError(
        "No LLM available for Tier 3. Either set OPENAI_API_KEY (and pip install langchain-openai), "
        "or install Ollama (https://ollama.com), run `ollama pull mistral`, and keep it running."
    )
