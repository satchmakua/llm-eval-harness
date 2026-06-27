"""Hosted provider for the Google Gemini API. Needs GEMINI_API_KEY."""

import os
import time

from ..types import Completion
from ._http import post_with_retries
from .base import Provider


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, base_url="https://generativelanguage.googleapis.com/v1beta",
                 timeout=120, max_retries=3):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_key = os.environ.get("GEMINI_API_KEY")

    def available(self):
        return bool(self.api_key)

    def complete(self, model, messages, options=None):
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        # Gemini takes the system prompt as a separate top-level field and names
        # the assistant role "model" rather than "assistant".
        system = None
        contents = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
                continue
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})

        body = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        gen = {}
        if options:
            if "temperature" in options:
                gen["temperature"] = options["temperature"]
            if "max_tokens" in options:
                gen["maxOutputTokens"] = options["max_tokens"]
        if gen:
            body["generationConfig"] = gen

        started = time.time()
        resp = post_with_retries(
            f"{self.base_url}/models/{model}:generateContent",
            headers={"x-goog-api-key": self.api_key, "content-type": "application/json"},
            json=body,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        data = resp.json()
        usage = data.get("usageMetadata", {})
        return Completion(
            text=_first_text(data),
            model=model,
            provider=self.name,
            prompt_tokens=usage.get("promptTokenCount", 0),
            completion_tokens=usage.get("candidatesTokenCount", 0),
            latency_s=round(time.time() - started, 3),
        )


def _first_text(data):
    """Concatenate the text parts of the first candidate; '' if there are none.

    A safety block or an otherwise empty response yields no candidates, so this
    returns an empty string rather than raising.
    """
    candidates = data.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts)
