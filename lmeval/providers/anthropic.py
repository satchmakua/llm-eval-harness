"""Hosted provider for the Anthropic Messages API. Needs ANTHROPIC_API_KEY."""

import os
import time

from ..types import Completion
from ._http import post_with_retries
from .base import Provider


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, base_url="https://api.anthropic.com/v1",
                 timeout=120, version="2023-06-01", max_retries=3):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.version = version
        self.max_retries = max_retries
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")

    def available(self):
        return bool(self.api_key)

    def complete(self, model, messages, options=None):
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        # Anthropic takes the system prompt as a separate top-level field.
        system = None
        conversation = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                conversation.append({"role": m["role"], "content": m["content"]})
        body = {
            "model": model,
            "messages": conversation,
            "max_tokens": (options or {}).get("max_tokens", 1024),
        }
        if system:
            body["system"] = system
        if options and "temperature" in options:
            body["temperature"] = options["temperature"]
        started = time.time()
        resp = post_with_retries(
            f"{self.base_url}/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": self.version,
                "content-type": "application/json",
            },
            json=body,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        data = resp.json()
        text = "".join(b.get("text", "") for b in data.get("content", [])
                       if b.get("type") == "text")
        usage = data.get("usage", {})
        return Completion(
            text=text,
            model=model,
            provider=self.name,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            latency_s=round(time.time() - started, 3),
        )
