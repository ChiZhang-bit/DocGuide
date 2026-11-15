"""Chat Completions transport."""
from __future__ import annotations

import urllib.request

from ..contracts import require
from .config import request_payload
from .parsing import extract_json
from .transport import build_ssl_context, request_json


def build_chat_endpoint(base_url):
    base_url = base_url.rstrip("/")
    return base_url if base_url.endswith("/chat/completions") else base_url + "/chat/completions"


class ChatLLM:
    def __init__(self, config, enabled=True):
        self.config = config
        self.enabled = enabled and config.enabled
        self.ssl_context = build_ssl_context()
        self.opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=self.ssl_context))

    def response_text(self, body):
        choices = body.get("choices") or []
        if not choices:
            raise ValueError("LLM response has no choices")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM response has no nonempty text content")
        return content

    def chat(self, prompt, temperature=0.0, max_tokens=1200):
        require(self.enabled, "LLM credentials are not configured")
        payload = request_payload(self.config.model, prompt, max_tokens, temperature)
        return request_json(self, build_chat_endpoint(self.config.base_url), payload)

    def chat_json(self, prompt, temperature=0.0, max_tokens=1200):
        if not self.enabled:
            return None
        return extract_json(self.chat(prompt, temperature=temperature, max_tokens=max_tokens))
