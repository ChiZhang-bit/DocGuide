"""Optional Responses transport."""
from __future__ import annotations

from ..contracts import require
from .chat import ChatLLM
from .config import request_parameters
from .transport import request_json


def build_responses_endpoint(base_url):
    base_url = base_url.rstrip("/")
    return base_url if base_url.endswith("/responses") else base_url + "/responses"


def responses_payload(model, prompt, max_tokens, temperature=0.0):
    return dict(model=model, input=[dict(role="user", content=prompt)],
        temperature=temperature, max_output_tokens=max_tokens,
        stream=False, store=False, **request_parameters())


class ResponsesLLM(ChatLLM):
    def __init__(self, config, enabled=True):
        responses_payload(config.model, "", 1)
        super().__init__(config, enabled)

    def response_text(self, body):
        content = "".join(part.get("text", "") for item in body.get("output", [])
            if item.get("type") == "message" for part in item.get("content", [])
            if part.get("type") == "output_text")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM response has no nonempty text content")
        return content

    def chat(self, prompt, temperature=0.0, max_tokens=1200):
        require(self.enabled, "LLM credentials are not configured")
        return request_json(self, build_responses_endpoint(self.config.base_url),
            responses_payload(self.config.model, prompt, max_tokens, temperature))
