"""Stage prompts, structured replies, and raw model outputs."""
from __future__ import annotations

import os

from ..common import canonical, require
from ..prompt_builder import compact_payload
from ..prompt_loader import load_prompt
from .chat import ChatLLM
from .config import load_llm_config
from .parsing import extract_json, extract_verification_response
from .responses import ResponsesLLM


class LoggedLLM:
    def __init__(self, model=None, timeout=None, call_sink=None, *, config=None, api_mode=None):
        self.api_mode = api_mode or os.getenv("DOCQA_LLM_API_MODE") or "chat"
        require(self.api_mode in {"chat", "responses"}, "API mode must be chat or responses")
        config = config or load_llm_config(model, timeout_seconds=timeout)
        client_class = ResponsesLLM if self.api_mode == "responses" else ChatLLM
        self.client = client_class(config)
        require(self.client.enabled, "LLM credentials are not configured")
        self.calls = []
        self.call_sink = call_sink or (lambda call: None)

    def ask(self, stage, instructions, payload, max_tokens=3000):
        payload = compact_payload(stage, payload)
        prefix = "component_" + (stage if stage in {"answer", "router", "verify"} else "default")
        prompt = load_prompt(prefix) + "\n" + instructions + "\nINPUT_DATA\n" + canonical(payload)
        call = dict(stage=stage, prompt=prompt, max_tokens=max_tokens, temperature=0.0)
        try:
            raw = self.client.chat(prompt, temperature=0.0, max_tokens=max_tokens)
            call["raw_response"] = raw
            result = extract_verification_response(raw) if stage == "verify" else extract_json(raw)
            require(isinstance(result, dict) or (stage == "verify" and result == "PASS"),
                    "Expected JSON object or verification PASS")
            call["parsed"] = result
            return result
        except Exception as exc:
            call["error"] = str(exc)
            raise
        finally:
            self.calls.append(call)
            self.call_sink(call)
