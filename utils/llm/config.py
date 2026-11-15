"""Shared configuration for a compatible API service."""
from __future__ import annotations

import math
import os
from urllib.parse import urlsplit

from ..contracts import require
from ..data.records import LLMConfig

API_CONFIG = {
    "model": "",
    "judge_model": "",
    "api_key": "",
    "base_url": "",
    "timeout": 60,
    "max_tokens_field": "max_tokens",
    "extra_body": {},
}


def _text_value(values, name):
    for value in values:
        if value is None:
            continue
        require(isinstance(value, str), f"{name} must be a string")
        if value.strip():
            return value.strip()
    return ""


def get_model_name(model=None, *, judge=False):
    values = [model]
    if judge:
        values.extend([os.getenv("SEMANTIC_EVAL_MODEL"), API_CONFIG["judge_model"]])
    values.extend([os.getenv("DOCQA_LLM_MODEL"), API_CONFIG["model"]])
    return _text_value(values, "model")


def get_request_timeout(value=None):
    if value is None:
        value = os.getenv("DOCQA_LLM_TIMEOUT") or API_CONFIG["timeout"]
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("API timeout must be a positive number") from exc
    require(not isinstance(value, bool) and math.isfinite(timeout) and timeout > 0,
            "API timeout must be a positive number")
    return timeout


def credential(*, judge=False):
    key = _text_value([os.getenv("SEMANTIC_EVAL_API_KEY") if judge else None,
                      os.getenv("DOCQA_LLM_API_KEY"), API_CONFIG["api_key"], os.getenv("OPENAI_API_KEY")], "api_key")
    require(bool(key), "Set api_key in utils/llm/config.py")
    return key


def configured_base_url(base_url=None, *, judge=False):
    endpoint = _text_value([base_url, os.getenv("SEMANTIC_EVAL_BASE_URL") if judge else None,
                           os.getenv("DOCQA_LLM_BASE_URL"), API_CONFIG["base_url"], os.getenv("OPENAI_BASE_URL")], "base_url")
    require(bool(endpoint), "Set base_url in utils/llm/config.py")
    parsed = urlsplit(endpoint)
    require(parsed.scheme in {"http", "https"} and bool(parsed.netloc), "base_url must be an HTTP or HTTPS URL")
    return endpoint.rstrip("/")


def get_max_tokens_field():
    field = API_CONFIG["max_tokens_field"]
    require(field in {"max_tokens", "max_completion_tokens"}, "Invalid API token-limit field")
    return field


def request_parameters():
    parameters = API_CONFIG["extra_body"]
    require(isinstance(parameters, dict), "API extra_body must be an object")
    reserved = {"model", "messages", "input", "temperature", "stream", "store",
                "max_tokens", "max_completion_tokens", "max_output_tokens"}
    require(not reserved.intersection(parameters), "API extra_body cannot override request fields")
    return dict(parameters)


def load_llm_config(model_name=None, *, base_url=None, api_key=None, timeout_seconds=None):
    model = get_model_name(model_name)
    require(bool(model), "Set model in utils/llm/config.py or pass --model")
    key = _text_value([api_key], "api_key") or credential()
    return LLMConfig(model=model, base_url=configured_base_url(base_url),
        api_key=key, timeout_seconds=get_request_timeout(timeout_seconds))


def request_payload(model, prompt, max_tokens, temperature=0.0):
    return dict(model=model, messages=[dict(role="user", content=prompt)],
                temperature=temperature, **{get_max_tokens_field(): max_tokens}, **request_parameters())
