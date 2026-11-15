"""LLM request configuration, transport, and structured stage calls."""
from importlib import import_module

__all__ = ["LoggedLLM", "ChatLLM", "ResponsesLLM", "API_CONFIG",
           "get_model_name", "get_request_timeout", "get_max_tokens_field", "credential",
           "configured_base_url", "load_llm_config", "request_payload", "build_chat_endpoint",
           "extract_json", "extract_verification_response", "strip_code_fence"]


def __getattr__(name):
    modules = {"LoggedLLM": "client", "ChatLLM": "chat",
               "build_chat_endpoint": "chat", "ResponsesLLM": "responses",
               "extract_json": "parsing", "extract_verification_response": "parsing",
               "strip_code_fence": "parsing"}
    if name not in __all__:
        raise AttributeError(name)
    return getattr(import_module("." + modules.get(name, "config"), __name__), name)
