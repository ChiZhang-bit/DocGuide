"""HTTPS requests with bounded retries and ordinary HTTP errors."""
from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.request


def build_ssl_context():
    return ssl.create_default_context()


def request_json(client, endpoint, payload):
    request = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {client.config.api_key}"},
        method="POST")
    for attempt, delay in enumerate((0, 1, 2, 4)):
        if delay:
            time.sleep(delay)
        try:
            with client.opener.open(request, timeout=client.config.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
            return client.response_text(body)
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            if client.config.api_key:
                message = message.replace(client.config.api_key, "[REDACTED]")
            status = exc.code
            exc.close()
            retryable = status >= 500 or status in (408, 429)
            if not retryable or attempt == 3:
                raise RuntimeError(f"LLM HTTP {status}: {message[:2000]}") from exc
        except Exception as exc:
            message = str(exc)
            if client.config.api_key:
                message = message.replace(client.config.api_key, "[REDACTED]")
            if attempt == 3:
                raise RuntimeError(f"LLM request failed: {message}") from exc
    raise RuntimeError("LLM request retries exhausted")
