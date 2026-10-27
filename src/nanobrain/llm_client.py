"""Minimal OpenAI-compatible chat client used by optional nanoBrain search.

The client uses Python's standard library so configuring a provider does not
force the training environment to install a vendor SDK.  It deliberately
returns decoded JSON and leaves candidate parsing/evaluation to the caller.
"""

from __future__ import annotations

import json
import socket
import ssl
import time
from typing import Any, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .api_config import APIConfig, APIConfigError


class APIRequestError(RuntimeError):
    """Raised when an OpenAI-compatible request cannot be completed."""


_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class OpenAICompatibleClient:
    """Small dependency-free client for ``/v1/chat/completions`` APIs."""

    def __init__(self, config: APIConfig | None = None):
        self.config = config or APIConfig.from_env()
        try:
            self.config.validate(require_credentials=True)
        except APIConfigError as exc:
            raise APIRequestError(str(exc)) from exc

    @classmethod
    def from_env(
        cls,
        environment: Mapping[str, str] | None = None,
        *,
        env_file: str | None = None,
    ) -> "OpenAICompatibleClient":
        return cls(APIConfig.from_env(environment, env_file=env_file))

    def _headers(self) -> dict[str, str]:
        value = self.config.api_key
        if self.config.api_key_scheme:
            value = f"{self.config.api_key_scheme} {value}"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            self.config.api_key_header: value,
            "User-Agent": "nanoBrain/1.0",
        }
        if self.config.organization:
            headers["OpenAI-Organization"] = self.config.organization
        return headers

    def _request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            self.config.chat_completions_url,
            data=body,
            headers=self._headers(),
            method="POST",
        )
        ssl_context = None
        if not self.config.tls_verify:
            # This is opt-in because some private gateways expose a
            # certificate for their parent domain rather than the API host.
            ssl_context = ssl._create_unverified_context()
        last_error: Exception | None = None
        attempt = 0
        while True:
            can_retry = self.config.max_retries < 0 or attempt < self.config.max_retries
            try:
                with urlopen(
                    request,
                    timeout=self.config.timeout_seconds,
                    context=ssl_context,
                ) as response:
                    raw = response.read()
                decoded = json.loads(raw.decode("utf-8"))
                if not isinstance(decoded, dict):
                    raise APIRequestError("API response must be a JSON object")
                return decoded
            except HTTPError as exc:
                last_error = exc
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:1000]
                except Exception:
                    detail = ""
                if exc.code not in _RETRYABLE_STATUS or not can_retry:
                    suffix = f": {detail}" if detail else ""
                    raise APIRequestError(f"API request failed with HTTP {exc.code}{suffix}") from exc
            except (URLError, TimeoutError, socket.timeout, json.JSONDecodeError) as exc:
                last_error = exc
                if not can_retry:
                    raise APIRequestError(f"API request failed: {type(exc).__name__}") from exc
            if self.config.retry_backoff_seconds:
                # Cap the exponent before converting to float; infinite retry
                # mode must remain numerically stable after many outages.
                delay = self.config.retry_backoff_seconds * (2.0 ** min(attempt, 5))
                time.sleep(min(delay, 30.0))
            attempt += 1

    def chat(
        self,
        messages: Iterable[Mapping[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: Mapping[str, Any] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        normalized = []
        for message in messages:
            if not isinstance(message, Mapping) or "role" not in message or "content" not in message:
                raise ValueError("each message must contain role and content")
            normalized.append({"role": str(message["role"]), "content": str(message["content"])})
        if not normalized:
            raise ValueError("messages must not be empty")
        payload: dict[str, Any] = {"model": self.config.model, "messages": normalized}
        selected_temperature = self.config.temperature if temperature is None else temperature
        selected_max_tokens = self.config.max_tokens if max_tokens is None else max_tokens
        if selected_temperature is not None:
            payload["temperature"] = float(selected_temperature)
        if selected_max_tokens is not None:
            payload["max_tokens"] = int(selected_max_tokens)
        if response_format is not None:
            payload["response_format"] = dict(response_format)
        payload.update(extra)
        return self._request(payload)

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Send one user prompt and extract the first assistant text."""

        response = self.chat([{"role": "user", "content": str(prompt)}], **kwargs)
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise APIRequestError("API response has no choices[0].message.content") from exc
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            if parts:
                return "".join(parts)
        raise APIRequestError("API response content is not text")


__all__ = ["APIRequestError", "OpenAICompatibleClient"]
