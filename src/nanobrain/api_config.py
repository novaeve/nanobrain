"""Configuration for an OpenAI-compatible third-party API.

The nanoBrain search loop is intentionally independent of a particular API
vendor.  This module provides one small, dependency-free configuration layer
that can be used by an optional LLM proposer, an evaluator, or a CLI.  Values
are read from the nanoBrain-specific variables first and then from common
OpenAI/DashScope aliases.  A local ``.env`` file is only used to fill values
that are not already present in the process environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Mapping
from urllib.parse import urlsplit


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ASSIGNMENT_RE = re.compile(
    r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$"
)

_ALIASES: dict[str, tuple[str, ...]] = {
    "api_key": (
        "NANOBRAIN_API_KEY",
        "OPENAI_API_KEY",
        "DASHSCOPE_API_KEY",
        "API_KEY",
    ),
    "base_url": (
        "NANOBRAIN_API_BASE_URL",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "DASHSCOPE_BASE_URL",
        "API_BASE_URL",
        "BASE_URL",
    ),
    "model": (
        "NANOBRAIN_API_MODEL",
        "OPENAI_MODEL",
        "DASHSCOPE_MODEL",
        "MODEL",
    ),
    "timeout_seconds": (
        "NANOBRAIN_API_TIMEOUT_SECONDS",
        "OPENAI_TIMEOUT_SECONDS",
    ),
    "max_retries": (
        "NANOBRAIN_API_MAX_RETRIES",
        "OPENAI_MAX_RETRIES",
    ),
    "retry_backoff_seconds": ("NANOBRAIN_API_RETRY_BACKOFF_SECONDS",),
    "temperature": ("NANOBRAIN_API_TEMPERATURE", "OPENAI_TEMPERATURE"),
    "max_tokens": ("NANOBRAIN_API_MAX_TOKENS", "OPENAI_MAX_TOKENS"),
    "enabled": ("NANOBRAIN_API_ENABLED",),
    "organization": ("NANOBRAIN_API_ORGANIZATION", "OPENAI_ORG_ID"),
    "api_key_header": ("NANOBRAIN_API_KEY_HEADER",),
    "api_key_scheme": ("NANOBRAIN_API_KEY_SCHEME",),
    "tls_verify": ("NANOBRAIN_API_TLS_VERIFY",),
}


class APIConfigError(ValueError):
    """Raised when the API configuration is incomplete or malformed."""


def _first_value(environment: Mapping[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = environment.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _parse_bool(value: str, *, field: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise APIConfigError(f"{field} must be true/false or 1/0")


def _parse_float(value: str, *, field: str, default: float) -> float:
    if not str(value).strip():
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise APIConfigError(f"{field} must be a number") from exc


def _parse_int(value: str, *, field: str, default: int) -> int:
    if not str(value).strip():
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise APIConfigError(f"{field} must be an integer") from exc


def _decode_dotenv_value(raw: str) -> str:
    """Decode the small, safe subset of dotenv syntax used by this project.

    Values are never evaluated as shell code.  Quotes are removed and common
    escaped quote/newline sequences in double-quoted values are interpreted.
    """

    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
        if raw.strip().startswith('"'):
            value = value.replace(r"\n", "\n").replace(r"\r", "\r").replace(r"\t", "\t")
            value = value.replace(r'\"', '"').replace(r"\\", "\\")
        else:
            value = value.replace("\\'", "'")
        return value
    # An inline comment is recognized only when separated by whitespace, so
    # URL fragments and other legitimate '#' characters remain intact.
    value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
    return value


def read_env_file(path: str | Path) -> dict[str, str]:
    """Read a dotenv file without executing its contents.

    Malformed lines are ignored deliberately; shell startup should remain
    usable even when a user is midway through editing the template.
    """

    target = Path(path).expanduser()
    if not target.is_file():
        return {}
    values: dict[str, str] = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _ASSIGNMENT_RE.match(line)
        if match is None:
            continue
        key, raw = match.groups()
        if _KEY_RE.fullmatch(key) is None:
            continue
        values[key] = _decode_dotenv_value(raw)
    return values


def merged_environment(
    environment: Mapping[str, str] | None = None,
    *,
    env_file: str | Path | None = None,
    override: bool = False,
) -> dict[str, str]:
    """Return process values plus optional dotenv values.

    Existing process values win by default.  This makes it safe to override a
    value for one run with ``NANOBRAIN_API_MODEL=...`` on the command line.
    """

    merged = dict(os.environ if environment is None else environment)
    selected = Path(env_file).expanduser() if env_file is not None else DEFAULT_ENV_FILE
    for key, value in read_env_file(selected).items():
        if override or key not in merged or not str(merged[key]).strip():
            merged[key] = value
    return merged


@dataclass(frozen=True)
class APIConfig:
    """Validated settings for a third-party OpenAI-compatible endpoint."""

    api_key: str = ""
    base_url: str = ""
    model: str = ""
    timeout_seconds: float = 120.0
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0
    temperature: float | None = None
    max_tokens: int | None = None
    enabled: bool = True
    organization: str = ""
    api_key_header: str = "Authorization"
    api_key_scheme: str = "Bearer"
    tls_verify: bool = True

    @classmethod
    def from_env(
        cls,
        environment: Mapping[str, str] | None = None,
        *,
        env_file: str | Path | None = None,
        override: bool = False,
    ) -> "APIConfig":
        values = merged_environment(environment, env_file=env_file, override=override)
        temperature_raw = _first_value(values, _ALIASES["temperature"])
        max_tokens_raw = _first_value(values, _ALIASES["max_tokens"])
        temperature = (
            None
            if not temperature_raw
            else _parse_float(temperature_raw, field="temperature", default=0.0)
        )
        max_tokens = (
            None
            if not max_tokens_raw
            else _parse_int(max_tokens_raw, field="max_tokens", default=0)
        )
        config = cls(
            api_key=_first_value(values, _ALIASES["api_key"]),
            base_url=_first_value(values, _ALIASES["base_url"]),
            model=_first_value(values, _ALIASES["model"]),
            timeout_seconds=_parse_float(
                _first_value(values, _ALIASES["timeout_seconds"]),
                field="timeout_seconds",
                default=120.0,
            ),
            max_retries=_parse_int(
                _first_value(values, _ALIASES["max_retries"]),
                field="max_retries",
                default=3,
            ),
            retry_backoff_seconds=_parse_float(
                _first_value(values, _ALIASES["retry_backoff_seconds"]),
                field="retry_backoff_seconds",
                default=1.0,
            ),
            temperature=temperature,
            max_tokens=max_tokens,
            enabled=_parse_bool(
                _first_value(values, _ALIASES["enabled"]) or "1",
                field="enabled",
            ),
            organization=_first_value(values, _ALIASES["organization"]),
            api_key_header=_first_value(values, _ALIASES["api_key_header"])
            or "Authorization",
            api_key_scheme=_first_value(values, _ALIASES["api_key_scheme"])
            or "Bearer",
            tls_verify=_parse_bool(
                _first_value(values, _ALIASES["tls_verify"]) or "1",
                field="tls_verify",
            ),
        )
        config.validate()
        return config

    @property
    def missing_fields(self) -> tuple[str, ...]:
        missing = []
        if not self.api_key:
            missing.append("api_key")
        if not self.base_url:
            missing.append("base_url")
        if not self.model:
            missing.append("model")
        return tuple(missing)

    @property
    def configured(self) -> bool:
        return bool(self.enabled and not self.missing_fields)

    @property
    def chat_completions_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    @property
    def safe_base_url(self) -> str | None:
        """Return a URL suitable for logs without credentials or query data."""

        if not self.base_url:
            return None
        parsed = urlsplit(self.base_url)
        host = parsed.hostname or ""
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        path = parsed.path.rstrip("/")
        return f"{parsed.scheme}://{host}{path}"

    def validate(self, *, require_credentials: bool = False) -> "APIConfig":
        if self.timeout_seconds <= 0:
            raise APIConfigError("timeout_seconds must be positive")
        # ``-1`` is the explicit infinite-retry setting.  It is useful for a
        # long AutoResearch campaign when a provider may be temporarily down.
        if self.max_retries < -1:
            raise APIConfigError("max_retries must be non-negative or -1 for infinite retries")
        if self.retry_backoff_seconds < 0:
            raise APIConfigError("retry_backoff_seconds must be non-negative")
        if self.temperature is not None and self.temperature < 0:
            raise APIConfigError("temperature must be non-negative")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise APIConfigError("max_tokens must be positive when supplied")
        if not self.api_key_header or not _KEY_RE.fullmatch(self.api_key_header.replace("-", "_")):
            raise APIConfigError("api_key_header contains invalid characters")
        if self.base_url:
            parsed = urlsplit(self.base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise APIConfigError("base_url must be an absolute http(s) URL")
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise APIConfigError("base_url must not contain credentials, query, or fragment data")
            try:
                _ = parsed.port
            except ValueError as exc:
                raise APIConfigError("base_url contains an invalid port") from exc
        if require_credentials and not self.configured:
            missing = ", ".join(self.missing_fields) or "enabled"
            raise APIConfigError(f"third-party API is not configured; missing {missing}")
        return self

    def redacted(self) -> dict[str, object]:
        """Return status metadata safe to print in logs or receipts."""

        return {
            "configured": self.configured,
            "enabled": self.enabled,
            "api_key_present": bool(self.api_key),
            "base_url": self.safe_base_url,
            "model": self.model or None,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "retry_backoff_seconds": self.retry_backoff_seconds,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "organization_present": bool(self.organization),
            "api_key_header": self.api_key_header,
            "tls_verify": self.tls_verify,
        }


# A descriptive alias makes the intended use explicit to callers that do not
# need to know the internal class name.
ThirdPartyAPIConfig = APIConfig


__all__ = [
    "APIConfig",
    "APIConfigError",
    "DEFAULT_ENV_FILE",
    "ThirdPartyAPIConfig",
    "merged_environment",
    "read_env_file",
]
