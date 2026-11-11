from pathlib import Path

import pytest

from nanobrain.api_config import APIConfig, APIConfigError, read_env_file


def test_dotenv_values_are_read_without_executing_shell(tmp_path: Path):
    path = tmp_path / ".env"
    path.write_text(
        "NANOBRAIN_API_KEY='secret-value'\n"
        "NANOBRAIN_API_BASE_URL=https://provider.example/v1 # comment\n"
        "NANOBRAIN_API_MODEL=brain-model\n"
        "IGNORED=$(touch SHOULD_NOT_EXIST)\n",
        encoding="utf-8",
    )
    values = read_env_file(path)
    assert values["NANOBRAIN_API_KEY"] == "secret-value"
    assert values["NANOBRAIN_API_BASE_URL"] == "https://provider.example/v1"
    assert values["IGNORED"] == "$(touch SHOULD_NOT_EXIST)"
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists()


def test_specific_variables_win_over_aliases_and_redaction(tmp_path: Path):
    path = tmp_path / ".env"
    path.write_text(
        "OPENAI_API_KEY=file-key\nOPENAI_BASE_URL=https://file.example/v1\nOPENAI_MODEL=file-model\n",
        encoding="utf-8",
    )
    config = APIConfig.from_env(
        {
            "NANOBRAIN_API_KEY": "specific-key",
            "NANOBRAIN_API_BASE_URL": "https://specific.example/v1",
            "NANOBRAIN_API_MODEL": "specific-model",
        },
        env_file=path,
    )
    assert config.configured
    assert config.chat_completions_url.endswith("/v1/chat/completions")
    safe = config.redacted()
    assert safe["api_key_present"] is True
    assert "specific-key" not in str(safe)
    assert safe["base_url"] == "https://specific.example/v1"


def test_aliases_and_required_validation():
    config = APIConfig.from_env(
        {
            "OPENAI_API_KEY": "alias-key",
            "OPENAI_BASE_URL": "https://alias.example/v1",
            "OPENAI_MODEL": "alias-model",
        },
        env_file="/does/not/exist",
    )
    assert config.configured
    assert config.model == "alias-model"
    with pytest.raises(APIConfigError):
        APIConfig.from_env({"NANOBRAIN_API_BASE_URL": "relative/path"}, env_file="/does/not/exist")


def test_infinite_retry_sentinel_is_valid():
    config = APIConfig.from_env(
        {
            "NANOBRAIN_API_KEY": "key",
            "NANOBRAIN_API_BASE_URL": "https://provider.example/v1",
            "NANOBRAIN_API_MODEL": "model",
            "NANOBRAIN_API_MAX_RETRIES": "-1",
        },
        env_file="/does/not/exist",
    )
    assert config.max_retries == -1


def test_tls_verification_is_explicit_and_redacted():
    config = APIConfig.from_env(
        {
            "NANOBRAIN_API_KEY": "key",
            "NANOBRAIN_API_BASE_URL": "https://provider.example/v1",
            "NANOBRAIN_API_MODEL": "model",
            "NANOBRAIN_API_TLS_VERIFY": "0",
        },
        env_file="/does/not/exist",
    )
    assert config.tls_verify is False
    assert config.redacted()["tls_verify"] is False
