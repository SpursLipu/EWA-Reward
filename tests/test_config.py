from pathlib import Path

import pytest

from ewa_reward.config import ConfigurationError, get_settings, load_dotenv


def test_load_dotenv_preserves_existing_env(monkeypatch, tmp_path: Path):
    """中文：验证 .env 加载不会覆盖已存在的环境变量。
English: Verify that .env loading does not overwrite existing environment variables."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "EWA_REWARD_MODEL=from-file\nEWA_REWARD_PORT=9000\n# ignored\nBAD_LINE\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EWA_REWARD_MODEL", "from-env")

    load_dotenv(env_file)

    assert __import__("os").environ["EWA_REWARD_MODEL"] == "from-env"
    assert __import__("os").environ["EWA_REWARD_PORT"] == "9000"


def test_get_settings_rejects_invalid_port(monkeypatch):
    """中文：验证非法端口配置会触发配置异常。
English: Verify that an invalid port setting raises a configuration error."""
    monkeypatch.setenv("EWA_REWARD_PORT", "not-a-port")

    with pytest.raises(ConfigurationError, match="EWA_REWARD_PORT"):
        get_settings()
