"""Config precedence: env > yaml > defaults, including nested overrides."""

from __future__ import annotations

from pathlib import Path

from aiwebtest.config import load_settings


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_yaml_values_loaded(tmp_path: Path, monkeypatch):
    # Isolate from any ambient AIWEBTEST_* env vars so we test the YAML source alone.
    monkeypatch.delenv("AIWEBTEST_MODEL", raising=False)
    monkeypatch.delenv("AIWEBTEST_BROWSER__HEADLESS", raising=False)
    cfg = _write_yaml(tmp_path, "model: claude-sonnet-4-6\nbrowser:\n  headless: false\n")
    settings = load_settings(cfg)
    assert settings.model == "claude-sonnet-4-6"
    assert settings.browser.headless is False


def test_env_overrides_yaml(tmp_path: Path, monkeypatch):
    cfg = _write_yaml(
        tmp_path,
        "agent_provider: anthropic\nmodel: claude-sonnet-4-6\nbrowser:\n  headless: false\n",
    )
    monkeypatch.setenv("AIWEBTEST_AGENT_PROVIDER", "openai")
    monkeypatch.setenv("AIWEBTEST_MODEL", "claude-opus-4-8")
    monkeypatch.setenv("AIWEBTEST_BROWSER__HEADLESS", "true")
    settings = load_settings(cfg)
    assert settings.agent_provider == "openai"
    assert settings.model == "claude-opus-4-8"
    assert settings.browser.headless is True


def test_defaults_when_absent(tmp_path: Path):
    settings = load_settings(tmp_path / "missing.yaml")
    assert settings.agent.max_steps == 40
    assert settings.effort == "high"
