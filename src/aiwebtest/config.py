"""Configuration: merges config/default.yaml with environment variables.

Precedence: environment variables > config/default.yaml > field defaults.
Nested env overrides use the delimiter ``__`` (e.g. ``AIWEBTEST_BROWSER__HEADLESS=true``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

# Repository root = three levels up from this file (src/aiwebtest/config.py).
ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "default.yaml"


class ViewportConfig(BaseModel):
    width: int = 1280
    height: int = 800


class BrowserConfig(BaseModel):
    headless: bool = False
    channel: str | None = "chrome"
    viewport: ViewportConfig = Field(default_factory=ViewportConfig)
    nav_timeout_ms: int = 30000
    action_timeout_ms: int = 10000


class AgentConfig(BaseModel):
    max_steps: int = 100
    max_retries: int = 2
    include_screenshots: bool = True
    allowed_domains: list[str] = Field(default_factory=list)
    # Run a pre-pass that rewrites the free-form instruction/URL/data into a canonical,
    # normalized spec before driving the browser. Trades one extra LLM call for more
    # deterministic, repeatable runs.
    normalize_instruction: bool = True


class ReportConfig(BaseModel):
    output_dir: str = "runs"


class _YamlSource(PydanticBaseSettingsSource):
    """A settings source that yields values from a pre-loaded YAML dict."""

    def __init__(self, settings_cls: type[BaseSettings], values: dict[str, Any]) -> None:
        super().__init__(settings_cls)
        self._values = values

    def get_field_value(self, field, field_name):  # pragma: no cover - unused hook
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        return dict(self._values)


class Settings(BaseSettings):
    """Top-level settings object.

    Precedence (high → low): init args > environment > YAML file > field defaults.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIWEBTEST_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Holds the YAML dict for the next instantiation (set by load_settings).
    _yaml_values: ClassVar[dict[str, Any]] = {}

    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    # Optional dedicated API keys for the normalizer pass. Each falls back to the matching
    # provider key above when empty, so the normalizer can run on a separate key/account
    # (or even a different provider) without affecting the main run.
    normalizer_anthropic_api_key: str = Field(default="", alias="NORMALIZER_ANTHROPIC_API_KEY")
    normalizer_openai_api_key: str = Field(default="", alias="NORMALIZER_OPENAI_API_KEY")
    normalizer_openrouter_api_key: str = Field(default="", alias="NORMALIZER_OPENROUTER_API_KEY")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_http_referer: str = ""
    openrouter_app_title: str = "aiwebtest"
    agent_provider: str = "anthropic"
    # The instruction normalizer can run on a different (e.g. cheaper/faster) provider
    # and model. Empty values fall back to agent_provider / model respectively.
    normalizer_provider: str = ""
    normalizer_model: str = ""
    # Token budget for the normalizer pass: a tight output cap keeps the canonical spec
    # compact (fewer input tokens for the downstream loop) and bounds the pass's own cost.
    normalizer_max_tokens: int = 1024
    # Lower effort = less thinking/tokens for what is a simple rewrite. Empty = reuse effort.
    normalizer_effort: str = "low"
    # /api/playwright/execute runs arbitrary Python: keep it opt-out and local-only.
    code_runner_enabled: bool = True
    code_runner_allow_remote: bool = False
    model: str = "claude-opus-4-8"
    # medium balances speed and quality for browser-driving; raise to high/xhigh for
    # harder sites, or lower to low for the fastest, simplest runs.
    effort: str = "medium"
    max_tokens: int = 8192
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    data: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        # env beats YAML; YAML beats field defaults.
        yaml_source = _YamlSource(settings_cls, dict(cls._yaml_values))
        return (init_settings, env_settings, dotenv_settings, yaml_source, file_secret_settings)

    @property
    def output_dir(self) -> Path:
        path = Path(self.report.output_dir)
        return path if path.is_absolute() else ROOT_DIR / path


def normalizer_settings(settings: Settings) -> Settings:
    """Derive the Settings the instruction normalizer should run under.

    Falls back to the run's provider/model when the normalizer-specific values are unset,
    so the normalizer adapter resolves the right model via ``settings.model``.
    """
    provider = settings.normalizer_provider or settings.agent_provider
    model = settings.normalizer_model or settings.model
    update: dict[str, Any] = {
        "agent_provider": provider,
        "model": model,
        # Cap the pass's output and effort to keep both its cost and the canonical spec
        # (the downstream loop's input) small. build_client / the adapters read these.
        "max_tokens": settings.normalizer_max_tokens,
        "effort": settings.normalizer_effort or settings.effort,
    }
    # Dedicated normalizer keys override the inherited provider keys when set; build_client
    # reads anthropic_api_key / openai_api_key / openrouter_api_key off these Settings.
    if settings.normalizer_anthropic_api_key:
        update["anthropic_api_key"] = settings.normalizer_anthropic_api_key
    if settings.normalizer_openai_api_key:
        update["openai_api_key"] = settings.normalizer_openai_api_key
    if settings.normalizer_openrouter_api_key:
        update["openrouter_api_key"] = settings.normalizer_openrouter_api_key
    return settings.model_copy(update=update)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_settings(config_path: Path | None = None, dotenv_path: Path | None = None) -> Settings:
    """Build Settings, applying env > YAML file > defaults precedence."""
    env_file = ROOT_DIR / ".env" if dotenv_path is None else dotenv_path
    if env_file.exists():
        load_dotenv(env_file, override=False)

    Settings._yaml_values = _load_yaml(config_path or DEFAULT_CONFIG_PATH)
    try:
        return Settings()
    finally:
        Settings._yaml_values = {}
