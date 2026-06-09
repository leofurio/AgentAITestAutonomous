"""Configuration: merges config/default.yaml with environment variables.

Precedence: environment variables > config/default.yaml > field defaults.
Nested env overrides use the delimiter ``__`` (e.g. ``AIWEBTEST_BROWSER__HEADLESS=true``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import yaml
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
    max_steps: int = 40
    max_retries: int = 2
    include_screenshots: bool = True
    allowed_domains: list[str] = Field(default_factory=list)


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
    agent_provider: str = "anthropic"
    model: str = "claude-opus-4-8"
    effort: str = "high"
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


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_settings(config_path: Path | None = None) -> Settings:
    """Build Settings, applying env > YAML file > defaults precedence."""
    Settings._yaml_values = _load_yaml(config_path or DEFAULT_CONFIG_PATH)
    try:
        return Settings()
    finally:
        Settings._yaml_values = {}
