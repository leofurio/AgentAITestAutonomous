"""Render a TestReport to a standalone HTML page via Jinja2."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..agent.schemas import TestReport

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=select_autoescape(["html", "j2"]),
)


def _basename(path: str | None) -> str | None:
    return Path(path).name if path else None


def render_html(report: TestReport) -> str:
    template = _env.get_template("report.html.j2")
    return template.render(report=report, basename=_basename)
