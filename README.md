# aiwebtest — Autonomous AI Web App Testing

Describe a test in plain language through a chat UI — *which site, what to test, how, and
with which data* — and an AI agent drives a real browser to carry it out, verifies the
expected outcomes, and produces a pass/fail report with screenshots and a full step log.

- **Engine**: Claude (`claude-opus-4-8` by default) via a manual tool-use agentic loop.
- **Browser**: Playwright. The agent perceives pages via an accessibility/DOM snapshot with
  stable ref ids, and acts by ref — robust against brittle selectors.
- **UI**: FastAPI + WebSocket streaming a live timeline (reasoning, steps, screenshots,
  assertions) to a lightweight web frontend.
- **Report**: structured JSON + standalone HTML per run.

## How it works

```
Chat instruction ─▶ Claude (tool-use loop) ─▶ Playwright tools ─▶ live page
                          ▲                          │
                          └──── snapshot / result ───┘
                                     │
                            EventBus ─▶ WebSocket ─▶ UI    ┌▶ report.json
                                     └────────────────────┴▶ report.html + screenshots
```

The agent calls tools — `navigate`, `get_page_snapshot`, `click`, `type_text`,
`select_option`, `press_key`, `wait_for`, `screenshot`, `get_text`, `assert_that`,
`finish_test`. Assertions are evaluated **deterministically in Python** (not by the model)
and recorded; any failed assertion forces an overall `fail`.

## Run locally (headful — browser visible)

```bash
pip install -e .
playwright install chromium
cp .env.example .env        # set ANTHROPIC_API_KEY
uvicorn aiwebtest.main:app --reload
```

Open http://localhost:8000, enter a target URL, an instruction and optional JSON data,
then **Run Test**. `config/default.yaml` has `headless: false`, so the browser is visible.

## Run tests (headless — CI / containers)

```bash
pip install -e ".[dev]"
playwright install --with-deps chromium
AIWEBTEST_BROWSER__HEADLESS=true pytest
```

The end-to-end test drives the full loop against a local fixture site using a **fake LLM**
(scripted tool calls) — no API key or network required.

## Configuration

`config/default.yaml` merged with environment variables (env wins). Nested overrides use
`__`, e.g. `AIWEBTEST_BROWSER__HEADLESS=true`. Key settings: `model`, `effort`,
`browser.headless`, `agent.max_steps`, `agent.allowed_domains` (empty = derived from the
target URL; keeps the agent on the site under test).

## Project layout

```
src/aiwebtest/
  config.py            settings (env > yaml > defaults)
  agent/   loop.py prompts.py schemas.py events.py
  browser/ session.py snapshot.py tools.py
  report/  builder.py html.py templates/
  web/     app.py routes.py ws.py manager.py
  static/  index.html app.js styles.css
tests/     unit/  e2e/  fixtures/site/
```
