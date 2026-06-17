# aiwebtest — Autonomous AI Web App Testing

Describe a test in plain language through a chat UI — *which site, what to test, how, and
with which data* — and an AI agent drives a real browser to carry it out, verifies the
expected outcomes, and produces a pass/fail report with screenshots and a full step log.

- **Engine**: provider-based agent loop. Anthropic Claude (`claude-opus-4-8`) is the
  default; OpenAI Responses API or OpenRouter Chat Completions function calling can be
  selected via configuration.
- **Browser**: Playwright. The agent perceives pages via an accessibility/DOM snapshot with
  stable ref ids, and acts by ref — robust against brittle selectors.
- **UI**: FastAPI + WebSocket streaming a live timeline (reasoning, steps, screenshots,
  assertions) to a lightweight web frontend.
- **Report**: structured JSON + standalone HTML per run.
- **Replay**: every completed run also writes a standalone `playwright_test.py` script
  that can be downloaded or pasted into the local runner at `/runner`. It launches the
  same browser channel as the live run (e.g. installed Chrome) and falls back to the
  bundled Chromium; override with `AIWEBTEST_REPLAY_CHANNEL`, run headless with
  `AIWEBTEST_REPLAY_HEADLESS=1`.

## How it works

```
Chat instruction ─▶ Normalizer ─▶ Claude (tool-use loop) ─▶ Playwright tools ─▶ live page
   + URL + data    (canonical spec)      ▲                        │
                                         └──── snapshot / result ──┘
                                     │
                            EventBus ─▶ WebSocket ─▶ UI    ┌▶ report.json
                                     └────────────────────┴▶ report.html + screenshots
```

### Instruction normalization

Before driving the browser, a small **normalizer** pass rewrites the free-form
instruction (plus the target URL and the data keys) into a canonical, numbered
specification — explicit ordered steps and verifiable expected results. Feeding the agent
this normalized text instead of raw prose reduces run-to-run variance, so the same intent
yields the same steps and assertions. The pass is conservative: it clarifies and
structures, never inventing steps or leaking secret values (data is referenced by key,
e.g. `{password}`). The canonical rewrite is streamed to the UI and recorded in the
report. It is best-effort — if it fails, the run falls back to the original instruction.

Toggle it with `agent.normalize_instruction` (default `true`). It can run on a cheaper or
faster model than the main loop via `normalizer_provider` / `normalizer_model` (empty =
reuse `agent_provider` / `model`):

```bash
AIWEBTEST_AGENT__NORMALIZE_INSTRUCTION=false   # disable
AIWEBTEST_NORMALIZER_MODEL=claude-haiku-4-5    # normalize on a lighter model
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
aiwebtest
```

Open http://localhost:8000, enter a target URL, an instruction and optional JSON data,
then **Run Test**. `config/default.yaml` has `headless: false`, so the browser is visible.

On Windows, prefer the `aiwebtest` entrypoint instead of `uvicorn --reload`: Playwright's
async driver needs an event loop that supports subprocesses.

After a run completes, use the `download Playwright code` link to save the generated
script, or open `run code` to load it into `/runner`. The runner executes pasted Python
code locally, so use it only for scripts you trust.

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
`__`, e.g. `AIWEBTEST_BROWSER__HEADLESS=true`. Key settings: `agent_provider`, `model`,
`effort`, `browser.headless`, `agent.max_steps`, `agent.allowed_domains` (empty = derived
from the target URL; keeps the agent on the site under test).

### Provider selection

Anthropic remains the default:

```bash
ANTHROPIC_API_KEY=sk-ant-...
AIWEBTEST_AGENT_PROVIDER=anthropic
AIWEBTEST_MODEL=claude-opus-4-8
```

To use OpenAI for the agentic test generation loop:

```bash
pip install -e ".[openai]"
OPENAI_API_KEY=sk-...
AIWEBTEST_AGENT_PROVIDER=openai
AIWEBTEST_MODEL=<openai-model>
```

To use OpenRouter:

```bash
pip install -e ".[openai]"
OPENROUTER_API_KEY=sk-or-...
AIWEBTEST_AGENT_PROVIDER=openrouter
AIWEBTEST_MODEL=~anthropic/claude-sonnet-latest
```

Optional OpenRouter attribution headers:

```bash
AIWEBTEST_OPENROUTER_HTTP_REFERER=http://localhost:8000
AIWEBTEST_OPENROUTER_APP_TITLE=aiwebtest
```

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
