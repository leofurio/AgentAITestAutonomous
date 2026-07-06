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
  `AIWEBTEST_REPLAY_HEADLESS=1`. The script is generated **deterministically — no model
  in the loop** — so it runs identically every time (CI, offline, zero API cost). Each
  element step is annotated with a `# locator:` comment showing the strongest idiomatic
  Playwright locator (`get_by_test_id` / `#id` / `get_by_role` / `get_by_text`), so the
  script reads like hand-written Playwright and can be adopted into a maintained suite,
  while the runtime still resolves through the robust candidate chain.

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
instruction (plus the target URL and the data keys) into a canonical, compact plain-text
spec:

```
GOAL: <one clause>
STEPS:
1. <action>
2. <action>
CHECKS:
- <verifiable check>
```

This line-oriented form is more token-efficient than JSON (no braces/quotes/repeated keys)
and far more reliable for models to emit, which cuts down on fallbacks. A tolerant parser
accepts common label/bullet variants and re-serializes them into the exact form above
(fixed section order, renumbered steps), so the same intent yields the same bytes. Feeding
the agent this canonical spec instead of raw prose reduces run-to-run variance. The pass is
conservative: it clarifies and structures, never inventing steps or leaking secret values
(data is referenced by key, e.g. `{password}`). The canonical rewrite is streamed to the UI
and recorded in the report. It is best-effort: if the output has no objective or steps, the
run falls back to the original instruction.

The pass is also a **token optimizer**: it compresses the request into a terse canonical
spec (short imperative steps, no filler or restated values), so the downstream loop carries
fewer input tokens on every turn. The pass itself runs under a tight output budget
(`normalizer_max_tokens`, default 1024) and low `normalizer_effort` to keep its own cost
small.

Toggle it with `agent.normalize_instruction` (default `true`). It can run on a cheaper or
faster model than the main loop via `normalizer_provider` / `normalizer_model` (empty =
reuse `agent_provider` / `model`), with optional dedicated keys
(`NORMALIZER_ANTHROPIC_API_KEY` / `NORMALIZER_OPENAI_API_KEY` /
`NORMALIZER_OPENROUTER_API_KEY`, each falling back to the matching provider key):

```bash
AIWEBTEST_AGENT__NORMALIZE_INSTRUCTION=false   # disable
AIWEBTEST_NORMALIZER_MODEL=claude-haiku-4-5    # normalize on a lighter model
AIWEBTEST_NORMALIZER_MAX_TOKENS=1024           # cap the canonical spec size
AIWEBTEST_NORMALIZER_EFFORT=low                # cheaper rewrite (empty = reuse effort)
```

### Debug logging

Set `log_level: DEBUG` (or `AIWEBTEST_LOG_LEVEL=DEBUG`) to trace the normalization rewrite
(request, raw model output, canonical JSON or fallback) and every agent call (provider,
model, message/tool counts, and a summary of each response — text snippets and tool calls).
Tool calls and results from the loop are logged too. DEBUG payloads can include data typed
into the page, so use it only for local debugging.

At startup the active level is printed (`... INFO aiwebtest: aiwebtest logging configured
at level DEBUG`) so you can confirm it took effect.

The simplest, cross-platform way is the `.env` file (loaded automatically):

```bash
echo "AIWEBTEST_LOG_LEVEL=DEBUG" >> .env
aiwebtest
```

Or set the variable in the shell — note the syntax differs per shell:

```bash
# bash / zsh
AIWEBTEST_LOG_LEVEL=DEBUG aiwebtest
```

```powershell
# Windows PowerShell — `VAR=value cmd` does NOT work here; set it first:
$env:AIWEBTEST_LOG_LEVEL = "DEBUG"
aiwebtest
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
