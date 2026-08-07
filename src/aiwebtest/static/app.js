"use strict";

const $ = (id) => document.getElementById(id);
const timeline = $("timeline");
const statusEl = $("status");
const verdictEl = $("verdict");
const runBtn = $("run");
const suiteList = $("suite-list");
const saveSuiteBtn = $("save-suite");
const runSuiteBtn = $("run-suite");
const compareEl = $("compare");
const compareModelsEl = $("compare-models");
const addModelBtn = $("add-model");
const runCompareBtn = $("run-compare");

// The last completed live run: saving a suite test attaches it as the recording.
let lastRunId = null;

function setStatus(state) {
  statusEl.textContent = state;
  statusEl.className = "status " + state;
}

function addCard(cls, label, bodyNode) {
  const card = document.createElement("div");
  card.className = "card " + cls;
  if (label) {
    const l = document.createElement("div");
    l.className = "label";
    l.textContent = label;
    card.appendChild(l);
  }
  if (bodyNode) card.appendChild(bodyNode);
  timeline.appendChild(card);
  card.scrollIntoView({ behavior: "smooth", block: "end" });
  return card;
}

function textNode(text, mono) {
  const el = document.createElement(mono ? "pre" : "div");
  el.textContent = text;
  if (mono) el.className = "mono";
  return el;
}

function handleEvent(evt) {
  const { type, data } = evt;
  if (type === "status") {
    setStatus(data.state);
  } else if (type === "normalized") {
    addCard("normalized", "normalized instruction", textNode(data.text, true));
  } else if (type === "warning") {
    addCard("warning", "warning", textNode(data.message));
  } else if (type === "reasoning") {
    addCard("reasoning", "reasoning", textNode(data.text));
  } else if (type === "step") {
    const body = textNode(JSON.stringify(data.input), true);
    addCard("step", "→ " + data.tool, body);
  } else if (type === "screenshot") {
    const img = document.createElement("img");
    img.src = data.url;
    addCard("step", "screenshot", img);
  } else if (type === "assertion") {
    const cls = data.passed ? "assertion pass" : "assertion fail";
    const verdict = data.passed ? "✓ PASS" : "✗ FAIL";
    const body = textNode(
      `${verdict} — ${data.description}` +
        (data.expected != null ? `\nexpected: ${data.expected}` : "") +
        (data.actual != null ? `\nactual: ${data.actual}` : "")
    );
    addCard(cls, "assertion", body);
  } else if (type === "report") {
    showVerdict(data);
  } else if (type === "error") {
    verdictEl.className = "verdict error";
    verdictEl.textContent = "Error: " + data.message;
    verdictEl.classList.remove("hidden");
  }
}

function showVerdict(data) {
  verdictEl.className = "verdict " + data.verdict;
  verdictEl.innerHTML = "";
  const span = document.createElement("span");
  span.textContent = `Verdict: ${data.verdict.toUpperCase()} — ${data.summary} `;
  verdictEl.appendChild(span);
  const link = document.createElement("a");
  link.href = data.report_url;
  link.target = "_blank";
  link.textContent = "(open full report)";
  verdictEl.appendChild(link);
  if (data.playwright_url) {
    verdictEl.appendChild(document.createTextNode(" "));
    const codeLink = document.createElement("a");
    codeLink.href = data.playwright_url;
    codeLink.target = "_blank";
    codeLink.textContent = "(download Playwright code)";
    verdictEl.appendChild(codeLink);
  }
  if (data.runner_url && data.playwright_url) {
    verdictEl.appendChild(document.createTextNode(" "));
    const runnerLink = document.createElement("a");
    runnerLink.href = `${data.runner_url}?script=${encodeURIComponent(data.playwright_url)}`;
    runnerLink.target = "_blank";
    runnerLink.textContent = "(run code)";
    verdictEl.appendChild(runnerLink);
  }
  verdictEl.classList.remove("hidden");
}

// Parses the test-data textarea. Returns {ok:false} after alerting on bad JSON, so
// every caller can bail out the same way.
function readTestData() {
  const raw = $("data").value.trim();
  if (!raw) return { ok: true, data: null };
  try {
    return { ok: true, data: JSON.parse(raw) };
  } catch (e) {
    alert("Test data is not valid JSON: " + e.message);
    return { ok: false };
  }
}

async function startRun() {
  const parsed = readTestData();
  if (!parsed.ok) return;
  const body = {
    instruction: $("instruction").value,
    target_url: $("target_url").value.trim() || null,
    data: parsed.data,
  };
  if (!body.instruction.trim()) {
    alert("Please enter an instruction.");
    return;
  }

  timeline.innerHTML = "";
  verdictEl.classList.add("hidden");
  compareEl.classList.add("hidden");
  runBtn.disabled = true;
  setStatus("running");

  let runId;
  try {
    const resp = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    runId = (await resp.json()).run_id;
  } catch (e) {
    setStatus("error");
    runBtn.disabled = false;
    alert("Failed to start run: " + e.message);
    return;
  }

  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/runs/${runId}`);
  ws.onmessage = (m) => {
    const evt = JSON.parse(m.data);
    if (evt.type === "report") lastRunId = runId; // becomes the suite recording
    handleEvent(evt);
  };
  ws.onclose = () => {
    runBtn.disabled = false;
  };
  ws.onerror = () => {
    setStatus("error");
    runBtn.disabled = false;
  };
}

runBtn.addEventListener("click", startRun);

// --- Suite -----------------------------------------------------------------------

async function api(method, url, body) {
  const resp = await fetch(url, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!resp.ok) {
    let detail = "HTTP " + resp.status;
    try { detail = (await resp.json()).detail || detail; } catch (_) { /* keep */ }
    throw new Error(detail);
  }
  return resp.json();
}

function badge(cls, text) {
  const b = document.createElement("span");
  b.className = "badge " + cls;
  b.textContent = text;
  return b;
}

function suiteItem(test) {
  const item = document.createElement("div");
  item.className = "suite-item";

  const row = document.createElement("div");
  row.className = "row";
  const name = document.createElement("div");
  name.className = "name";
  name.textContent = test.name;
  name.title = test.instruction;
  row.appendChild(name);

  const last = test.history[test.history.length - 1];
  row.appendChild(last ? badge(last.verdict, last.verdict) : badge("none", "never run"));
  item.appendChild(row);

  const meta = document.createElement("div");
  meta.className = "meta";
  const runs = test.history.length;
  meta.textContent =
    `${runs} run${runs === 1 ? "" : "s"}` +
    (test.source_run_id ? " · recorded" : " · no recording") +
    (last && last.healed ? " · " : "");
  if (last && last.healed) meta.appendChild(badge("healed", "healed"));
  item.appendChild(meta);

  const actions = document.createElement("div");
  actions.className = "actions";
  actions.style.marginTop = "6px";
  const mkBtn = (label, title, fn) => {
    const b = document.createElement("button");
    b.textContent = label;
    b.title = title;
    b.addEventListener("click", () => fn(b));
    actions.appendChild(b);
  };
  mkBtn("▶ Run", "Replay the recording without the AI; if it broke, heal the failing step in-place, then a full AI re-run only if needed",
    (b) => runSuiteTest(test.test_id, "auto", b));
  mkBtn("🤖 Re-run with AI", "Full agent run that re-records the test when it passes",
    (b) => runSuiteTest(test.test_id, "agent", b));
  mkBtn("✏️", "Rename this test", async () => {
    const name = prompt("New name for this test:", test.name);
    if (name === null) return;            // cancelled
    if (!name.trim()) {
      alert("The name must not be empty.");
      return;
    }
    try {
      await api("PATCH", `/api/suite/${test.test_id}`, { name: name.trim() });
      loadSuite();
    } catch (e) {
      alert("Rename failed: " + e.message);
    }
  });
  mkBtn("✕", "Delete this suite test", async () => {
    if (!confirm(`Delete suite test "${test.name}"?`)) return;
    await api("DELETE", `/api/suite/${test.test_id}`);
    loadSuite();
  });
  item.appendChild(actions);

  if (test.history.length) item.appendChild(historyBlock(test.history));
  return item;
}

// What each run mode means, in the user's terms: the history is where you check what a
// saved test actually did — especially the AI-free replays.
const MODE_LABEL = {
  replay: "replayed without AI",
  heal: "self-healed (AI fixed one step)",
  agent: "full AI run",
};

function historyBlock(history) {
  const details = document.createElement("details");
  details.className = "run-log";
  const summary = document.createElement("summary");
  summary.textContent = `Run log (${history.length})`;
  details.appendChild(summary);

  // Newest first: the last run is what you almost always want to inspect.
  for (const record of [...history].reverse()) {
    const row = document.createElement("div");
    row.className = "run-entry";

    const head = document.createElement("div");
    head.className = "row";
    head.appendChild(badge(record.verdict, record.verdict));
    const what = document.createElement("span");
    what.className = "run-mode";
    what.textContent = MODE_LABEL[record.mode] || record.mode;
    head.appendChild(what);
    row.appendChild(head);

    const when = document.createElement("div");
    when.className = "meta";
    when.textContent = new Date(record.finished_at).toLocaleString();
    row.appendChild(when);

    if (record.summary) {
      const text = document.createElement("div");
      text.className = "run-summary";
      text.textContent = record.summary;
      row.appendChild(text);
    }

    const links = document.createElement("div");
    links.className = "run-links";
    if (record.log_url) links.appendChild(link(record.log_url, "steps log"));
    if (record.report_url) links.appendChild(link(record.report_url, "report"));
    if (links.children.length) row.appendChild(links);

    details.appendChild(row);
  }
  return details;
}

function link(href, text) {
  const a = document.createElement("a");
  a.href = href;
  a.target = "_blank";
  a.textContent = text;
  return a;
}

async function loadSuite() {
  try {
    const data = await api("GET", "/api/suite");
    suiteList.innerHTML = "";
    if (!data.tests.length) {
      const empty = document.createElement("p");
      empty.className = "hint suite-empty";
      empty.textContent =
        "No saved tests yet — write a test above and click “Save current test to suite”.";
      suiteList.appendChild(empty);
      return;
    }
    for (const test of data.tests) suiteList.appendChild(suiteItem(test));
  } catch (e) {
    // Suite UI is secondary: never block the main flow on it.
    console.error("suite load failed:", e);
  }
}

async function runSuiteTest(testId, mode, btn) {
  btn.disabled = true;
  const original = btn.textContent;
  btn.textContent = "…";
  try {
    const record = await api("POST", `/api/suite/${testId}/run`, { mode });
    setStatus(record.verdict === "pass" ? "done" : "error");
  } catch (e) {
    alert("Suite run failed: " + e.message);
  } finally {
    btn.textContent = original;
    btn.disabled = false;
    loadSuite();
  }
}

saveSuiteBtn.addEventListener("click", async () => {
  const instruction = $("instruction").value;
  if (!instruction.trim()) {
    alert("Fill in the instruction first — it becomes the suite test.");
    return;
  }
  const name = prompt("Name for this suite test:", instruction.slice(0, 60));
  if (!name) return;
  const parsed = readTestData();
  if (!parsed.ok) return;
  try {
    await api("POST", "/api/suite", {
      name,
      instruction,
      target_url: $("target_url").value.trim() || null,
      data: parsed.data,
      source_run_id: lastRunId, // last completed live run becomes the recording
    });
    loadSuite();
  } catch (e) {
    alert("Save failed: " + e.message);
  }
});

runSuiteBtn.addEventListener("click", async () => {
  runSuiteBtn.disabled = true;
  runSuiteBtn.textContent = "Running…";
  try {
    const res = await api("POST", "/api/suite/run_all", { mode: "auto" });
    if (!res.total) {
      alert("No saved tests yet — save one first with “Save current test to suite”.");
    } else {
      setStatus(res.passed === res.total ? "done" : "error");
      alert(`Suite: ${res.passed} of ${res.total} passed`);
    }
  } catch (e) {
    alert("Suite run failed: " + e.message);
  } finally {
    runSuiteBtn.disabled = false;
    runSuiteBtn.textContent = "Run all saved tests";
    loadSuite();
  }
});

// --- Model comparison -------------------------------------------------------------

const PROVIDERS = ["anthropic", "openai", "openrouter"];
let config = { agent_provider: "", model: "", min_models: 2, max_models: 6 };

function modelRow(model, provider) {
  const row = document.createElement("div");
  row.className = "model-row";

  const select = document.createElement("select");
  select.title = "Provider — leave on default to use the configured one";
  for (const name of ["", ...PROVIDERS]) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name || "default";
    option.selected = name === (provider || "");
    select.appendChild(option);
  }

  const input = document.createElement("input");
  input.type = "text";
  input.placeholder = "model id";
  input.value = model || "";

  const remove = document.createElement("button");
  remove.className = "rm";
  remove.textContent = "×";
  remove.title = "Remove this model";
  remove.addEventListener("click", () => {
    row.remove();
    syncModelButtons();
  });

  row.append(select, input, remove);
  return row;
}

function modelRows() {
  return Array.from(compareModelsEl.querySelectorAll(".model-row"));
}

function syncModelButtons() {
  addModelBtn.disabled = modelRows().length >= config.max_models;
}

function addModel(model, provider) {
  compareModelsEl.appendChild(modelRow(model, provider));
  syncModelButtons();
}

function readModels() {
  return modelRows()
    .map((row) => ({
      provider: row.querySelector("select").value || null,
      model: row.querySelector("input").value.trim(),
    }))
    .filter((entry) => entry.model);
}

function compareColumn(entry) {
  const col = document.createElement("div");
  col.className = "compare-col";

  const head = document.createElement("header");
  const name = document.createElement("div");
  name.className = "model";
  name.textContent = entry.model;
  name.title = `${entry.provider} / ${entry.model}`;
  head.append(name, badge("none", "running"));

  const log = document.createElement("div");
  log.className = "compare-log";
  col.append(head, log);
  col.dataset.runId = entry.run_id;
  return col;
}

function logLine(col, cls, text) {
  const log = col.querySelector(".compare-log");
  const line = document.createElement("div");
  line.className = "log-line " + cls;
  line.textContent = text;
  log.appendChild(line);
  log.scrollTop = log.scrollHeight;
}

function setColumnState(col, cls, text) {
  const state = col.querySelector(".badge");
  state.className = "badge " + cls;
  state.textContent = text;
}

function handleCompareEvent(col, evt) {
  const { type, data } = evt;
  if (type === "step") {
    logLine(col, "step", "→ " + data.tool);
  } else if (type === "reasoning") {
    // The columns are narrow; each run's full reasoning is in its own report.
    const text = data.text.length > 240 ? data.text.slice(0, 240) + "…" : data.text;
    logLine(col, "reasoning", text);
  } else if (type === "assertion") {
    logLine(col, data.passed ? "pass" : "fail",
      (data.passed ? "✓ " : "✗ ") + data.description);
  } else if (type === "normalized") {
    // Each contender normalizes with its own model, so the specs differ — how a model
    // read the request is part of what is being compared.
    logLine(col, "normalized", data.text);
  } else if (type === "warning") {
    logLine(col, "warn", "⚠ " + data.message);
  } else if (type === "report") {
    setColumnState(col, data.verdict, data.verdict);
  } else if (type === "error") {
    setColumnState(col, "error", "error");
    logLine(col, "fail", data.message);
  }
}

const num = (n) => (n || 0).toLocaleString();

const NUMERIC_COLUMNS = ["Steps", "Assertions", "Calls", "Input", "Output",
  "Cache read", "Time"];

function renderCompareSummary(payload, container) {
  container.innerHTML = "";
  const table = document.createElement("table");
  table.className = "compare-table";

  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const label of ["Model", "Provider", "Verdict", ...NUMERIC_COLUMNS, ""]) {
    const th = document.createElement("th");
    th.textContent = label;
    if (NUMERIC_COLUMNS.includes(label)) th.className = "num";
    headRow.appendChild(th);
  }
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  for (const r of payload.results) {
    const tr = document.createElement("tr");
    const cell = (text, cls) => {
      const td = document.createElement("td");
      if (cls) td.className = cls;
      td.textContent = text;
      tr.appendChild(td);
    };
    cell(r.model, "mono");
    cell(r.provider);
    const verdict = document.createElement("td");
    verdict.appendChild(badge(r.verdict || "none", r.verdict || r.status));
    verdict.title = r.summary || "";
    tr.appendChild(verdict);
    cell(r.steps, "num");
    cell(`${r.assertions_passed}/${r.assertions}`, "num");
    cell(r.calls, "num");
    cell(num(r.input_tokens), "num");
    cell(num(r.output_tokens), "num");
    cell(num(r.cache_read_tokens), "num");
    cell(r.duration_seconds == null ? "—" : `${r.duration_seconds}s`, "num");
    const links = document.createElement("td");
    const link = document.createElement("a");
    link.href = `/api/runs/${r.run_id}/report.html`;
    link.target = "_blank";
    link.textContent = "report";
    links.appendChild(link);
    tr.appendChild(links);
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);

  const title = document.createElement("h2");
  title.textContent = "Results";
  const note = document.createElement("p");
  note.className = "hint";
  note.textContent =
    "Token counts cover the whole pipeline for each model — its own normalizer pass " +
    "included, since that is part of what choosing this model costs.";
  container.append(title, note, table);

  const specs = payload.results.filter((r) => r.normalized_instruction);
  if (specs.length) {
    const heading = document.createElement("h2");
    heading.textContent = "How each model read the request";
    const grid = document.createElement("div");
    grid.className = "compare-grid";
    for (const r of specs) {
      const card = document.createElement("div");
      card.className = "card normalized";
      const label = document.createElement("div");
      label.className = "label";
      label.textContent = r.model;
      card.append(label, textNode(r.normalized_instruction, true));
      grid.appendChild(card);
    }
    container.append(heading, grid);
  }
}

async function startComparison() {
  const instruction = $("instruction").value;
  if (!instruction.trim()) {
    alert("Please enter an instruction.");
    return;
  }
  const parsed = readTestData();
  if (!parsed.ok) return;
  const models = readModels();
  if (models.length < config.min_models) {
    alert(`Enter at least ${config.min_models} models to compare.`);
    return;
  }

  timeline.innerHTML = "";
  verdictEl.classList.add("hidden");
  compareEl.innerHTML = "";
  compareEl.classList.remove("hidden");
  runCompareBtn.disabled = true;
  runBtn.disabled = true;
  setStatus("normalizing");

  let comparison;
  try {
    comparison = await api("POST", "/api/compare", {
      instruction,
      target_url: $("target_url").value.trim() || null,
      data: parsed.data,
      models,
    });
  } catch (e) {
    setStatus("error");
    runCompareBtn.disabled = false;
    runBtn.disabled = false;
    alert("Failed to start comparison: " + e.message);
    return;
  }

  setStatus("running");
  const grid = document.createElement("div");
  grid.className = "compare-grid";
  const summary = document.createElement("div");
  compareEl.append(grid, summary);

  const proto = location.protocol === "https:" ? "wss" : "ws";
  let openSockets = comparison.runs.length;
  for (const entry of comparison.runs) {
    const col = compareColumn(entry);
    grid.appendChild(col);

    const ws = new WebSocket(`${proto}://${location.host}/ws/runs/${entry.run_id}`);
    ws.onmessage = (m) => handleCompareEvent(col, JSON.parse(m.data));
    // onclose and onerror can both fire for one socket; count each run only once.
    let settled = false;
    const finish = async () => {
      if (settled) return;
      settled = true;
      if (--openSockets > 0) return;
      runCompareBtn.disabled = false;
      runBtn.disabled = false;
      setStatus("done");
      try {
        renderCompareSummary(
          await api("GET", `/api/compare/${comparison.comparison_id}`), summary);
      } catch (e) {
        console.error("comparison results failed:", e);
      }
    };
    ws.onclose = finish;
    ws.onerror = finish;
  }
}

addModelBtn.addEventListener("click", () => addModel("", ""));
runCompareBtn.addEventListener("click", startComparison);

async function loadConfig() {
  try {
    config = { ...config, ...(await api("GET", "/api/config")) };
  } catch (e) {
    console.error("config load failed:", e);
  }
  // Three rows by default, the first on the configured model as the baseline.
  addModel(config.model, "");
  addModel("", "");
  addModel("", "");
}

loadConfig();
loadSuite();
