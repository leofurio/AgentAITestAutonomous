"use strict";

const $ = (id) => document.getElementById(id);
const timeline = $("timeline");
const statusEl = $("status");
const verdictEl = $("verdict");
const runBtn = $("run");
const suiteList = $("suite-list");
const saveSuiteBtn = $("save-suite");
const runSuiteBtn = $("run-suite");

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

async function startRun() {
  let data = null;
  const rawData = $("data").value.trim();
  if (rawData) {
    try {
      data = JSON.parse(rawData);
    } catch (e) {
      alert("Test data is not valid JSON: " + e.message);
      return;
    }
  }
  const body = {
    instruction: $("instruction").value,
    target_url: $("target_url").value.trim() || null,
    data,
  };
  if (!body.instruction.trim()) {
    alert("Please enter an instruction.");
    return;
  }

  timeline.innerHTML = "";
  verdictEl.classList.add("hidden");
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
  mkBtn("▶ auto", "Replay the recording; self-heal via the agent if it broke",
    (b) => runSuiteTest(test.test_id, "auto", b));
  mkBtn("🤖 agent", "Full agentic run (re-records on pass)",
    (b) => runSuiteTest(test.test_id, "agent", b));
  mkBtn("✕", "Delete this suite test", async () => {
    if (!confirm(`Delete suite test "${test.name}"?`)) return;
    await api("DELETE", `/api/suite/${test.test_id}`);
    loadSuite();
  });
  item.appendChild(actions);
  return item;
}

async function loadSuite() {
  try {
    const data = await api("GET", "/api/suite");
    suiteList.innerHTML = "";
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
  let data = null;
  const rawData = $("data").value.trim();
  if (rawData) {
    try { data = JSON.parse(rawData); } catch (e) {
      alert("Test data is not valid JSON: " + e.message);
      return;
    }
  }
  try {
    await api("POST", "/api/suite", {
      name,
      instruction,
      target_url: $("target_url").value.trim() || null,
      data,
      source_run_id: lastRunId, // last completed live run becomes the recording
    });
    loadSuite();
  } catch (e) {
    alert("Save failed: " + e.message);
  }
});

runSuiteBtn.addEventListener("click", async () => {
  runSuiteBtn.disabled = true;
  runSuiteBtn.textContent = "Running suite…";
  try {
    const res = await api("POST", "/api/suite/run_all", { mode: "auto" });
    setStatus(res.passed === res.total ? "done" : "error");
    alert(`Suite: ${res.passed}/${res.total} passed`);
  } catch (e) {
    alert("Suite run failed: " + e.message);
  } finally {
    runSuiteBtn.disabled = false;
    runSuiteBtn.textContent = "Run all (auto)";
    loadSuite();
  }
});

loadSuite();
