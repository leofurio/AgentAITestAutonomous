"use strict";

const $ = (id) => document.getElementById(id);
const timeline = $("timeline");
const statusEl = $("status");
const verdictEl = $("verdict");
const runBtn = $("run");

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
  ws.onmessage = (m) => handleEvent(JSON.parse(m.data));
  ws.onclose = () => {
    runBtn.disabled = false;
  };
  ws.onerror = () => {
    setStatus("error");
    runBtn.disabled = false;
  };
}

runBtn.addEventListener("click", startRun);
