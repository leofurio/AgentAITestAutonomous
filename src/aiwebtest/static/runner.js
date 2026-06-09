"use strict";

const codeEl = document.getElementById("code");
const timeoutEl = document.getElementById("timeout");
const executeBtn = document.getElementById("execute");
const statusEl = document.getElementById("runner-status");
const stdoutEl = document.getElementById("stdout");
const stderrEl = document.getElementById("stderr");

function setStatus(state) {
  statusEl.textContent = state;
  statusEl.className = "status " + state;
}

async function loadScriptFromQuery() {
  const scriptUrl = new URLSearchParams(location.search).get("script");
  if (!scriptUrl) return;

  try {
    const resp = await fetch(scriptUrl);
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    codeEl.value = await resp.text();
  } catch (error) {
    setStatus("error");
    stderrEl.textContent = "Failed to load script: " + error.message;
  }
}

async function executeCode() {
  const code = codeEl.value.trim();
  if (!code) {
    alert("Paste Python Playwright code first.");
    return;
  }

  executeBtn.disabled = true;
  stdoutEl.textContent = "";
  stderrEl.textContent = "";
  setStatus("running");

  try {
    const resp = await fetch("/api/playwright/execute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        code,
        timeout_seconds: Number(timeoutEl.value || 120),
      }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "HTTP " + resp.status);
    stdoutEl.textContent = data.stdout || "";
    stderrEl.textContent = data.stderr || "";
    setStatus(data.timed_out ? "error" : data.exit_code === 0 ? "done" : "error");
  } catch (error) {
    stderrEl.textContent = error.message;
    setStatus("error");
  } finally {
    executeBtn.disabled = false;
  }
}

executeBtn.addEventListener("click", executeCode);
loadScriptFromQuery();
