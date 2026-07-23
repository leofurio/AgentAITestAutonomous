"use strict";

const codeEl = document.getElementById("code");
const timeoutEl = document.getElementById("timeout");
const executeBtn = document.getElementById("execute");
const autohealEl = document.getElementById("autoheal");
const statusEl = document.getElementById("runner-status");
const healStatusEl = document.getElementById("heal-status");
const reportLinkEl = document.getElementById("report-link");
const stdoutEl = document.getElementById("stdout");
const stderrEl = document.getElementById("stderr");

function setStatus(state) {
  statusEl.textContent = state;
  statusEl.className = "status " + state;
}

function setHeal(text) {
  healStatusEl.textContent = text || "";
}

// The run id of the recording behind the currently loaded script, if the runner was
// opened via a completed test's "run code" link (/runner?script=/api/runs/<id>/...).
// Healing needs the recorded step intent, so it is only offered when this is known.
function sourceRunId() {
  const script = new URLSearchParams(location.search).get("script");
  const match = script && script.match(/\/api\/runs\/([A-Za-z0-9_-]+)\/playwright_test\.py/);
  return match ? match[1] : null;
}

async function runVerdict(workDir) {
  // The replay writes report.json into its work dir; read back the verdict so auto-heal
  // fires on a broken locator (error) but never on a real regression (assertion fail).
  if (!workDir) return null;
  const runId = workDir.split(/[\\/]/).filter(Boolean).pop();
  try {
    const resp = await fetch(`/api/runs/${encodeURIComponent(runId)}/report.json`);
    if (!resp.ok) return null;
    return (await resp.json()).verdict || null;
  } catch (error) {
    return null;
  }
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

async function showReportLink(workDir) {
  // Replays write report.json/report.html into their work dir, which the API
  // serves under /api/runs/<dir name>/ — link it when it exists.
  reportLinkEl.innerHTML = "";
  if (!workDir) return;
  const runId = workDir.split(/[\\/]/).filter(Boolean).pop();
  if (!runId) return;
  const url = `/api/runs/${encodeURIComponent(runId)}/report.html`;
  try {
    const resp = await fetch(url, { method: "HEAD" });
    if (!resp.ok) return;
    const link = document.createElement("a");
    link.href = url;
    link.target = "_blank";
    link.textContent = "Open replay report";
    reportLinkEl.appendChild(link);
  } catch (error) {
    /* no report produced — leave the area empty */
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
  reportLinkEl.innerHTML = "";
  setHeal("");
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
    const ok = !data.timed_out && data.exit_code === 0;
    setStatus(ok ? "done" : "error");
    await showReportLink(data.work_dir);
    if (!ok && autohealEl.checked) {
      await maybeHeal(data.work_dir);
    }
  } catch (error) {
    stderrEl.textContent = error.message;
    setStatus("error");
  } finally {
    executeBtn.disabled = false;
  }
}

async function maybeHeal(workDir) {
  const runId = sourceRunId();
  if (!runId) {
    setHeal("Auto-heal skipped: open the script via a completed test's “run code” "
      + "link so the recording is known.");
    return;
  }
  const verdict = await runVerdict(workDir);
  if (verdict === "fail") {
    setHeal("Not healed: an assertion failed — a real regression, not a broken locator.");
    return;
  }
  await healRecording(runId);
}

async function healRecording(runId) {
  setStatus("healing");
  setHeal("Auto-healing: re-pointing the broken step against the live site…");
  try {
    const resp = await fetch("/api/playwright/heal", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        run_id: runId,
        timeout_seconds: Number(timeoutEl.value || 120),
      }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "HTTP " + resp.status);
    if (data.verdict === "pass") {
      setStatus("done");
      setHeal(data.repaired
        ? "Healed ✓ re-pointed broken locator(s). Load the corrected script below."
        : "Replayed clean ✓ no locator repair was needed.");
      showHealResult(data);
    } else if (data.verdict === "fail") {
      setStatus("error");
      setHeal("Not a broken locator: assertions failed (real regression). Verdict unchanged.");
      showHealResult(data);
    } else {
      setStatus("error");
      setHeal("Could not auto-heal: " + (data.summary || "the site may have changed structurally."));
    }
  } catch (error) {
    setStatus("error");
    setHeal("Heal failed: " + error.message);
  }
}

function showHealResult(data) {
  reportLinkEl.innerHTML = "";
  if (data.script_url) {
    const load = document.createElement("button");
    load.textContent = "Load healed script";
    load.addEventListener("click", async () => {
      try {
        const resp = await fetch(data.script_url);
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        codeEl.value = await resp.text();
        // Point future runs/heals at the corrected recording.
        history.replaceState(null, "", `/runner?script=${encodeURIComponent(data.script_url)}`);
        setHeal("Loaded the healed script. Run it to confirm it now passes deterministically.");
      } catch (error) {
        setHeal("Could not load healed script: " + error.message);
      }
    });
    reportLinkEl.appendChild(load);
  }
  if (data.report_url) {
    const link = document.createElement("a");
    link.href = data.report_url;
    link.target = "_blank";
    link.textContent = "Open heal report";
    reportLinkEl.appendChild(link);
  }
}

executeBtn.addEventListener("click", executeCode);
loadScriptFromQuery();
