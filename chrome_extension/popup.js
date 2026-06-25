const API_URL = "http://127.0.0.1:5000/api/analyze";
const WEB_APP_URL = "http://127.0.0.1:5000";

const currentUrlElement = document.getElementById("currentUrl");
const resultPanel = document.getElementById("resultPanel");
const scanButton = document.getElementById("scanButton");
const openWebAppButton = document.getElementById("openWebAppButton");

let currentTabUrl = "";

function threatClass(threatLevel) {
  return String(threatLevel || "safe").toLowerCase().replace(/\s+/g, "-");
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

async function getCurrentTab() {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  return tabs[0];
}

async function loadCurrentUrl() {
  const tab = await getCurrentTab();
  currentTabUrl = tab?.url || "";
  currentUrlElement.textContent = currentTabUrl || "No active URL detected";
}

async function analyzeCurrentUrl() {
  if (!currentTabUrl || !/^https?:\/\//i.test(currentTabUrl)) {
    resultPanel.textContent = "Only HTTP and HTTPS URLs can be analyzed.";
    resultPanel.className = "result-panel muted";
    return;
  }

  scanButton.disabled = true;
  resultPanel.textContent = "Analyzing current page URL...";
  resultPanel.className = "result-panel muted";

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: currentTabUrl, source: "chrome_extension" }),
    });
    const payload = await response.json();
    if (!response.ok || payload.ok === false) {
      throw new Error(payload.error || "Local detection service is not available.");
    }
    renderResult(payload.result);
    await chrome.storage.local.set({ lastDetection: payload.result });
  } catch (error) {
    resultPanel.innerHTML = `
      <strong>Detection service unavailable</strong>
      <p class="muted">Start the local web app at ${WEB_APP_URL}, then run the scan again.</p>
      <p>${escapeHtml(error.message)}</p>
    `;
  } finally {
    scanButton.disabled = false;
  }
}

function renderResult(result) {
  const reasons = (result.reasoning || [])
    .slice(0, 4)
    .map((reason) => `<li>${escapeHtml(reason)}</li>`)
    .join("");

  resultPanel.className = "result-panel";
  resultPanel.innerHTML = `
    <span class="badge ${threatClass(result.threat_level)}">${escapeHtml(result.threat_level)}</span>
    <div class="score-row">
      <div class="score-card">
        <span>Risk</span>
        <strong>${result.risk_score}</strong>
      </div>
      <div class="score-card">
        <span>Confidence</span>
        <strong>${result.confidence_score}%</strong>
      </div>
    </div>
    <strong>${escapeHtml(result.recommended_action)}</strong>
    <ul>${reasons}</ul>
  `;
}

openWebAppButton.addEventListener("click", () => {
  chrome.tabs.create({ url: WEB_APP_URL });
});

scanButton.addEventListener("click", analyzeCurrentUrl);
loadCurrentUrl();
