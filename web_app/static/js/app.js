const analyzeForm = document.getElementById("analyzeForm");
const trainForm = document.getElementById("trainForm");
const batchForm = document.getElementById("batchForm");
const resultCard = document.getElementById("resultCard");
const trainingOutput = document.getElementById("trainingOutput");
const batchOutput = document.getElementById("batchOutput");
const historyList = document.getElementById("historyList");
const logList = document.getElementById("logList");
const modelStatus = document.getElementById("modelStatus");
const modelStatusDetails = document.getElementById("modelStatusDetails");
const trainingStatus = document.getElementById("trainingStatus");
const clearHistoryButton = document.getElementById("clearHistoryButton");
const clearLogsButton = document.getElementById("clearLogsButton");

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error || "Request failed");
  }
  return payload;
}

function threatClass(threatLevel) {
  return String(threatLevel || "safe").toLowerCase().replace(/\s+/g, "-");
}

function renderResult(result) {
  const reasons = (result.reasoning || [])
    .map((reason) => `<li>${escapeHtml(reason)}</li>`)
    .join("");

  resultCard.className = "result-card";
  resultCard.innerHTML = `
    <div class="threat-row">
      <div>
        <p class="eyebrow">Detection Result</p>
        <h2>${escapeHtml(result.url)}</h2>
      </div>
      <span class="threat-badge ${threatClass(result.threat_level)}">${escapeHtml(result.threat_level)}</span>
    </div>
    <div class="score-grid">
      <div class="score-card">
        <span>Risk Score</span>
        <strong>${result.risk_score}</strong>
      </div>
      <div class="score-card">
        <span>Confidence</span>
        <strong>${result.confidence_score}%</strong>
      </div>
    </div>
    <p><strong>Recommended Action:</strong> ${escapeHtml(result.recommended_action)}</p>
    <p><strong>Model Source:</strong> ${escapeHtml(result.model_source)}</p>
    <p><strong>Reasons:</strong></p>
    <ul class="reason-list">${reasons}</ul>
  `;
}

function setTrainingStatus(status, message) {
  trainingStatus.className = "status-strip";
  const normalized = String(status || "").toUpperCase();
  if (normalized === "SUCCESS") {
    trainingStatus.classList.add("success");
  }
  if (normalized === "FAILED") {
    trainingStatus.classList.add("failed");
  }
  trainingStatus.textContent = message;
}

function renderError(target, message) {
  target.innerHTML = `<span class="error">${escapeHtml(message)}</span>`;
}

function renderBatch(results) {
  if (!results.length) {
    batchOutput.textContent = "No URLs were found in the uploaded CSV.";
    return;
  }

  batchOutput.innerHTML = results
    .slice(0, 50)
    .map(
      (result) => `
        <div class="history-item">
          <div>
            <strong>${escapeHtml(result.url)}</strong>
            <small>${escapeHtml(result.threat_level)} · Risk ${result.risk_score} · Confidence ${result.confidence_score}%</small>
          </div>
          <span class="threat-badge ${threatClass(result.threat_level)}">${escapeHtml(result.threat_level)}</span>
        </div>
      `
    )
    .join("");
}

function renderTrainingSummary(training) {
  const distribution = training.class_distribution || {};
  const metrics = training.test_metrics || {};
  const validation = training.validation || {};
  trainingOutput.innerHTML = `
    <div class="training-summary">
      <div class="training-summary-row">
        <span>Status</span>
        <strong>${escapeHtml(training.message || training.status || "Training completed")}</strong>
      </div>
      <div class="training-summary-row">
        <span>Best model</span>
        <strong>${escapeHtml(training.best_model_name || "Not available")}</strong>
      </div>
      <div class="training-summary-row">
        <span>Training records</span>
        <strong>${escapeHtml(training.cleaned_records || validation.records || "0")}</strong>
      </div>
      <div class="training-summary-row">
        <span>Class balance</span>
        <strong>Legitimate: ${escapeHtml(distribution.legitimate ?? 0)} · Phishing: ${escapeHtml(distribution.phishing ?? 0)}</strong>
      </div>
      <div class="training-summary-row">
        <span>Holdout F1</span>
        <strong>${escapeHtml(metrics.f1 ?? "Not available")}</strong>
      </div>
      <div class="training-summary-row">
        <span>Recall</span>
        <strong>${escapeHtml(metrics.recall ?? "Not available")}</strong>
      </div>
      <div class="training-summary-row">
        <span>Artifact status</span>
        <strong>${training.artifact_saved ? "Model artifact saved and activated" : "Artifact was not saved"}</strong>
      </div>
    </div>
  `;
}

async function refreshHealth() {
  try {
    const payload = await requestJson("/api/health");
    modelStatus.textContent = payload.model_loaded
      ? "Trained model loaded from local artifacts"
      : "Using fallback scoring until a CSV model is trained";
    renderModelStatus(payload.model);
  } catch (error) {
    modelStatus.textContent = "Health check unavailable";
  }
}

function renderModelStatus(model) {
  if (!model) {
    modelStatusDetails.innerHTML = `<p class="muted">Model status is not available.</p>`;
    return;
  }
  const training = model.training_status || {};
  modelStatusDetails.innerHTML = `
    <div class="status-tile">
      <span>Model Source</span>
      <strong>${escapeHtml(model.model_source)}</strong>
    </div>
    <div class="status-tile">
      <span>Best Model</span>
      <strong>${escapeHtml(model.best_model_name || "Not trained yet")}</strong>
    </div>
    <div class="status-tile">
      <span>Feature Count</span>
      <strong>${escapeHtml(model.feature_count)}</strong>
    </div>
    <div class="status-tile">
      <span>Training Status</span>
      <strong>${escapeHtml(training.status || "NO_TRAINING_RUN")}</strong>
    </div>
    <div class="status-tile">
      <span>Model Artifact</span>
      <strong>${escapeHtml(model.model_path || "No saved model artifact")}</strong>
    </div>
    <div class="status-tile">
      <span>Status Message</span>
      <strong>${escapeHtml(training.message || "No training run has been recorded.")}</strong>
    </div>
  `;
  setTrainingStatus(training.status, training.message || "No training run has started in this session.");
}

async function refreshHistory() {
  try {
    const payload = await requestJson("/api/history");
    const history = payload.history || [];
    if (!history.length) {
      historyList.innerHTML = `<p class="muted">No detections have been stored yet.</p>`;
      return;
    }
    historyList.innerHTML = history
      .slice(0, 12)
      .map(
        (item) => `
          <div class="history-item">
            <div>
              <strong>${escapeHtml(item.url)}</strong>
              <small>${escapeHtml(item.analyzed_at || "")} · ${escapeHtml(item.recommended_action || "")}</small>
            </div>
            <span class="threat-badge ${threatClass(item.threat_level)}">${escapeHtml(item.threat_level)}</span>
          </div>
        `
      )
      .join("");
  } catch (error) {
    renderError(historyList, error.message);
  }
}

async function refreshLogs() {
  try {
    const payload = await requestJson("/api/logs");
    const logs = payload.logs || [];
    if (!logs.length) {
      logList.innerHTML = `<p class="muted">No system logs have been stored yet.</p>`;
      return;
    }
    logList.innerHTML = logs
      .slice(0, 18)
      .map((item) => {
        const details = item.details || {};
        const summary = details.url || details.dataset_file || details.message || "";
        return `
          <div class="history-item">
            <div>
              <strong class="log-event">${escapeHtml(item.event_type)}</strong>
              <small>${escapeHtml(item.logged_at || "")} · ${escapeHtml(summary)}</small>
            </div>
          </div>
        `;
      })
      .join("");
  } catch (error) {
    renderError(logList, error.message);
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

analyzeForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = analyzeForm.querySelector("button");
  button.disabled = true;
  resultCard.className = "result-card empty";
  resultCard.textContent = "Analyzing URL...";

  try {
    const payload = await requestJson("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: document.getElementById("urlInput").value, source: "web" }),
    });
    renderResult(payload.result);
    await refreshHistory();
    await refreshLogs();
    await refreshHealth();
  } catch (error) {
    renderError(resultCard, error.message);
  } finally {
    button.disabled = false;
  }
});

trainForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const fileInput = document.getElementById("trainFile");
  const button = trainForm.querySelector("button");
  const formData = new FormData();
  formData.append("dataset", fileInput.files[0] || "");
  button.disabled = true;
  setTrainingStatus("TRAINING_STARTED", "Training started");
  trainingOutput.textContent = "Training started. Validating CSV and building model...";

  try {
    const payload = await requestJson("/api/train", { method: "POST", body: formData });
    setTrainingStatus(payload.training.status, payload.training.message || "Training successful");
    renderTrainingSummary(payload.training);
    await refreshHealth();
    await refreshLogs();
  } catch (error) {
    setTrainingStatus("FAILED", "Training failed");
    trainingOutput.textContent = error.message;
    await refreshLogs();
    await refreshHealth();
  } finally {
    button.disabled = false;
  }
});

batchForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const fileInput = document.getElementById("batchFile");
  const button = batchForm.querySelector("button");
  const formData = new FormData();
  formData.append("dataset", fileInput.files[0] || "");
  button.disabled = true;
  batchOutput.textContent = "Analyzing uploaded CSV...";

  try {
    const payload = await requestJson("/api/batch-analyze", { method: "POST", body: formData });
    renderBatch(payload.batch.results || []);
    await refreshHistory();
    await refreshLogs();
  } catch (error) {
    renderError(batchOutput, error.message);
  } finally {
    button.disabled = false;
  }
});

clearHistoryButton.addEventListener("click", async () => {
  await requestJson("/api/history", { method: "DELETE" });
  await refreshHistory();
  await refreshLogs();
});

clearLogsButton.addEventListener("click", async () => {
  await requestJson("/api/logs", { method: "DELETE" });
  await refreshLogs();
});

refreshHealth();
refreshHistory();
refreshLogs();
