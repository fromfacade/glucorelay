const page = document.body.dataset.page;
const POLL_INTERVAL_MS = 2500;
const COUNTDOWN_SECONDS = 30;
const TREND_ICONS = {
  double_down: "↓↓",
  single_down: "↓",
  flat: "→",
  single_up: "↑",
  double_up: "↑↑",
  unknown: "?",
};

let currentState = null;
let countdownTimer = null;
let countdownRemaining = COUNTDOWN_SECONDS;
let lastEventId = null;
let lastEventStatus = null;
let patientChart = null;
let caregiverChart = null;
let lastGemmaStateKey = null;
let chartPoints = loadChartPoints();

function $(id) {
  return document.getElementById(id);
}

function loadChartPoints() {
  try {
    return JSON.parse(sessionStorage.getItem("glucorelay_chart") || "[]");
  } catch {
    return [];
  }
}

function persistChartPoints() {
  sessionStorage.setItem("glucorelay_chart", JSON.stringify(chartPoints.slice(-40)));
}

function formatDate(value) {
  if (!value) return "—";
  return new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(value));
}

function friendlyStatus(value) {
  if (!value) return "No active event";
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function friendlyResponse(value) {
  if (!value) return "—";
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

async function loadGemmaExplanation(force = false) {
  const event = currentState?.active_event;
  const output = $("aiExplanation");
  const status = $("aiStatus");
  const button = $("explainButton");
  if (!output || !status || !button) return;

  if (!event) {
    output.textContent = page === "caregiver"
      ? "No active event to summarize."
      : "Submit an abnormal reading to create an event and generate an explanation.";
    status.textContent = "Waiting for an event";
    button.disabled = true;
    lastGemmaStateKey = null;
    return;
  }

  const stateKey = [
    event.id,
    event.status,
    event.latest_reading.id,
    event.patient_response || "",
    event.acknowledged_by || "",
  ].join(":");

  button.disabled = false;
  if (!force && stateKey === lastGemmaStateKey) return;
  lastGemmaStateKey = stateKey;
  output.classList.add("loading");
  output.textContent = "Gemma is reviewing the current event...";
  status.textContent = "Generating";
  button.disabled = true;

  try {
    const data = await api(`/api/events/${event.id}/ai-explanation`);
    output.textContent = data.explanation;
    status.textContent = data.model;
  } catch (error) {
    output.textContent = `${error.message}. Configure GEMMA_ENABLED, GEMMA_API_URL, and GEMMA_MODEL in your .env file.`;
    status.textContent = "Unavailable";
  } finally {
    output.classList.remove("loading");
    button.disabled = !currentState?.active_event;
  }
}

function showToast(message, type = "default") {
  const region = $("toastRegion");
  if (!region) return;
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  region.appendChild(toast);
  setTimeout(() => toast.remove(), 3600);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch {}
    throw new Error(detail);
  }
  return response.json();
}

function appendChartReading(reading) {
  if (!reading) return;
  const alreadyPresent = chartPoints.some((point) => point.id === reading.id);
  if (alreadyPresent) return;
  chartPoints.push({
    id: reading.id,
    x: reading.recorded_at,
    y: reading.value_mg_dl,
  });
  chartPoints = chartPoints.slice(-40);
  persistChartPoints();
  updateCharts();
}

function chartConfig() {
  return {
    type: "line",
    data: {
      labels: chartPoints.map((point) => formatDate(point.x)),
      datasets: [{
        label: "Glucose (mg/dL)",
        data: chartPoints.map((point) => point.y),
        borderWidth: 3,
        tension: 0.3,
        pointRadius: 4,
        pointHoverRadius: 6,
        fill: false,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { intersect: false, mode: "index" },
      scales: {
        y: { suggestedMin: 40, suggestedMax: 300, title: { display: true, text: "mg/dL" } },
      },
      plugins: { legend: { display: false } },
    },
  };
}

function initCharts() {
  if ($("glucoseChart") && window.Chart) patientChart = new Chart($("glucoseChart"), chartConfig());
  if ($("caregiverChart") && window.Chart) caregiverChart = new Chart($("caregiverChart"), chartConfig());
}

function updateCharts() {
  for (const chart of [patientChart, caregiverChart]) {
    if (!chart) continue;
    chart.data.labels = chartPoints.map((point) => formatDate(point.x));
    chart.data.datasets[0].data = chartPoints.map((point) => point.y);
    chart.update();
  }
}

function readingTone(reading, thresholds) {
  if (!reading || !thresholds) return "neutral";
  if (reading.value_mg_dl <= thresholds.urgent_low) return "danger";
  if (reading.value_mg_dl <= thresholds.low || reading.value_mg_dl >= thresholds.high) return "warning";
  return "success";
}

function updatePatientPage(state) {
  const reading = state.latest_reading;
  const event = state.active_event;
  const tone = readingTone(reading, state.thresholds);

  $("connectionBadge").textContent = "Live";
  $("connectionBadge").className = "status-pill success";
  $("urgentLowValue").textContent = `${state.thresholds.urgent_low} mg/dL`;
  $("lowValue").textContent = `${state.thresholds.low} mg/dL`;
  $("highValue").textContent = `${state.thresholds.high} mg/dL`;

  if (reading) {
    $("readingValue").textContent = reading.value_mg_dl;
    $("readingMeta").textContent = `${friendlyResponse(reading.trend)} · ${reading.source} · ${formatDate(reading.recorded_at)}`;
    $("trendIcon").textContent = TREND_ICONS[reading.trend] || "?";
    $("trendIcon").setAttribute("aria-label", friendlyResponse(reading.trend));
    $("readingValue").dataset.tone = tone;
    appendChartReading(reading);
  }

  if (!event) {
    $("eventStatus").textContent = "No active event";
    $("eventStatus").className = "status-pill neutral";
    $("eventEmpty").classList.remove("hidden");
    $("eventDetails").classList.add("hidden");
  } else {
    $("eventEmpty").classList.add("hidden");
    $("eventDetails").classList.remove("hidden");
    $("eventStatus").textContent = friendlyStatus(event.status);
    $("eventStatus").className = `status-pill ${event.status === "contacting" ? "danger" : event.status === "acknowledged" ? "success" : "warning"}`;
    $("eventReason").textContent = event.reason;
    $("eventUpdated").textContent = `Updated ${formatDate(event.updated_at)}`;
    $("eventAck").textContent = event.acknowledged_by ? `Acknowledged by ${event.acknowledged_by}` : "Not yet acknowledged";
    updateStateTrack(event.status);
  }

  $("timeoutButton").disabled = !event || !["check_in_required", "monitoring"].includes(event.status);
  $("resolveButton").disabled = !event;
  renderHistory(state.history);
  syncEmergencyOverlay(event);
  loadGemmaExplanation();
}

function updateStateTrack(status) {
  const order = ["check_in_required", "monitoring", "contacting", "acknowledged"];
  const currentIndex = order.indexOf(status);
  document.querySelectorAll(".state-track [data-state]").forEach((node) => {
    node.classList.remove("active", "complete");
    const index = order.indexOf(node.dataset.state);
    if (index < currentIndex) node.classList.add("complete");
    if (index === currentIndex) node.classList.add("active");
  });
}

function renderHistory(history) {
  const container = $("historyList");
  if (!container) return;
  if (!history.length) {
    container.innerHTML = '<p class="muted">No resolved events yet.</p>';
    return;
  }
  container.innerHTML = history.map((event) => `
    <div class="history-item">
      <div>
        <strong>${event.latest_reading.value_mg_dl} mg/dL · ${friendlyStatus(event.status)}</strong>
        <span>${event.reason}</span>
      </div>
      <span>${formatDate(event.updated_at)}</span>
    </div>
  `).join("");
}

function shouldShowOverlay(event) {
  return event && ["check_in_required", "monitoring"].includes(event.status);
}

function syncEmergencyOverlay(event) {
  const overlay = $("emergencyOverlay");
  if (!overlay) return;

  if (!shouldShowOverlay(event)) {
    overlay.classList.add("hidden");
    stopCountdown();
    return;
  }

  $("overlayMessage").textContent = `${event.reason}. Latest reading: ${event.latest_reading.value_mg_dl} mg/dL.`;
  overlay.classList.remove("hidden");

  if (event.id !== lastEventId || event.status !== lastEventStatus) {
    startCountdown(event.id);
  }
}

function startCountdown(eventId) {
  stopCountdown();
  countdownRemaining = COUNTDOWN_SECONDS;
  $("countdownValue").textContent = countdownRemaining;
  countdownTimer = setInterval(async () => {
    countdownRemaining -= 1;
    $("countdownValue").textContent = Math.max(0, countdownRemaining);
    if (countdownRemaining <= 0) {
      stopCountdown();
      try {
        await api(`/api/events/${eventId}/timeout`, { method: "POST" });
        showToast("No response received. Caregiver escalation started.", "error");
        await refreshState();
      } catch (error) {
        showToast(error.message, "error");
      }
    }
  }, 1000);
}

function stopCountdown() {
  if (countdownTimer) clearInterval(countdownTimer);
  countdownTimer = null;
}

async function handlePatientResponse(response) {
  const event = currentState?.active_event;
  if (!event) return;
  $("overlayStatus").textContent = "Updating…";
  try {
    await api(`/api/events/${event.id}/patient-response`, {
      method: "POST",
      body: JSON.stringify({ response }),
    });
    showToast(response === "need_help" ? "Caregiver contact started." : "Response recorded.", response === "need_help" ? "error" : "success");
    await refreshState();
  } catch (error) {
    showToast(error.message, "error");
  } finally {
    $("overlayStatus").textContent = "";
  }
}

function updateCaregiverPage(state) {
  const reading = state.latest_reading;
  const event = state.active_event;
  const hero = $("caregiverHero");

  if (reading) {
    $("caregiverReading").textContent = reading.value_mg_dl;
    $("caregiverReadingMeta").textContent = `${friendlyResponse(reading.trend)} · ${reading.source} · ${formatDate(reading.recorded_at)}`;
    $("caregiverTrend").textContent = TREND_ICONS[reading.trend] || "?";
    appendChartReading(reading);
  }

  hero.className = "caregiver-hero calm";
  $("caregiverStatus").className = "status-pill neutral";

  if (!event) {
    $("caregiverHeadline").textContent = "No active emergency";
    $("caregiverSubhead").textContent = "This page checks for updates automatically.";
    $("caregiverStatus").textContent = "Standing by";
    $("caregiverActionEmpty").classList.remove("hidden");
    $("ackForm").classList.add("hidden");
    $("caregiverResolveButton").classList.add("hidden");
    fillDetails(null);
    loadGemmaExplanation();
    return;
  }

  const tone = event.status === "contacting" ? "danger" : event.status === "acknowledged" ? "success" : "warning";
  hero.className = `caregiver-hero ${tone}`;
  $("caregiverHeadline").textContent = event.status === "contacting" ? "Patient may need immediate help" : friendlyStatus(event.status);
  $("caregiverSubhead").textContent = `${event.reason} Latest reading: ${event.latest_reading.value_mg_dl} mg/dL.`;
  $("caregiverStatus").textContent = friendlyStatus(event.status);
  $("caregiverStatus").className = `status-pill ${tone}`;

  const needsAck = event.status === "contacting";
  $("caregiverActionEmpty").classList.toggle("hidden", needsAck || event.status === "acknowledged");
  $("ackForm").classList.toggle("hidden", !needsAck);
  $("caregiverResolveButton").classList.toggle("hidden", event.status !== "acknowledged");
  fillDetails(event);
  loadGemmaExplanation();
}

function fillDetails(event) {
  $("detailEventId").textContent = event?.id || "—";
  $("detailStatus").textContent = event ? friendlyStatus(event.status) : "—";
  $("detailReason").textContent = event?.reason || "—";
  $("detailPatientResponse").textContent = event ? friendlyResponse(event.patient_response) : "—";
  $("detailAck").textContent = event?.acknowledged_by || "—";
  $("detailUpdated").textContent = event ? formatDate(event.updated_at) : "—";
}

async function refreshState() {
  try {
    const state = await api("/api/state");
    const previousEventId = lastEventId;
    const previousStatus = lastEventStatus;
    currentState = state;

    if (page === "patient") updatePatientPage(state);
    if (page === "caregiver") updateCaregiverPage(state);

    const event = state.active_event;
    lastEventId = event?.id || null;
    lastEventStatus = event?.status || null;

    if (page === "caregiver" && event && (event.id !== previousEventId || event.status !== previousStatus)) {
      if (event.status === "contacting") showToast("New caregiver acknowledgement requested.", "error");
      if (event.status === "acknowledged") showToast(`Acknowledged by ${event.acknowledged_by}.`, "success");
    }
  } catch (error) {
    if ($("connectionBadge")) {
      $("connectionBadge").textContent = "Offline";
      $("connectionBadge").className = "status-pill danger";
    }
    showToast(error.message, "error");
  }
}

function bindPatientControls() {
  $("explainButton")?.addEventListener("click", () => loadGemmaExplanation(true));
  $("readingForm")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = $("sendReadingButton");
    button.disabled = true;
    try {
      await api("/api/readings", {
        method: "POST",
        body: JSON.stringify({
          value_mg_dl: Number($("readingInput").value),
          trend: $("trendInput").value,
          source: "simulator",
        }),
      });
      showToast("Reading sent.", "success");
      await refreshState();
    } catch (error) {
      showToast(error.message, "error");
    } finally {
      button.disabled = false;
    }
  });

  document.querySelectorAll(".preset").forEach((button) => {
    button.addEventListener("click", () => {
      $("readingInput").value = button.dataset.value;
      $("trendInput").value = button.dataset.trend;
    });
  });

  $("treatingButton")?.addEventListener("click", () => handlePatientResponse("treating"));
  $("helpButton")?.addEventListener("click", () => handlePatientResponse("need_help"));
  $("falseAlarmButton")?.addEventListener("click", () => handlePatientResponse("false_alarm"));

  $("timeoutButton")?.addEventListener("click", async () => {
    const event = currentState?.active_event;
    if (!event) return;
    try {
      await api(`/api/events/${event.id}/timeout`, { method: "POST" });
      showToast("Timeout escalation triggered.", "error");
      await refreshState();
    } catch (error) {
      showToast(error.message, "error");
    }
  });

  $("resolveButton")?.addEventListener("click", async () => {
    const event = currentState?.active_event;
    if (!event) return;
    try {
      await api(`/api/events/${event.id}/resolve`, { method: "POST" });
      showToast("Event resolved.", "success");
      await refreshState();
    } catch (error) {
      showToast(error.message, "error");
    }
  });

  $("resetButton")?.addEventListener("click", async () => {
    try {
      await api("/api/reset", { method: "POST" });
      chartPoints = [];
      persistChartPoints();
      updateCharts();
      showToast("Demo reset.", "success");
      await refreshState();
    } catch (error) {
      showToast(error.message, "error");
    }
  });

  $("clearChartButton")?.addEventListener("click", () => {
    chartPoints = [];
    persistChartPoints();
    updateCharts();
  });
}

function bindCaregiverControls() {
  $("explainButton")?.addEventListener("click", () => loadGemmaExplanation(true));
  $("ackForm")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const activeEvent = currentState?.active_event;
    if (!activeEvent) return;
    const name = $("caregiverName").value.trim();
    if (!name) return;
    try {
      await api(`/api/events/${activeEvent.id}/acknowledge`, {
        method: "POST",
        body: JSON.stringify({ caregiver_name: name }),
      });
      showToast("Event acknowledged.", "success");
      await refreshState();
    } catch (error) {
      showToast(error.message, "error");
    }
  });

  $("caregiverResolveButton")?.addEventListener("click", async () => {
    const activeEvent = currentState?.active_event;
    if (!activeEvent) return;
    try {
      await api(`/api/events/${activeEvent.id}/resolve`, { method: "POST" });
      showToast("Event resolved.", "success");
      await refreshState();
    } catch (error) {
      showToast(error.message, "error");
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initCharts();
  if (page === "patient") bindPatientControls();
  if (page === "caregiver") bindCaregiverControls();
  refreshState();
  setInterval(refreshState, POLL_INTERVAL_MS);
});
