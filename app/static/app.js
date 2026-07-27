const page = document.body.dataset.page;
const POLL_MS = 1500;
const DEFAULT_CAREGIVER_NAME = "Helper";
let state = null;
let activeEventId = null;
let countdownInterval = null;
let lastCountdownCue = null;
let gemmaSubmitting = false;
let gemmaResultEventId = null;
let audioEnabled = localStorage.getItem("glucorelayAudioEnabled") !== "false";
let audioUnlocked = false;
let lastSpokenEventKey = sessionStorage.getItem("glucorelayLastSpokenEventKey") || "";
let lastCaregiverEventKey = sessionStorage.getItem("glucorelayLastCaregiverEventKey") || "";
let sessionReadings = JSON.parse(sessionStorage.getItem("glucorelayReadings") || "[]");

const $ = (id) => document.getElementById(id);
const fmt = (value) => String(value || "").replaceAll("_", " ");
const formatTime = (iso) => iso ? new Intl.DateTimeFormat(undefined,{hour:"numeric",minute:"2-digit",second:"2-digit"}).format(new Date(iso)) : "--";

// Handles both FastAPI error shapes this backend can return:
//   {"detail": "message"}
//   {"detail": {"message": "...", "current_status": "...", "allowed_transitions": [...]}}
function getApiError(data) {
  if (typeof data?.detail === "string") return data.detail;
  if (data?.detail?.message) return data.detail.message;
  if (data?.detail?.current_status) {
    return `This action is no longer available. Current status: ${data.detail.current_status}.`;
  }
  return "Request failed.";
}

async function api(path, options = {}) {
  const response = await fetch(path, {headers:{"Content-Type":"application/json"}, ...options});
  let data = null;
  try { data = await response.json(); } catch { data = {}; }
  if ($("apiOutput")) $("apiOutput").textContent = JSON.stringify(data, null, 2);
  if (!response.ok) {
    const error = new Error(getApiError(data));
    error.status = response.status;
    error.data = data;
    throw error;
  }
  return data;
}

function toast(message, type = "info") {
  const region = $("toastRegion");
  if (!region) return;
  const item = document.createElement("div");
  item.className = `toast ${type}`;
  item.textContent = message;
  region.append(item);
  setTimeout(() => item.remove(), 4200);
}

function updateAudioButton() {
  const button = $("audioToggleBtn");
  if (!button) return;
  button.textContent = audioEnabled ? "Voice alerts on" : "Voice alerts off";
  button.setAttribute("aria-pressed", String(audioEnabled));
  button.title = audioEnabled
    ? "Disable spoken alerts and tones"
    : "Enable spoken alerts and tones";
}

function unlockAudio() {
  if (audioUnlocked) return;
  audioUnlocked = true;
  try {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (AudioContextClass) {
      const context = new AudioContextClass();
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      gain.gain.value = 0;
      oscillator.connect(gain);
      gain.connect(context.destination);
      oscillator.start();
      oscillator.stop(context.currentTime + 0.01);
      context.resume?.();
      setTimeout(() => context.close?.(), 100);
    }
  } catch (error) {
    console.debug("Audio unlock skipped", error);
  }
}

function speak(message, { interrupt = true, rate = 0.95 } = {}) {
  if (!audioEnabled || !audioUnlocked || !("speechSynthesis" in window)) return;
  if (interrupt) window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(message);
  utterance.rate = rate;
  utterance.pitch = 1;
  utterance.volume = 1;
  window.speechSynthesis.speak(utterance);
}

function playAlertTone(level = "warning") {
  if (!audioEnabled || !audioUnlocked) return;
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) return;
  try {
    const context = new AudioContextClass();
    const frequencies = level === "urgent" ? [880, 660, 880] : [660, 520];
    frequencies.forEach((frequency, index) => {
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      const start = context.currentTime + index * 0.2;
      oscillator.type = "sine";
      oscillator.frequency.value = frequency;
      gain.gain.setValueAtTime(0.0001, start);
      gain.gain.exponentialRampToValueAtTime(0.18, start + 0.025);
      gain.gain.exponentialRampToValueAtTime(0.0001, start + 0.16);
      oscillator.connect(gain);
      gain.connect(context.destination);
      oscillator.start(start);
      oscillator.stop(start + 0.18);
    });
    setTimeout(() => context.close?.(), frequencies.length * 250 + 200);
  } catch (error) {
    console.debug("Alert tone unavailable", error);
  }
}

function announcePatientEvent(event, thresholds) {
  if (!event) return;
  const key = `${event.id}:${event.status}:${event.latest_reading.id}`;
  if (key === lastSpokenEventKey) return;
  lastSpokenEventKey = key;
  sessionStorage.setItem("glucorelayLastSpokenEventKey", key);
  const value = event.latest_reading.value_mg_dl;
  const level = readingClass(event.latest_reading, thresholds);
  playAlertTone(level);
  if (event.status === "contacting") {
    speak(`Urgent glucose alert. The reading is ${value} milligrams per deciliter. Caregiver contact has started.`);
  } else if (event.status === "acknowledged") {
    speak(`${event.acknowledged_by || DEFAULT_CAREGIVER_NAME} is responding.`);
  } else if (event.status === "check_in_required") {
    speak(`Glucose warning. The reading is ${value} milligrams per deciliter. Please respond before the countdown ends.`);
  }
}

function announceCaregiverEvent(event) {
  if (!event) return;
  const key = `${event.id}:${event.status}:${event.updated_at}`;
  if (key === lastCaregiverEventKey) return;
  lastCaregiverEventKey = key;
  sessionStorage.setItem("glucorelayLastCaregiverEventKey", key);
  if (event.status === "contacting") {
    playAlertTone("urgent");
    speak(`New GlucoRelay alert. Patient glucose is ${event.latest_reading.value_mg_dl} milligrams per deciliter. Acknowledgement is requested.`);
  } else if (event.status === "acknowledged") {
    speak(`Alert acknowledged by ${event.acknowledged_by || DEFAULT_CAREGIVER_NAME}.`);
  }
}

function toggleAudio() {
  audioEnabled = !audioEnabled;
  localStorage.setItem("glucorelayAudioEnabled", String(audioEnabled));
  if (audioEnabled) {
    unlockAudio();
    speak("Audio feedback enabled.");
  } else if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
  updateAudioButton();
}

function setConnection(ok) {
  const pill = $("connectionPill");
  if (!pill) return;
  pill.textContent = ok ? "Live" : "Disconnected";
  pill.className = `pill ${ok ? "pill-online" : "pill-offline"}`;
}

function readingClass(reading, thresholds) {
  if (!reading) return "neutral";
  if (reading.value_mg_dl <= thresholds.urgent_low) return "urgent";
  if (reading.value_mg_dl <= thresholds.low || reading.value_mg_dl >= thresholds.high) return "warning";
  return "normal";
}

function recordSessionReading(reading) {
  if (!reading || sessionReadings.some((r) => r.id === reading.id)) return;
  sessionReadings.push(reading);
  sessionReadings = sessionReadings.slice(-30);
  sessionStorage.setItem("glucorelayReadings", JSON.stringify(sessionReadings));
}

function eventStep(status) {
  const map = {check_in_required:2, monitoring:2, contacting:3, acknowledged:4};
  return map[status] || 0;
}

function renderWorkflow(status) {
  const current = eventStep(status);
  ["stepDetected","stepCheckin","stepContact","stepAck"].forEach((id, i) => {
    const el = $(id); if (!el) return;
    const step = i + 1;
    el.classList.toggle("done", current > step);
    el.classList.toggle("active", current === step || (step === 1 && current > 0));
  });
}

function renderHistory(history = []) {
  const list = $("historyList"); if (!list) return;
  if (!history.length) { list.innerHTML = '<p class="muted">No resolved events yet.</p>'; return; }
  list.innerHTML = history.slice(0,8).map((event) => `
    <div class="history-item">
      <span class="history-dot"></span>
      <div><strong>${event.latest_reading.value_mg_dl} mg/dL · ${fmt(event.reason)}</strong><small>${fmt(event.patient_response || "Resolved")}</small></div>
      <small>${formatTime(event.updated_at)}</small>
    </div>`).join("");
}

// --- Patient dashboard: a single state-driven renderer ---------------------
//
// `renderApp` is the only place that decides what's visible on the patient
// dashboard. Every sub-renderer derives its output from the backend event
// object (never from which button was last clicked), so the UI can never
// get stuck showing a prompt/countdown/form the backend no longer considers
// active - it's corrected on the very next poll regardless of what caused
// the status to change (a button, a Gemma check-in, a backend timeout, or
// the caregiver acknowledging from a different tab).

function renderApp(s) {
  state = s;
  const event = s.active_event;
  activeEventId = event?.id ?? null;

  renderReading(s);
  renderWorkflow(event?.status);
  renderPatientControls(event);
  renderGemmaCheckIn(event);
  renderOverlay(event);
  renderHistory(s.history);
  drawChart(s.thresholds);
  startCountdown(event);
  announcePatientEvent(event, s.thresholds);
}

function renderReading(s) {
  const reading = s.latest_reading;
  const event = s.active_event;
  recordSessionReading(reading);
  const level = readingClass(reading, s.thresholds);
  $("readingCard").className = `card reading-card state-${level}`;
  $("readingValue").textContent = reading ? reading.value_mg_dl : "--";
  $("readingBadge").textContent = level === "neutral" ? "No data" : level;
  $("trendValue").textContent = reading ? `Trend: ${fmt(reading.trend)}` : "Trend unknown";
  $("readingTime").textContent = reading ? `Recorded ${formatTime(reading.recorded_at)}` : "Waiting for a reading";
  $("urgentThreshold").textContent = `Urgent ≤ ${s.thresholds.urgent_low}`;
  $("lowThreshold").textContent = `Low ≤ ${s.thresholds.low}`;
  $("highThreshold").textContent = `High ≥ ${s.thresholds.high}`;

  $("eventTitle").textContent = event ? "Emergency workflow active" : "No active event";
  $("eventStatus").textContent = event ? fmt(event.status) : "Idle";
  $("eventReason").textContent = event?.reason || "A warning will appear automatically when the rules engine detects an abnormal reading.";
}

function showCheckInPrompt() {
  $("checkInPrompt").hidden = false;
  $("patientStatusMessage").hidden = true;
}

function hideCheckInPrompt() {
  $("checkInPrompt").hidden = true;
}

function showPatientStatusMessage(text) {
  const el = $("patientStatusMessage");
  if (!el) return;
  el.hidden = false;
  el.textContent = text;
}

// The prompt (the three response buttons + the Gemma check-in form) is only
// ever visible while `event.status === "check_in_required"` - every other
// status hides it here, in one place, instead of leaving visibility to
// whichever button handler last ran.
function renderPatientControls(event) {
  const treatingBtn = $("treatingBtn");
  const helpBtn = $("helpBtn");
  const falseAlarmBtn = $("falseAlarmBtn");
  const resolveBtn = $("resolveBtn");
  const timeoutBtn = $("timeoutBtn");

  const checkInActive = event?.status === "check_in_required";
  treatingBtn.disabled = !checkInActive;
  helpBtn.disabled = !checkInActive;
  falseAlarmBtn.disabled = !checkInActive;

  switch (event?.status) {
    case "check_in_required":
      showCheckInPrompt();
      resolveBtn.disabled = false;
      break;
    case "monitoring":
      hideCheckInPrompt();
      showPatientStatusMessage("Check-in received. GlucoRelay will continue monitoring the event.");
      resolveBtn.disabled = false;
      break;
    case "contacting":
      hideCheckInPrompt();
      showPatientStatusMessage("Caregiver contact has started. Waiting for acknowledgement from the caregiver console.");
      resolveBtn.disabled = false;
      break;
    case "acknowledged":
      hideCheckInPrompt();
      showPatientStatusMessage(`${event.acknowledged_by || DEFAULT_CAREGIVER_NAME} is responding.`);
      resolveBtn.disabled = false;
      break;
    case "resolved":
      hideCheckInPrompt();
      showPatientStatusMessage("This event has been resolved.");
      resolveBtn.disabled = true;
      break;
    default:
      hideCheckInPrompt();
      showPatientStatusMessage("No active event. Submit a reading to begin a demo scenario.");
      resolveBtn.disabled = true;
  }

  timeoutBtn.disabled = !event || !["check_in_required","monitoring"].includes(event.status);
}

// --- AI-assisted (Gemma) check-in -------------------------------------------

function renderGemmaCheckIn(event) {
  const form = $("gemmaForm");
  const notice = $("gemmaDisabledNotice");
  const textarea = $("gemmaTranscript");
  const submitBtn = $("gemmaSubmitBtn");
  const active = Boolean(event && event.status === "check_in_required");

  form.hidden = !active;
  notice.hidden = active;
  if (!active) {
    notice.textContent = event
      ? "The AI-assisted check-in is only available while a check-in is required."
      : "A concerning reading must be detected first before an AI-assisted check-in is available.";
  }

  textarea.disabled = !active || gemmaSubmitting;
  submitBtn.disabled = !active || gemmaSubmitting || !textarea.value.trim();

  // Clear a previous result/status once the event has moved on (a new
  // reading, a reset, or a resolved event) so a stale AI response is never
  // shown for a different event.
  if (event?.id !== gemmaResultEventId) {
    gemmaResultEventId = null;
    $("gemmaResult").hidden = true;
    setGemmaStatus("");
  }
}

function setGemmaStatus(text, kind = "") {
  const el = $("gemmaStatus");
  if (!el) return;
  el.textContent = text;
  el.className = `gemma-status${kind ? ` status-${kind}` : ""}`;
}

function renderGemmaResult(data) {
  // Defensively support both possible response shapes for the analysis and
  // handoff payloads, per the backend contract.
  const analysis = data.analysis || data.interpretation || null;
  const validatedTool = data.validated_tool || null;
  const handoffRaw = data.handoff || data.event?.caregiver_handoff || null;
  const interpretationSource = data.source?.interpretation || null;
  const isGemma = interpretationSource === "gemma";

  $("gemmaResult").hidden = false;

  $("gemmaSource").textContent = interpretationSource
    ? (isGemma ? "Gemma 4" : "Deterministic fallback (Gemma was unavailable)")
    : "Unknown";
  $("gemmaSummary").textContent = analysis?.summary || "Not stated";
  $("gemmaAction").textContent = analysis?.action ? fmt(analysis.action) : "Not stated";
  $("gemmaContact").textContent = analysis?.requested_contact || "Not stated";
  $("gemmaCondition").textContent = analysis?.reported_condition || "Not stated";
  $("gemmaReportedAction").textContent = analysis?.reported_action || "Not stated";
  $("gemmaSupplyLocation").textContent = analysis?.supply_location || "Not stated";
  $("gemmaLanguage").textContent = analysis?.detected_language || "Not stated";
  $("gemmaValidatedTool").textContent = validatedTool ? fmt(validatedTool.name) : "No action taken (unclear response)";

  const handoffText = typeof handoffRaw === "string" ? handoffRaw : handoffRaw?.handoff;
  const handoffHeadline = (handoffRaw && typeof handoffRaw === "object") ? handoffRaw.headline : null;
  const handoffBlock = $("gemmaHandoffBlock");
  if (handoffText) {
    handoffBlock.hidden = false;
    $("gemmaHandoffHeadline").textContent = handoffHeadline || "Caregiver handoff";
    $("gemmaHandoffText").textContent = handoffText;
  } else {
    handoffBlock.hidden = true;
  }

  if (!interpretationSource) {
    setGemmaStatus("Check-in processed.", "success");
  } else if (isGemma) {
    setGemmaStatus("Check-in interpreted successfully with Gemma 4.", "success");
  } else {
    setGemmaStatus("Gemma was unavailable. Emergency fallback mode was used.", "fallback");
  }
}

async function submitGemmaCheckIn(domEvent) {
  domEvent.preventDefault();
  if (gemmaSubmitting) return;

  if (!activeEventId) {
    toast("No active event to check in on.", "error");
    return;
  }
  const transcript = $("gemmaTranscript").value.trim();
  if (!transcript) {
    toast("Describe how you're feeling before analyzing.", "error");
    return;
  }

  const submittedForEventId = activeEventId;
  gemmaSubmitting = true;
  $("gemmaSubmitBtn").disabled = true;
  $("gemmaTranscript").disabled = true;
  $("gemmaResult").hidden = true;
  setGemmaStatus("Gemma is interpreting your check-in…", "loading");

  try {
    const data = await api(`/api/events/${submittedForEventId}/voice-check-in`, {
      method: "POST",
      body: JSON.stringify({ transcript, language: "en-US" }),
    });
    gemmaResultEventId = submittedForEventId;
    renderGemmaResult(data);
    await refresh();
  } catch (error) {
    const message = error.status === 409
      ? "The check-in expired while the response was being processed."
      : error.message || "Request failed.";
    setGemmaStatus(message, "error");
    toast(message, "error");
    await refresh();
  } finally {
    gemmaSubmitting = false;
    // Re-enable the input only if the event is still check_in_required -
    // renderGemmaCheckIn (already called by refresh() -> renderApp) is the
    // single source of truth for that, so just re-run it defensively in
    // case refresh() itself failed to reach the server.
    renderGemmaCheckIn(state?.active_event || null);
  }
}

// --- Emergency overlay -------------------------------------------------------

function renderOverlay(event) {
  const overlay = $("emergencyOverlay");
  if (!overlay) return;
  const show = Boolean(event && event.status === "check_in_required");
  overlay.hidden = !show;
  document.body.style.overflow = show ? "hidden" : "";
  if (!show) return;
  $("overlayReading").textContent = `${event.latest_reading.value_mg_dl} mg/dL`;
  $("overlayReason").textContent = event.reason;
}

// --- Countdown (backend deadline is the source of truth) --------------------
//
// The backend's own asyncio timer (see app/main.py::schedule_check_in_timeout)
// is the only thing that actually escalates the event on timeout - this is
// presentation only. It never runs an independent 30-second clock: it reads
// `check_in_deadline` from the active event and recomputes the remaining
// time from the current browser clock every tick, so it can never drift
// from the backend and always reflects the real deadline after every poll.

function clearCountdownDisplay() {
  const countdownText = $("countdownText");
  if (countdownText) { countdownText.hidden = true; countdownText.textContent = ""; }
  const overlayValue = $("countdownValue");
  if (overlayValue) overlayValue.textContent = "--";
  const overlayText = $("overlayCountdownText");
  if (overlayText) overlayText.textContent = "Caregiver escalation begins if there is no response.";
}

function renderCountdown(remainingSeconds, event) {
  const contactName = event?.requested_contact || DEFAULT_CAREGIVER_NAME;
  const message = `Contacting ${contactName} in ${remainingSeconds} second${remainingSeconds === 1 ? "" : "s"}`;

  const countdownText = $("countdownText");
  if (countdownText) { countdownText.hidden = false; countdownText.textContent = message; }
  const overlayValue = $("countdownValue");
  if (overlayValue) overlayValue.textContent = String(remainingSeconds);
  const overlayText = $("overlayCountdownText");
  if (overlayText) overlayText.textContent = message;

  if ([10, 5, 4, 3, 2, 1].includes(remainingSeconds) && lastCountdownCue !== remainingSeconds) {
    lastCountdownCue = remainingSeconds;
    speak(String(remainingSeconds), { interrupt: false, rate: 1 });
  }
}

function stopCountdown() {
  if (countdownInterval) {
    clearInterval(countdownInterval);
    countdownInterval = null;
  }
  lastCountdownCue = null;
}

function startCountdown(event) {
  stopCountdown();

  if (
    !event ||
    event.status !== "check_in_required" ||
    !event.check_in_deadline
  ) {
    clearCountdownDisplay();
    return;
  }

  function updateCountdown() {
    const deadline = new Date(event.check_in_deadline).getTime();
    const now = Date.now();
    const remainingSeconds = Math.max(
      0,
      Math.ceil((deadline - now) / 1000)
    );

    renderCountdown(remainingSeconds, event);

    if (remainingSeconds <= 0) {
      stopCountdown();
      // The frontend never escalates on its own - it just asks the backend
      // what actually happened once the deadline is reached.
      setTimeout(refresh, 500);
    }
  }

  updateCountdown();
  countdownInterval = setInterval(updateCountdown, 1000);
}

function drawChart(thresholds) {
  const canvas = $("glucoseChart"); if (!canvas) return;
  const empty = $("chartEmpty");
  empty.style.display = sessionReadings.length ? "none" : "grid";
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1,rect.width*dpr); canvas.height = Math.max(1,rect.height*dpr);
  const ctx = canvas.getContext("2d"); ctx.scale(dpr,dpr);
  const w=rect.width,h=rect.height,p={l:38,r:12,t:14,b:28};
  ctx.clearRect(0,0,w,h);
  const min=20,max=Math.max(320,...sessionReadings.map(r=>r.value_mg_dl+25));
  const y=(v)=>p.t+(max-v)/(max-min)*(h-p.t-p.b);
  const x=(i)=>p.l+(sessionReadings.length===1?0:(i/(sessionReadings.length-1))*(w-p.l-p.r));
  ctx.font="11px system-ui"; ctx.fillStyle="#7890aa"; ctx.strokeStyle="#263a54"; ctx.lineWidth=1;
  [50,100,150,200,250,300].filter(v=>v<=max).forEach(v=>{ctx.beginPath();ctx.moveTo(p.l,y(v));ctx.lineTo(w-p.r,y(v));ctx.stroke();ctx.fillText(String(v),4,y(v)+4)});
  [thresholds.low,thresholds.high].forEach((v,i)=>{ctx.setLineDash([5,5]);ctx.strokeStyle=i?"#ffc96b":"#ff8a70";ctx.beginPath();ctx.moveTo(p.l,y(v));ctx.lineTo(w-p.r,y(v));ctx.stroke();ctx.setLineDash([])});
  if (!sessionReadings.length) return;
  ctx.strokeStyle="#6da7ff";ctx.lineWidth=3;ctx.lineJoin="round";ctx.lineCap="round";ctx.beginPath();sessionReadings.forEach((r,i)=>i?ctx.lineTo(x(i),y(r.value_mg_dl)):ctx.moveTo(x(i),y(r.value_mg_dl)));ctx.stroke();
  sessionReadings.forEach((r,i)=>{const level=readingClass(r,thresholds);ctx.fillStyle=level==="urgent"?"#ff6b7a":level==="warning"?"#ffc96b":"#4dd6a4";ctx.beginPath();ctx.arc(x(i),y(r.value_mg_dl),4,0,Math.PI*2);ctx.fill()});
}

// --- Caregiver console -------------------------------------------------------

function showCaregiverError(message) {
  const el = $("caregiverErrorText");
  if (!el) return;
  el.hidden = !message;
  el.textContent = message || "";
}

// Priority for the headline caregiver-facing text: the Gemma-generated
// handoff object's `.handoff` text, then a plain-string `caregiver_handoff`
// (defensive - some backend shapes return it pre-flattened), then the raw
// patient response summary, then a safe default.
function resolveHandoffText(event) {
  const handoff = event?.caregiver_handoff;
  if (handoff && typeof handoff === "object" && handoff.handoff) return handoff.handoff;
  if (typeof handoff === "string" && handoff) return handoff;
  if (event?.patient_response_summary) return event.patient_response_summary;
  return "No patient response was provided.";
}

function renderCaregiverHandoff(event) {
  const handoff = event?.caregiver_handoff;
  const handoffObject = handoff && typeof handoff === "object" ? handoff : null;
  const handoffText = resolveHandoffText(event);
  const hasRealHandoff = handoffText !== "No patient response was provided.";

  $("caregiverHandoffHeadline").textContent = hasRealHandoff
    ? (handoffObject?.headline || "Patient check-in")
    : handoffText;
  $("caregiverHandoffText").textContent = hasRealHandoff ? handoffText : "";

  const defaultableStatuses = ["contacting", "acknowledged", "resolved"];
  const contact = event?.requested_contact
    || handoffObject?.requested_contact
    || (event && defaultableStatuses.includes(event.status) ? DEFAULT_CAREGIVER_NAME : null);

  $("caregiverRequestedContact").textContent = contact || "Not stated";
  $("caregiverReportedCondition").textContent = event?.reported_condition || handoffObject?.reported_condition || "Not stated";
  $("caregiverReportedAction").textContent = event?.reported_action || handoffObject?.reported_action || "Not stated";
  $("caregiverSupplyLocation").textContent = event?.supply_location || handoffObject?.supply_location || "Not stated";
}

function updateCaregiver(s) {
  state = s;
  const event = s.active_event; activeEventId = event?.id || null;
  const level = readingClass(event?.latest_reading, s.thresholds);
  $("caregiverHero").className = `card caregiver-hero state-${level}`;
  $("caregiverTitle").textContent = event ? "Patient alert active" : "No active emergency";
  $("caregiverReason").textContent = event?.reason || "Waiting for an event from the patient dashboard.";
  $("caregiverReading").textContent = event?.latest_reading.value_mg_dl ?? "--";
  $("caregiverTrend").textContent = event ? fmt(event.latest_reading.trend) : "No trend";
  $("caregiverStatus").textContent = event ? fmt(event.status) : "Idle";
  $("acknowledgeBtn").disabled = !event || event.status !== "contacting";
  $("caregiverResolveBtn").disabled = !event;
  $("acknowledgedText").textContent = event?.acknowledged_by
    ? `${event.acknowledged_by} is responding.`
    : "No caregiver acknowledgement yet.";
  $("openedAt").textContent = formatTime(event?.opened_at);
  $("updatedAt").textContent = formatTime(event?.updated_at);
  $("patientResponse").textContent = event?.patient_response_summary || fmt(event?.patient_response || "No response");
  $("caregiverEventId").textContent = event?.id || "--";
  renderCaregiverHandoff(event);
  renderHistory(s.history);
  announceCaregiverEvent(event);
}

async function refresh() {
  try {
    const s = await api("/api/state");
    setConnection(true);
    if (page === "caregiver") updateCaregiver(s); else renderApp(s);
  } catch (error) {
    setConnection(false);
    console.error(error);
  }
}

async function submitReading(value, trend) {
  try {
    const result=await api("/api/readings",{method:"POST",body:JSON.stringify({value_mg_dl:Number(value),trend,source:"simulator"})});
    toast(result.decision==="none"?"Reading is within the configured demo range.":`Automatic result: ${fmt(result.decision)}`);
    await refresh();
  } catch(error){toast(error.message,"error")}
}

async function patientResponse(response) {
  if (!activeEventId) return;

  [$("treatingBtn"),$("helpBtn"),$("falseAlarmBtn"),$("overlayTreatingBtn"),$("overlayHelpBtn"),$("overlayFalseBtn")]
    .forEach((button) => { if (button) button.disabled = true; });

  try {
    await api(`/api/events/${activeEventId}/patient-response`, {
      method: "POST",
      body: JSON.stringify({ response }),
    });
    await refresh();
    toast(`Patient response recorded: ${fmt(response)}.`);

    const messages = {
      treating: "Thank you. Your treatment response has been recorded.",
      need_help: "Help request recorded. Caregiver contact has started.",
      false_alarm: "False alarm recorded. The event is resolved.",
    };
    speak(messages[response] || "Response recorded.");
  } catch (error) {
    await refresh();
    toast(error.message, "error");
  }
}
async function eventAction(action,payload){if(!activeEventId)throw new Error("No active event.");const opts={method:"POST"};if(payload)opts.body=JSON.stringify(payload);const data=await api(`/api/events/${activeEventId}/${action}`,opts);await refresh();return data}

function bindPatient() {
  $("audioToggleBtn")?.addEventListener("click", toggleAudio);
  $("readingForm")?.addEventListener("submit",(e)=>{e.preventDefault();const value=Number($("readingInput")?.value);if(!Number.isInteger(value)||value<20||value>600){toast("Enter a whole number between 20 and 600.","error");return}submitReading(value,$("trendInput")?.value || "unknown")});
  document.querySelectorAll(".preset").forEach(btn=>btn.addEventListener("click",()=>submitReading(btn.dataset.reading,btn.dataset.trend)));
  [["treatingBtn","treating"],["overlayTreatingBtn","treating"],["helpBtn","need_help"],["overlayHelpBtn","need_help"],["falseAlarmBtn","false_alarm"],["overlayFalseBtn","false_alarm"]].forEach(([id,response])=>$(id)?.addEventListener("click",()=>patientResponse(response)));
  $("timeoutBtn")?.addEventListener("click",()=>eventAction("timeout").catch(e=>toast(e.message,"error")));
  $("resolveBtn")?.addEventListener("click",()=>eventAction("resolve").then(()=>toast("Event resolved.")).catch(e=>toast(e.message,"error")));
  $("resetBtn")?.addEventListener("click",async()=>{try{await api("/api/reset",{method:"POST"});sessionReadings=[];sessionStorage.removeItem("glucorelayReadings");gemmaResultEventId=null;stopCountdown();await refresh();toast("Demo reset.")}catch(e){toast(e.message,"error")}});
  $("clearChartBtn")?.addEventListener("click",()=>{sessionReadings=[];sessionStorage.removeItem("glucorelayReadings");drawChart(state?.thresholds||{low:70,high:250});});
  $("gemmaForm")?.addEventListener("submit", submitGemmaCheckIn);
  $("gemmaTranscript")?.addEventListener("input", () => renderGemmaCheckIn(state?.active_event || null));
  window.addEventListener("resize",()=>state&&drawChart(state.thresholds));
}

function bindCaregiver(){
  $("audioToggleBtn")?.addEventListener("click", toggleAudio);
  $("acknowledgeBtn")?.addEventListener("click",async()=>{
    const name=$("caregiverName")?.value.trim() || "";
    if(!name){toast("Enter the caregiver name.","error");return}
    showCaregiverError(null);
    try{
      await eventAction("acknowledge",{caregiver_name:name});
      toast("Alert acknowledged.");
      speak("Caregiver acknowledgement recorded.");
    }catch(e){
      showCaregiverError(e.message);
      toast(e.message,"error");
    }
  });
  $("caregiverResolveBtn")?.addEventListener("click",()=>{
    showCaregiverError(null);
    eventAction("resolve").then(()=>toast("Event resolved.")).catch(e=>{showCaregiverError(e.message);toast(e.message,"error")});
  });
}

document.addEventListener("DOMContentLoaded",()=>{
  updateAudioButton();
  document.addEventListener("pointerdown", unlockAudio, { once: true });
  document.addEventListener("keydown", unlockAudio, { once: true });
  page==="caregiver"?bindCaregiver():bindPatient();
  refresh();
  setInterval(refresh,POLL_MS);
});
