const page = document.body.dataset.page;
const POLL_MS = 2500;
const COUNTDOWN_SECONDS = 30;
let state = null;
let activeEventId = null;
let countdownTimer = null;
let countdownEventId = null;
let timeoutRequested = false;
let audioEnabled = localStorage.getItem("glucorelayAudioEnabled") !== "false";
let audioUnlocked = false;
let lastSpokenEventKey = sessionStorage.getItem("glucorelayLastSpokenEventKey") || "";
let lastCaregiverEventKey = sessionStorage.getItem("glucorelayLastCaregiverEventKey") || "";
let lastCountdownCue = null;
let sessionReadings = JSON.parse(sessionStorage.getItem("glucorelayReadings") || "[]");

const $ = (id) => document.getElementById(id);
const fmt = (value) => String(value || "").replaceAll("_", " ");
const formatTime = (iso) => iso ? new Intl.DateTimeFormat(undefined,{hour:"numeric",minute:"2-digit",second:"2-digit"}).format(new Date(iso)) : "--";

async function api(path, options = {}) {
  const response = await fetch(path, {headers:{"Content-Type":"application/json"}, ...options});
  let data = null;
  try { data = await response.json(); } catch { data = {}; }
  if ($("apiOutput")) $("apiOutput").textContent = JSON.stringify(data, null, 2);
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
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
  button.textContent = audioEnabled ? "Audio on" : "Audio off";
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
  } else {
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
    speak(`Alert acknowledged by ${event.acknowledged_by || "the caregiver"}.`);
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

function updatePatient(s) {
  const reading = s.latest_reading;
  const event = s.active_event;
  activeEventId = event?.id || null;
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
  renderWorkflow(event?.status);

  const patientCanRespond = Boolean(event && ["check_in_required","monitoring","contacting"].includes(event.status));
  $("treatingBtn").disabled = !patientCanRespond;
  $("helpBtn").disabled = !patientCanRespond;
  $("falseAlarmBtn").disabled = !patientCanRespond;
  $("timeoutBtn").disabled = !event || !["check_in_required","monitoring"].includes(event.status);
  $("resolveBtn").disabled = !event;
  renderHistory(s.history);
  drawChart(s.thresholds);
  updateOverlay(event);
  announcePatientEvent(event, s.thresholds);
}

function updateOverlay(event) {
  const overlay = $("emergencyOverlay"); if (!overlay) return;
  const show = Boolean(event && event.status === "check_in_required");
  overlay.hidden = !show;
  document.body.style.overflow = show ? "hidden" : "";
  if (!show) { stopCountdown(); return; }
  $("overlayReading").textContent = `${event.latest_reading.value_mg_dl} mg/dL`;
  $("overlayReason").textContent = event.reason;
  if (countdownEventId !== event.id) startCountdown(event.id);
}

function startCountdown(eventId) {
  stopCountdown();
  countdownEventId = eventId;
  timeoutRequested = false;
  lastCountdownCue = null;
  let remaining = COUNTDOWN_SECONDS;
  speak(`Glucose warning. Caregiver escalation begins in ${COUNTDOWN_SECONDS} seconds.`);
  $("countdownValue").textContent = remaining;
  countdownTimer = setInterval(async () => {
    remaining -= 1;
    if ($("countdownValue")) $("countdownValue").textContent = Math.max(remaining,0);
    if ([10, 5, 4, 3, 2, 1].includes(remaining) && lastCountdownCue !== remaining) {
      lastCountdownCue = remaining;
      speak(String(remaining), { interrupt: false, rate: 1 });
    }
    if (remaining <= 0) {
      stopCountdown();
      if (!timeoutRequested && activeEventId === eventId) {
        timeoutRequested = true;
        try { await eventAction("timeout"); toast("No response detected. Caregiver contact started."); }
        catch (error) { toast(error.message,"error"); }
      }
    }
  },1000);
}
function stopCountdown(){ if(countdownTimer) clearInterval(countdownTimer); countdownTimer=null; countdownEventId=null; lastCountdownCue=null; }

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

function updateCaregiver(s) {
  const event=s.active_event; activeEventId=event?.id||null;
  const level=readingClass(event?.latest_reading,s.thresholds);
  $("caregiverHero").className=`card caregiver-hero state-${level}`;
  $("caregiverTitle").textContent=event?"Patient alert active":"No active emergency";
  $("caregiverReason").textContent=event?.reason||"Waiting for an event from the patient dashboard.";
  $("caregiverReading").textContent=event?.latest_reading.value_mg_dl??"--";
  $("caregiverTrend").textContent=event?fmt(event.latest_reading.trend):"No trend";
  $("caregiverStatus").textContent=event?fmt(event.status):"Idle";
  $("acknowledgeBtn").disabled=!event||event.status!=="contacting";
  $("caregiverResolveBtn").disabled=!event;
  $("acknowledgedText").textContent=event?.acknowledged_by?`Acknowledged by ${event.acknowledged_by}.`:"No caregiver acknowledgement yet.";
  $("openedAt").textContent=formatTime(event?.opened_at); $("updatedAt").textContent=formatTime(event?.updated_at);
  $("patientResponse").textContent=fmt(event?.patient_response||"No response"); $("caregiverEventId").textContent=event?.id||"--";
  renderHistory(s.history);
  announceCaregiverEvent(event);
}

async function refresh() {
  try { state=await api("/api/state"); setConnection(true); page==="caregiver"?updateCaregiver(state):updatePatient(state); }
  catch(error){setConnection(false);console.error(error)}
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

  const overlay = $("emergencyOverlay");
  if (overlay) overlay.hidden = true;
  document.body.style.overflow = "";
  stopCountdown();

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
  $("resetBtn")?.addEventListener("click",async()=>{try{await api("/api/reset",{method:"POST"});sessionReadings=[];sessionStorage.removeItem("glucorelayReadings");stopCountdown();await refresh();toast("Demo reset.")}catch(e){toast(e.message,"error")}});
  $("clearChartBtn")?.addEventListener("click",()=>{sessionReadings=[];sessionStorage.removeItem("glucorelayReadings");drawChart(state?.thresholds||{low:70,high:250});});
  window.addEventListener("resize",()=>state&&drawChart(state.thresholds));
}

function bindCaregiver(){
  $("audioToggleBtn")?.addEventListener("click", toggleAudio);
  $("acknowledgeBtn")?.addEventListener("click",async()=>{const name=$("caregiverName")?.value.trim() || "";if(!name){toast("Enter the caregiver name.","error");return}try{await eventAction("acknowledge",{caregiver_name:name});toast("Alert acknowledged.");speak("Caregiver acknowledgement recorded.")}catch(e){toast(e.message,"error")}});
  $("caregiverResolveBtn")?.addEventListener("click",()=>eventAction("resolve").then(()=>toast("Event resolved.")).catch(e=>toast(e.message,"error")));
}

document.addEventListener("DOMContentLoaded",()=>{
  updateAudioButton();
  document.addEventListener("pointerdown", unlockAudio, { once: true });
  document.addEventListener("keydown", unlockAudio, { once: true });
  page==="caregiver"?bindCaregiver():bindPatient();
  refresh();
  setInterval(refresh,POLL_MS);
});
