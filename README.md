# GlucoRelay

An AI-assisted emergency escalation **prototype** for people with Type 1 diabetes,
built around Gemma as the natural-language core of its emergency communication
workflow.

> **⚠️ Hackathon prototype. Not a medical device.** GlucoRelay does not
> diagnose, treat, or manage diabetes. It does not contact emergency
> services. It simulates glucose readings and caregiver notifications for
> demonstration purposes only.

## Project purpose

When a simulated CGM (continuous glucose monitor) reading becomes
concerning, GlucoRelay:

1. Creates an emergency event (deterministic Python decision).
2. Asks the patient to check in.
3. Lets the patient respond with buttons **or** by speaking.
4. Converts speech to text in the browser and sends the transcript to the
   backend.
5. Uses Gemma to **understand** the transcript, **propose** one constrained
   application tool, and (on escalation) **draft** a grounded caregiver
   handoff.
6. Validates all of that in deterministic Python and drives a state machine.
7. Contacts a trusted caregiver if the patient asks for help or doesn't
   respond in time, using the Gemma-drafted (or deterministic fallback)
   handoff.
8. Lets a caregiver acknowledge that they're responding, from a private
   link, and see the handoff.
9. Records every step on both an incident timeline and a presentation-safe
   Gemma trace.
10. Lets the event be resolved.

## Why Gemma is core to this project

GlucoRelay's differentiator isn't the threshold engine (that's a handful of
`if` statements) - it's turning an unstructured, possibly multilingual,
possibly emotional spoken check-in into something a caregiver can act on in
seconds, without ever letting a language model make a safety-critical call.
Gemma has exactly three responsibilities:

1. **Understand** the patient's natural-language check-in -
   `app/gemma_service.py::analyze_patient_checkin` returns a strict
   `PatientCheckInAnalysis` (action, responsive, summary, requested_contact,
   reported_condition, reported_action, supply_location, follow_up_minutes,
   detected_language, english_summary, confidence).
2. **Propose** exactly one constrained application tool - Gemma's `action`
   field is itself constrained to six literal values by the JSON schema it
   must follow, and `app/tools.py::propose_tool` deterministically maps that
   to a `ProposedToolCall`. Gemma is never allowed to execute it.
3. **Draft** a grounded caregiver handoff after an escalation is already
   validated - `app/gemma_service.py::generate_caregiver_handoff` returns a
   strict `CaregiverHandoff`, built only from verified backend facts.

Everything else - whether a glucose value is dangerous, whether a check-in
or escalation should start, whether a proposed tool is actually allowed to
run, and every state transition - is deterministic Python in `app/engine.py`,
`app/tools.py`, and `app/transitions.py`.

## Safety architecture

**Gemma never makes a medical or safety decision, and never executes
anything.** The code is structured, and Gemma is explicitly instructed, so
that it never:

- Determines whether a glucose value is medically dangerous
- Recommends insulin, medication, or dosages
- Diagnoses the patient
- Changes glucose thresholds
- Claims the patient is unconscious
- Invents symptoms, contacts, or supply locations not explicitly stated
- Automatically contacts emergency services
- Directly executes a backend function or bypasses `app/transitions.py`

Concretely, the safety boundary is enforced at five independent layers:

1. **Prompting** - both the interpretation and handoff prompts
   (`app/gemma_service.py`) explicitly list these boundaries and require
   facts to come only from the transcript / supplied data.
2. **Strict Pydantic schemas** - `PatientCheckInAnalysis`, `ProposedToolCall`,
   and `CaregiverHandoff` are validated on every response; anything that
   doesn't parse is discarded.
3. **A post-hoc safety re-scan** - even schema-valid output is scanned for
   banned language (e.g. "unconscious", "insulin", "dosage", "medication")
   before it's trusted; a hit discards the Gemma output entirely and falls
   back to the deterministic parser/handoff (`_contains_unsafe_content` in
   `app/gemma_service.py`).
4. **`app/tools.py::validate_tool_call`** - the last line of defense before
   anything executes. It independently re-checks that the tool name is
   supported, the event's current status allows it, every argument matches
   a fact Gemma actually extracted (never invented; e.g. `schedule_patient_recheck`'s
   `minutes` argument is verified against `PatientCheckInAnalysis.follow_up_minutes`),
   numeric arguments (e.g. `minutes`) are within a sane demo range, no
   unexpected argument fields exist, no treatment/medication language
   appears anywhere in the arguments, and the implied state transition is
   legal per `app/transitions.py`. Only a `ValidatedToolCall` may execute.

If Gemma is unavailable, disabled, misconfigured, or returns output that
fails schema or safety validation, a small deterministic keyword-based
fallback parser (`app/gemma_service.py::_fallback_interpret`) is used
instead, the interpretation source is marked `"fallback"`, and the event's
status never changes based on unusable output - worst case, the check-in
simply stays active and the failure is recorded on both the timeline and
the Gemma trace.
5. **Deterministic semantic validation** - even schema-valid, safety-scanned
   output can be *semantically* wrong (see below); a final deterministic
   pass either confirms or corrects it before anything transitions.

### Deterministic semantic validation (catches model over-inference)

A schema-valid, safety-scanned Gemma response can still be semantically
wrong - e.g. a real run of this project once saw the transcript "My
backpack is red." classified as `action: "okay"` simply because nothing in
it sounded alarming. Passing Pydantic validation and the banned-phrase
re-scan does not mean the classification is actually supported by what the
patient said, so `app/gemma_service.py::_apply_deterministic_corrections`
runs on **every** interpretation (Gemma or fallback) before it is trusted:

- **Action-evidence check.** Each action (`okay`, `treating`, `false_alarm`,
  `need_help`, `schedule_recheck`) has a fixed list of explicit English/
  Spanish phrases (e.g. "i'm okay", "estoy bien", "drank juice", "false
  alarm", "need help", "llama", "check on me again") that must actually
  appear in the original transcript. If the claimed action has no such
  phrase, it is deterministically downgraded to `"unknown"` - the event is
  **not** transitioned. This is recorded as a `semantic_correction_applied`
  step on both the timeline and the Gemma trace, explicitly **not** as a
  Gemma failure (`interpretation_source` stays `"gemma"`) - it's a backend
  safety correction, the same way a spell-checker underlining a word isn't
  "the writer failing."
- **Recheck-request precedence.** "Everything is okay, but check on me
  again in ten minutes." must produce `schedule_recheck`, not `okay` - an
  explicit, evidenced follow-up request always takes precedence over a
  general wellness statement when both are present.
- **English summary normalization.** When `detected_language` is English,
  `english_summary` is deterministically set equal to the already-validated
  `summary` - Gemma's prompt asks it not to reword an English summary, but
  the backend enforces this directly rather than trusting that instruction
  alone. Non-English transcripts are unaffected and still receive a real
  translated `english_summary`.

See `tests/test_semantic_validation.py` for the mocked-Gemma regression
tests covering all of the above.

## On "function calling" (what this project actually does, and doesn't, claim)

Before implementing tool selection, this project's `GEMMA_MODEL` access path
(the Gemini API via the `google-genai` Python SDK) was checked against
Google's current documentation. The Gemini API's native
`tools`/`function_declarations` parameter is documented for **Gemini**
reasoning models. Google's documentation for **Gemma**-family models
describes function calling only for a **locally-hosted** model driven
through `transformers.apply_chat_template()`, where the developer manually
parses the model's raw text output to extract a function call - not the
Gemini API's native tool-calling mechanism.

Since GlucoRelay calls its configured Gemma model through the Gemini API,
**this project does not use, and does not claim to use, native function
calling.** Instead:

- `analyze_patient_checkin` constrains Gemma's response with
  `response_schema=PatientCheckInAnalysis` (structured JSON output), which
  itself constrains `action` to one of six literal values.
- `app/tools.py::propose_tool` is pure, deterministic Python that maps that
  already-constrained `action` to a `ProposedToolCall` - it is **not** a
  second call to Gemma, and Gemma never sees or picks from a list of tool
  definitions directly.
- This is the "equivalent constrained tool selection using strict
  structured output" required when native function calling isn't available
  for the configured endpoint, and it is documented as such here and in
  `app/tools.py` and `app/gemma_service.py`'s module docstrings.

Similarly, GlucoRelay does not do any native audio processing - the browser's
Web Speech API produces a text transcript client-side, and only that text
is ever sent to the backend or to Gemma.

## Architecture

```
app/
  main.py            FastAPI routes, state-transition wiring, timers, the voice-check-in pipeline
  engine.py           Deterministic glucose threshold logic (unchanged - no AI)
  models.py           Pydantic models: Reading, EmergencyEvent, PatientCheckInAnalysis,
                       ProposedToolCall, CaregiverHandoff, GemmaTrace, ...
  transitions.py       Single source of truth for allowed event-status transitions
  timeline.py          One reusable helper for appending incident-timeline entries
  gemma_trace.py       Reusable helpers for building a safe, presentation-ready Gemma trace
  gemma_service.py    Gemma calls (interpretation + handoff), safety re-scan, fallback parser/handoff
  tools.py            Deterministic tool proposal, backend validation, and execution
  notifications.py    Simulated / Twilio SMS caregiver notifications (handoff-aware)
  store.py            In-memory store (single active event + short history)
  static/
    index.html         Patient-facing demo UI (readings, buttons, voice check-in)
    caregiver.html      Caregiver-facing page, reachable only via a public token
scripts/
  smoke_test_gemma.py  Manual, real-network smoke test against the configured Gemma model
tests/                 pytest suite (Gemma and Twilio are mocked - no network/API keys needed)
```

Design choices:

- **Single shared transition table** (`app/transitions.py`) is used by
  every endpoint/tool that changes an event's status.
- **One timeline helper** (`app/timeline.py::add_timeline_entry`) is used
  everywhere an incident-timeline entry is recorded, and one trace helper
  (`app/gemma_trace.py::add_trace_step`) everywhere a Gemma-pipeline step
  is recorded.
- **Tool proposal/validation/execution is a separate module** (`app/tools.py`)
  from both the Gemma call and the HTTP layer, so it can be unit-tested
  independently and so main.py never has to trust Gemma's output directly.
- **Services are structured so persistence can be swapped later.**
  `app/store.py` exposes a small, explicit interface; nothing else in the
  codebase reaches into its internals.

## How voice check-ins work

The voice feature is **only** active while an event's status is
`check_in_required` - it is not a continuously-listening wake-word feature.
In the demo UI, pressing "Speak check-in" starts the browser's
`SpeechRecognition` API for a single utterance, fills in a transcript box,
and the patient (or you, for a demo) sends it to:

```
POST /api/events/{event_id}/voice-check-in
{ "transcript": "...", "language": "en-US" }
```

The backend pipeline (`app/main.py::voice_check_in`):

1. Confirms the event exists and is `check_in_required` (otherwise `409`).
2. Rejects empty or excessively long transcripts (`400`).
3. Records a `voice_transcript_received` timeline entry and a Gemma-trace
   `input_received` step.
4. Calls `analyze_patient_checkin` - Gemma, or the deterministic fallback if
   Gemma is unavailable/disabled/fails validation/fails the safety re-scan.
5. Derives a `ProposedToolCall` from the analysis (`app/tools.py::propose_tool`).
6. Validates that proposal (`app/tools.py::validate_tool_call`); an invalid
   proposal is rejected, recorded (`tool_rejected`), and **does not change
   the event's state**.
7. Executes the validated tool (`app/tools.py::execute_validated_tool`),
   which applies the single legal state transition for that tool, if any.
   For `schedule_patient_recheck`, this also (re)schedules the same
   in-process timer mechanism used for check-in deadlines
   (`app/main.py::schedule_patient_recheck_timeout`) for the requested
   number of minutes - if the event is still `monitoring` once that window
   elapses, it auto-escalates to `contacting` as a precaution. Scheduling a
   new timer always cancels any previous one for the event first, so at
   most one timer is ever pending.
8. If the tool was `request_caregiver_help` (i.e. an escalation to
   `contacting`), generates a grounded caregiver handoff
   (`generate_caregiver_handoff`) and sends the caregiver alert using it.
9. Returns the full pipeline result (see below), the updated event, its
   timeline, and its Gemma trace.

### Tool -> transition mapping

```
action            -> tool                       -> status
okay              -> record_patient_okay         -> monitoring
treating          -> record_patient_treating     -> monitoring
need_help         -> request_caregiver_help      -> contacting (handoff generated, caregiver notified)
schedule_recheck  -> schedule_patient_recheck    -> monitoring (minutes recorded, recheck timer scheduled)
false_alarm       -> resolve_false_alarm         -> resolved
unknown           -> report_unclear_response     -> no change (check-in stays active)
```

Note: the analysis field is `follow_up_minutes` (what Gemma/the fallback
extracted), but the *tool argument* is deliberately named `minutes` (e.g.
`{"name": "schedule_patient_recheck", "arguments": {"minutes": 10}}`) -
`app/tools.py::TOOL_ARGUMENT_SOURCE_FIELD` maps between the two names so
validation still checks it against the verified analysis field.

### Response shape

```json
{
  "transcript": "...",
  "analysis": {
    "action": "need_help",
    "responsive": true,
    "summary": "...",
    "requested_contact": "Helper",
    "reported_condition": "confused",
    "reported_action": null,
    "supply_location": "red backpack",
    "follow_up_minutes": null,
    "detected_language": "en",
    "english_summary": "..."
  },
  "proposed_tool": { "name": "request_caregiver_help", "arguments": {} },
  "validated_tool": { "name": "request_caregiver_help", "arguments": {} },
  "handoff": { "headline": "...", "handoff": "...", "unknown_information": [] },
  "source": { "interpretation": "gemma", "handoff": "gemma" },
  "event": { "...": "the full updated event" },
  "timeline": [ "..." ],
  "gemma_trace": { "...": "see below" },
  "notification": { "delivery": "simulated", "message": "..." }
}
```

A rejected tool proposal returns HTTP `200` (the request itself was valid)
with `"validated_tool": null` and the event unchanged - it is not treated
as a client error. Calling the endpoint again once the event has left
`check_in_required` returns `409`.

## Grounded caregiver handoff generation

After (and only after) `request_caregiver_help` is validated and executed,
`app/gemma_service.py::generate_caregiver_handoff` builds a `CaregiverHandoff`
from **only verified backend facts**: the current simulated reading and
trend, the event reason and status, the original transcript, the validated
analysis fields, and whether location was actually shared. The prompt
requires Gemma to label the reading as simulated, never give treatment
advice, never claim unconsciousness, explicitly list what's unknown
(`unknown_information`), stay concise, and always answer in English (even
for a Spanish transcript) while preserving the detected language in
`detected_language`.

If handoff generation fails (Gemma unavailable, request error, invalid
output, or a safety re-scan hit), GlucoRelay does **not** undo the
already-valid escalation - it uses `_fallback_handoff`, a deterministic
handoff assembled from the same verified fields, marks
`caregiver_handoff_source: "fallback"`, records the failure on the timeline
and Gemma trace (`handoff_failed`), and still sends a real (never empty)
caregiver notification.

## Gemma trace

Every voice check-in accumulates a `GemmaTrace` on the event
(`event.gemma_trace`), with ordered `GemmaTraceStep`s such as:

```
input_received -> interpretation_started -> interpretation_completed (or fallback_used)
  -> tool_proposed -> tool_validated (or tool_rejected) -> tool_executed
  -> handoff_started -> handoff_completed (or handoff_failed)
```

It also records `interpretation_source`, `handoff_source`, the configured
`model_name`, the `original_language`, and the proposed/validated tool. It
is intentionally presentation-safe: it **never** includes API keys, raw
system prompts, hidden model reasoning, full SDK responses, or stack
traces - only observable application events. This is covered by
`tests/test_gemma_trace.py`.

## Multilingual support

`analyze_patient_checkin` asks Gemma to detect the transcript's original
language, produce a same-language `summary`, and separately produce an
`english_summary`. For example:

- English: *"I feel confused. Please contact Helper."* -> `detected_language: "en"`.
- Spanish: *"Me siento confundida. Por favor llama a Helper."* ->
  `detected_language: "es"`, `action: "need_help"`,
  `requested_contact: "Helper"`, `english_summary` populated in English.

The caregiver handoff is always generated in English regardless of the
original language (see prompt rules above), while the original transcript
remains available internally (`event.patient_transcript`) and nothing is
invented for either language. No separate translation service is used -
Gemma performs both understanding and English generation in one step.

The deterministic fallback parser is **English-only by design**. When it
detects likely non-English input (a small set of Spanish markers) that
doesn't also match one of its explicit English phrases, it does not guess -
it returns `action: "unknown"` with `detected_language: "unknown-non-english"`
and a summary explaining that Gemma is required for multilingual
understanding.

## Why threshold decisions are deterministic

Diabetic emergencies are safety-critical. An LLM can misread a transcript,
hallucinate, or be prompt-injected; it must never be the thing that decides
whether a reading is dangerous or whether to escalate. `app/engine.py`'s
`evaluate_reading()` is pure Python, has no dependency on any AI service,
and is the only place glucose thresholds are evaluated. Gemma is used
exclusively downstream of that decision, to interpret the patient's
spoken response to a check-in that Python already decided to require.

## State machine

```
check_in_required:
  okay              -> monitoring
  treating          -> monitoring
  need_help         -> contacting   (caregiver notified with handoff)
  schedule_recheck  -> monitoring   (minutes recorded, recheck timer scheduled)
  false_alarm       -> resolved
  unknown           -> check_in_required (no change)
  deadline expires  -> contacting   (caregiver notified)

monitoring:
  new urgent reading -> contacting (caregiver notified)
  timeout / missed follow-up -> contacting (caregiver notified)
  requested recheck window elapses -> contacting (caregiver notified)
  resolve -> resolved

contacting:
  caregiver acknowledges -> acknowledged
  resolve -> resolved
  (no duplicate alerts for the same escalation)

acknowledged:
  resolve -> resolved

resolved:
  (terminal - no further transitions)
```

Invalid transitions return **HTTP 409** with both the current status and
the allowed next statuses:

```json
{
  "detail": {
    "message": "Cannot perform 'acknowledge' while event is 'check_in_required'.",
    "current_status": "check_in_required",
    "allowed_transitions": ["contacting", "monitoring", "resolved"]
  }
}
```

## Environment setup

1. Copy `.env.example` to `.env` and fill in values as needed. `.env` is
   git-ignored - never commit it.
2. Create/activate a virtual environment and install dependencies:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

Minimum viable `.env` for a fully-simulated demo (no external services):

```
ENABLE_GEMMA=false
ENABLE_SMS=false
PUBLIC_BASE_URL=http://127.0.0.1:8000
CHECK_IN_TIMEOUT_SECONDS=30
DEFAULT_CAREGIVER_NAME=Helper
```

`DEFAULT_CAREGIVER_NAME` (default `Helper`) is the generic caregiver
display name used in the caregiver handoff, caregiver alert text, and the
caregiver page whenever the patient did **not** explicitly name a contact.
It never overrides a contact name the patient actually stated (e.g. "call
Luis" always displays "Luis").

To use real Gemma interpretation and handoff generation, also set
`GEMINI_API_KEY` and `GEMMA_MODEL` (a Gemma model name available through the
Gemini API), and leave `ENABLE_GEMMA=true` / `ENABLE_GEMMA_HANDOFF=true`
(the defaults). Without a configured key/model, or with either flag set to
`false`, the app automatically uses the deterministic fallback parser
and/or fallback handoff builder - the demo still works end-to-end, just
without natural-language understanding.

To use real SMS, set `ENABLE_SMS=true` plus the `TWILIO_*` and
`EMERGENCY_CONTACT_NUMBER` variables.

## Running the server

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Then open <http://127.0.0.1:8000> for the patient-facing demo UI, or
<http://127.0.0.1:8000/docs> for interactive Swagger docs.

## Running tests

```powershell
.venv\Scripts\python.exe -m pytest
```

All Gemma and Twilio calls are mocked/stubbed in the automated suite - no
API keys or network access are required. The suite covers: glucose-triggered
event creation, every voice-check-in action (including `schedule_recheck`),
tool proposal/validation/rejection (including invented-fact and
out-of-range rejection), Gemma interpretation and handoff failure fallback
paths, **deterministic semantic validation** (unsupported-action downgrade,
recheck-request precedence over "okay", English-summary normalization -
`tests/test_semantic_validation.py`), Gemma-trace stage sequencing and
secret-safety, multilingual (Spanish -> English handoff, Spanish "Helper"
contact requests) behavior, the default `Helper` caregiver name and
explicit-name overrides, timeout escalation (including that it only fires
once) and the recheck timer (including that only one timer is ever
pending), caregiver acknowledgement and the caregiver-safe view (including
that it excludes raw AI/internal data), duplicate-submission idempotency,
invalid-transition `409` responses, and location validation.

### Live Gemma smoke test (manual, not part of the automated suite)

`scripts/smoke_test_gemma.py` exercises the **real** configured Gemma model
over the network - it is intentionally excluded from pytest. To run it:

1. Set real `GEMINI_API_KEY` and `GEMMA_MODEL` values in `.env` (or your
   shell environment).
2. Run:

   ```powershell
   .venv\Scripts\python.exe scripts/smoke_test_gemma.py
   ```

It refuses to run if either variable is missing, sends seven representative
transcripts (including an ambiguous retraction and a Spanish "Helper"
contact request), and for each one prints the validated
`PatientCheckInAnalysis` and proposed tool **and asserts the expected
semantic outcome** - it does not merely print schema-valid output. It exits
nonzero if any transcript fails validation, silently falls back to the
deterministic parser, or produces the wrong semantic result, e.g.:

- *"My backpack is red."* must resolve to `action: "unknown"` (regression
  test for the "unsupported okay" bug described above).
- *"Everything is okay, but check on me again in ten minutes."* must
  resolve to `action: "schedule_recheck"` with `follow_up_minutes: 10` -
  not `okay`.
- English transcripts must have `english_summary == summary` exactly (no
  reworded/rewritten summary).
- *"I feel confused and I need Helper to help me."* and its Spanish
  equivalent must both extract `requested_contact: "Helper"`.

It also never prints the API key.

## Demo sequence (via the UI or Swagger)

1. `POST /api/reset` to start clean.
2. `POST /api/readings` with `value_mg_dl: 67` -> creates a `check_in_required`
   event with a 30s deadline (or your `CHECK_IN_TIMEOUT_SECONDS`).
3. `POST /api/events/{id}/voice-check-in` with the target demo transcript:

   ```json
   { "transcript": "I'm awake, but I feel confused. Please contact Helper. My glucagon is in my red backpack." }
   ```

   -> Gemma (or the fallback) extracts only stated facts, proposes
   `request_caregiver_help`, the backend validates it, the event moves to
   `contacting`, and a caregiver handoff is generated.
4. Open the printed caregiver link (`/caregiver/{public_token}`), or call
   `GET /api/caregiver/events/{public_token}`, to see the caregiver-safe
   view, including the handoff headline and text.
5. `POST /api/caregiver/events/{public_token}/acknowledge` with
   `{"caregiver_name": "Helper"}` -> event moves to `acknowledged`.
6. `POST /api/events/{id}/resolve` -> event moves to `resolved` and is
   archived into `history`.

Try also:

- `POST /api/readings` with `value_mg_dl: 48` to see an urgent-low reading
  go straight to `contacting`.
- A Spanish transcript, *"Estoy despierta, pero me siento confundida. Por
  favor llama a Helper."*, to see `detected_language: "es"`,
  `english_summary` populated, `need_help` triggered, `requested_contact:
  "Helper"`, and an English caregiver handoff generated.
- *"Everything is okay, but check on me again in ten minutes."* to see
  `schedule_recheck` (not `okay`) and `minutes: 10` recorded on the tool
  call, plus a recheck timer scheduled, without escalating.
- *"My backpack is red."* to see `action: "unknown"` - an unrelated detail
  must never be interpreted as "the patient is okay."
- A transcript with no explicit contact named (e.g. *"I need help, I feel
  confused."*) to see the caregiver handoff/alert default to `Helper`
  rather than a blank contact.

### Testing the voice check-in flow through Swagger

1. Open `/docs`.
2. Call `POST /api/readings` with `{"value_mg_dl": 67, "trend": "flat"}` and
   copy the returned `event.id`.
3. Call `POST /api/events/{event_id}/voice-check-in` with a body such as
   `{"transcript": "I feel confused, please call Helper"}`.
4. Inspect the response: `analysis` (the structured understanding),
   `proposed_tool` / `validated_tool`, `handoff`, `source`
   (`{"interpretation": "gemma"|"fallback", "handoff": "gemma"|"fallback"|"not_generated"}`),
   `event` (now `contacting`, with `requested_contact`/`reported_condition`
   filled in), `timeline`, `gemma_trace`, and `notification`.
5. Re-run the same call - you'll get a `409` because the check-in is no
   longer active.

## What is simulated

- Glucose readings (`POST /api/readings`) - no real CGM/Dexcom integration.
- Caregiver SMS, unless `ENABLE_SMS=true` and real Twilio credentials are
  configured (even then, only one demo phone number is contacted).
- Patient location - only ever set if the frontend explicitly sends it
  (browser geolocation); never invented.
- Speech-to-text happens in the browser (Web Speech API); the backend only
  ever sees the resulting text transcript, never audio.

## Which features still work without Gemma

With `ENABLE_GEMMA=false` (or no `GEMINI_API_KEY`/`GEMMA_MODEL` configured),
GlucoRelay keeps working end-to-end using the deterministic fallback
parser and fallback handoff builder: readings, threshold evaluation,
check-in timeouts, button-based responses, a small set of explicit English
voice commands (help/contact/false-alarm/treating/okay/recheck keyword
phrases), caregiver notification with a plain deterministic handoff,
caregiver acknowledgement, location sharing, and the full timeline/state
machine.

## Which defining features are lost without Gemma

Without Gemma, GlucoRelay retains basic explicit-command fallback
behavior, but loses natural-language understanding, indirect intent
recognition, multilingual interpretation, constrained tool selection, and
contextual caregiver handoff generation.

Concretely: the fallback parser cannot understand phrasing outside its
fixed keyword list, cannot resolve ambiguous or contradictory statements
(it returns `unknown` rather than reasoning about context), cannot
interpret non-English transcripts, and cannot produce a nuanced,
context-aware caregiver handoff - only a templated one built from whatever
few fields the keyword parser managed to extract.

## Current prototype limitations

- **In-memory store**: a single active event plus a short history live in
  process memory (`app/store.py`). Restarting the server clears all state.
  A production system would need a durable database.
- **In-process timer**: the check-in deadline uses an `asyncio` task per
  event. This is fine for a single-process demo but would need a durable
  job queue (Celery, RQ, a cloud scheduler, etc.) in production so
  deadlines survive restarts and work across multiple workers.
- **No authentication**: the caregiver view is protected only by an
  unguessable token in the URL, and the patient-facing API has no auth at
  all. Not suitable for real patient data.
- **No native function calling**: as documented above, tool selection uses
  structured output, not the Gemini API's native `tools` parameter, because
  the configured Gemma model is not called through a path that supports it.
- Not implemented (intentionally, out of scope for this prototype): real
  Dexcom/CGM integration, insulin or medication recommendations, automatic
  911/emergency-services calls, medical diagnosis, full user
  authentication, native mobile apps, fall detection, continuous
  microphone/wake-word listening, model fine-tuning, hidden
  chain-of-thought display, and PostgreSQL (the project only ever used the
  in-memory store).
