from app.gemma_service import analyze_patient_checkin


def test_fallback_used_when_gemma_not_configured():
    analysis, source, reason = analyze_patient_checkin("I'm okay")
    assert source == "fallback"
    assert reason == "gemma_not_configured"


def test_fallback_recognizes_need_help():
    analysis, _, _ = analyze_patient_checkin("I need help please")
    assert analysis.action == "need_help"


def test_fallback_recognizes_named_contact_request():
    analysis, _, _ = analyze_patient_checkin("please call Luis")
    assert analysis.action == "need_help"
    assert analysis.requested_contact == "Luis"


def test_fallback_recognizes_false_alarm():
    analysis, _, _ = analyze_patient_checkin("false alarm, cancel it")
    assert analysis.action == "false_alarm"


def test_fallback_recognizes_treating():
    analysis, _, _ = analyze_patient_checkin("I drank some juice")
    assert analysis.action == "treating"


def test_fallback_recognizes_okay():
    analysis, _, _ = analyze_patient_checkin("I'm okay, feeling fine")
    assert analysis.action == "okay"


def test_fallback_recognizes_schedule_recheck_with_minutes():
    analysis, _, _ = analyze_patient_checkin("check on me again in ten minutes")
    assert analysis.action == "schedule_recheck"
    assert analysis.follow_up_minutes == 10


def test_fallback_returns_unknown_for_unrelated_text():
    analysis, _, _ = analyze_patient_checkin("what time is it")
    assert analysis.action == "unknown"


def test_fallback_reports_inability_to_handle_non_english():
    analysis, source, _ = analyze_patient_checkin(
        "Estoy despierta, pero me siento confundida."
    )
    assert source == "fallback"
    assert analysis.action == "unknown"
    assert analysis.detected_language == "unknown-non-english"
