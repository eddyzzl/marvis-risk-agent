from marvis.agent.json_reply import load_json_object, strip_thinking


def test_strip_thinking_removes_paired_blocks():
    text = "before <think>reasoning</think> after"
    assert strip_thinking(text) == "before  after"


def test_strip_thinking_removes_unclosed_trailing_block():
    text = 'answer {"a": 1} <think>still thinking with no close'
    assert strip_thinking(text) == 'answer {"a": 1} '


def test_strip_thinking_is_case_insensitive_and_handles_multiple():
    text = "<THINK>a</THINK>x<think>b</think>y"
    assert strip_thinking(text) == "xy"


def test_load_json_object_prefers_final_answer_over_think_draft():
    raw = '<think>draft: {"action":"confirm"}</think>{"action":"halt","reason":"anomaly"}'
    data, error = load_json_object(raw)
    assert error is None
    assert data == {"action": "halt", "reason": "anomaly"}


def test_load_json_object_prefers_last_object_when_two_present():
    raw = 'noise {"action":"confirm"} more noise {"action":"halt"} trailing'
    data, error = load_json_object(raw)
    assert error is None
    assert data == {"action": "halt"}


def test_load_json_object_still_parses_a_clean_single_object():
    data, error = load_json_object('{"ok": true}')
    assert error is None
    assert data == {"ok": True}


def test_load_json_object_returns_error_on_non_json():
    data, error = load_json_object("no json here")
    assert data is None
    assert error
