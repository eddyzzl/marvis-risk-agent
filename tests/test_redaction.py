"""TST-6: positive/negative redaction fixture matrix.

Locks in the precision fixes: bare 13-19 digit runs are only masked as bank
cards when they either pass a Luhn checksum or match a recognized card BIN
prefix+length -- ordinary large integers (timestamps, row counts, order
numbers) must stay legible in audit evidence. Also covers the previously
under-masked 15-digit legacy national ID and separator/country-code phone
formats, and the write-time redaction of agent_messages content.
"""

from marvis.db import TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.redaction import redact_text, redact_value


# --- positive cases: must be masked -----------------------------------------


def test_luhn_valid_card_number_is_masked():
    redacted = redact_text("card 4111111111111111 charged")
    assert "4111111111111111" not in redacted
    assert "4111********1111" in redacted


def test_unionpay_style_prefix_card_is_masked_even_without_valid_luhn():
    # 6222000000001234 does not satisfy Luhn but matches the UnionPay BIN
    # prefix (62) + 16-digit length -- this is the canonical fixture used
    # across the codebase's other redaction tests.
    redacted = redact_text("bank=6222000000001234")
    assert "6222000000001234" not in redacted
    assert "6222********1234" in redacted


def test_amex_style_prefix_card_is_masked():
    redacted = redact_text("amex 371234567890123 on file")
    assert "371234567890123" not in redacted


def test_18digit_national_id_is_masked():
    redacted = redact_text("id 110105199001011234 recorded")
    assert "110105199001011234" not in redacted


def test_18digit_national_id_with_x_checksum_is_masked():
    redacted = redact_text("id 11010519900101123X recorded")
    assert "11010519900101123X" not in redacted


def test_15digit_legacy_national_id_is_masked():
    # TST-6 under-masking gap: pre-1999 legacy resident IDs are 15 digits
    # with no check character and were not matched at all before this fix.
    redacted = redact_text("legacy id 110105850101123 filed")
    assert "110105850101123" not in redacted


def test_bare_mobile_number_is_masked():
    redacted = redact_text("call 13812345678 now")
    assert "13812345678" not in redacted


def test_country_code_and_hyphen_separated_phone_is_masked():
    redacted = redact_text("call +86 138-1234-5678 now")
    assert "138-1234-5678" not in redacted


def test_space_separated_phone_is_masked():
    redacted = redact_text("call 138 1234 5678 now")
    assert "138 1234 5678" not in redacted


def test_country_code_without_separators_phone_is_masked():
    redacted = redact_text("call 8613812345678 now")
    assert "13812345678" not in redacted


def test_email_is_still_masked():
    redacted = redact_text("contact eddy@example.com please")
    assert "eddy@example.com" not in redacted
    assert "[REDACTED_EMAIL]" in redacted


# --- negative cases: must NOT be masked (over-masking regressions) ---------


def test_13digit_millisecond_timestamp_is_not_masked():
    text = "created_at 1735689600123 logged"
    assert redact_text(text) == text


def test_13digit_row_count_is_not_masked():
    text = "processed 9999999999999 rows"
    assert redact_text(text) == text


def test_18digit_order_number_is_not_masked():
    text = "order 20260701000012345 confirmed"
    assert redact_text(text) == text


def test_ordinary_16digit_integer_without_card_prefix_is_not_masked():
    # Starts with "90", not a recognized BIN prefix, and not Luhn-valid.
    text = "batch id 9012345678901234 queued"
    assert redact_text(text) == text


def test_12digit_customer_number_is_not_masked():
    # Below the 13-digit floor entirely.
    text = "客户号 622200000001 已建档"
    assert redact_text(text) == text


# --- structured values -------------------------------------------------------


def test_redact_value_masks_nested_dict_and_list_strings():
    result = redact_value(
        {
            "note": "银行卡 6222000000001234 只保留汇总口径。",
            "history": ["order 20260701000012345 confirmed", "call 13812345678 now"],
        }
    )
    assert "6222000000001234" not in result.value["note"]
    assert "20260701000012345" in result.value["history"][0]
    assert "13812345678" not in result.value["history"][1]
    assert result.redacted_count >= 2


def test_redact_value_masks_sensitive_key_regardless_of_content():
    result = redact_value({"password": "harmless-short-value"})
    assert result.value["password"] == "[REDACTED]"
    assert result.redacted_count == 1


# --- write-time redaction of agent_messages (source transcript) ------------


def _task_create(**overrides) -> TaskCreate:
    values = {
        "model_name": "模型",
        "model_version": "v1",
        "validator": "验证人员",
        "source_dir": "/tmp/source",
        "algorithm": "lgb",
        "run_mode": "manual",
        "target_col": "y",
        "score_col": "pred",
        "split_col": "split",
        "time_col": "apply_month",
        "feature_columns": [],
        "notebook_path": None,
        "sample_path": None,
        "pmml_path": None,
        "dictionary_path": None,
        "report_values": {},
    }
    values.update(overrides)
    return TaskCreate(**values)


def _create_task(repo: TaskRepository) -> str:
    task = repo.create_task(_task_create())
    return task.id


def test_add_agent_message_redacts_content_before_persisting(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task_id = _create_task(repo)

    message = repo.add_agent_message(
        task_id,
        role="user",
        stage="chat",
        content="样本行：姓名张三，身份证110105199001011234，手机13812345678",
    )

    assert "110105199001011234" not in message["content"]
    assert "13812345678" not in message["content"]
    stored = repo.list_agent_messages(task_id)
    assert "110105199001011234" not in stored[0]["content"]
    assert "13812345678" not in stored[0]["content"]


def test_add_agent_message_does_not_redact_ordinary_numbers():
    # Regression guard for the over-masking half of TST-6: ordinary chat
    # content with large-but-ordinary integers must survive unredacted.
    text = "本次处理 9999999999999 行，耗时 1735689600123 毫秒"
    assert redact_text(text) == text


def test_update_agent_message_redacts_streamed_content(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task_id = _create_task(repo)

    message = repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content="",
        metadata={"streaming": True},
    )
    updated = repo.update_agent_message(
        message["id"],
        content="客户手机号是13812345678，请核实",
        metadata={"streaming": False},
    )

    assert "13812345678" not in updated["content"]


def test_add_agent_message_does_not_redact_metadata(tmp_path):
    # metadata carries structured control-flow state consumed by key/shape
    # (e.g. gate detection keyed on "join_c1" being present) -- only content
    # is redacted, never metadata.
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task_id = _create_task(repo)

    message = repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content="ok",
        metadata={"join_c1": "13812345678", "phone": "13812345678"},
    )

    assert message["metadata"]["join_c1"] == "13812345678"
    assert message["metadata"]["phone"] == "13812345678"
