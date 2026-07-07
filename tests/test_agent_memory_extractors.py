from marvis.agent_memory.extractors import (
    extract_field_convention,
    extract_memory_candidates,
    extract_model_experience,
    extract_task_experience,
    extract_user_preference,
    extract_validation_pitfall,
)


def test_extract_model_experience_from_structured_validation_results():
    result = {
        "task_id": "task-202601",
        "model_name": "分润通用A卡模型",
        "model_version": "V2026",
        "month": "202601",
        "channel": "自营",
        "scope": "mob3贷前A卡",
        "metrics": {"ks": 30.4, "auc": 0.721, "psi": 0.083},
        "important_feature_sources": ["征信", "交易"],
    }

    candidate = extract_model_experience(result)

    assert candidate is not None
    assert candidate.memory_type == "model_experience"
    assert candidate.source_task_id == "task-202601"
    assert candidate.confidence == "high"
    assert candidate.payload == {
        "ks": 30.4,
        "auc": 0.721,
        "psi": 0.083,
        "month": "202601",
        "channel": "自营",
        "model_name": "分润通用A卡模型",
        "model_version": "V2026",
        "scope": "mob3贷前A卡",
        "source_task_id": "task-202601",
        "important_feature_sources": ["征信", "交易"],
    }


def test_extract_model_experience_keeps_metrics_when_model_version_missing():
    result = {
        "task_id": "task-202601",
        "model_name": "分润通用A卡模型",
        "model_version": "",
        "month": "202601",
        "channel": "自营",
        "scope": "mob3贷前A卡",
        "metrics": {"ks": 30.4, "auc": 0.721, "psi": 0.083},
        "important_feature_sources": ["征信", "交易"],
    }

    candidate = extract_model_experience(result)

    assert candidate is not None
    assert candidate.memory_type == "model_experience"
    assert candidate.payload["model_version"] == "未标注"
    assert "PSI为0.083" in candidate.summary


def test_extract_model_experience_summary_formats_long_float_metrics():
    result = {
        "task_id": "task-202601",
        "model_name": "A卡模型",
        "model_version": "",
        "month": "202601",
        "channel": "自营",
        "scope": "mob3贷前A卡",
        "metrics": {
            "ks": 0.33306610440224665,
            "auc": 0.7244784277697583,
            "psi": 0.048773171474789045,
        },
        "important_feature_sources": ["征信"],
    }

    candidate = extract_model_experience(result)

    assert candidate is not None
    assert "KS为0.333066" in candidate.summary
    assert "AUC为0.724478" in candidate.summary
    assert "PSI为0.0487732" not in candidate.summary
    assert "PSI为0.048773" in candidate.summary


def test_extract_validation_pitfall_from_notebook_pmml_field_execution_and_report_failures():
    failures = [
        {"kind": "notebook", "message": "RMC_SCORE_FN missing, notebook cannot score sample"},
        {"kind": "pmml", "message": "PMML input type mismatch for score"},
        {"kind": "field", "message": "target column y_true not found"},
        {"kind": "execution", "message": "validation process timed out"},
        {"kind": "report", "message": "Word report rendering failed"},
    ]

    candidates = extract_validation_pitfall(
        {"task_id": "task-failed", "failures": failures}
    )

    assert [candidate.memory_type for candidate in candidates] == [
        "validation_pitfall",
        "validation_pitfall",
        "validation_pitfall",
        "validation_pitfall",
        "validation_pitfall",
    ]
    assert {candidate.payload["failure_kind"] for candidate in candidates} == {
        "notebook",
        "pmml",
        "field",
        "execution",
        "report",
    }
    assert all(candidate.source_task_id == "task-failed" for candidate in candidates)


def test_extract_task_experience_from_completed_and_failed_summaries():
    completed = extract_task_experience(
        {
            "task_id": "task-ok",
            "status": "completed",
            "summary": "完成模型验证，分数一致性通过，报告已生成。",
        }
    )
    failed = extract_task_experience(
        {
            "task_id": "task-bad",
            "status": "failed",
            "summary": "Notebook 缺少 RMC_SAMPLE_DF，任务失败。",
        }
    )

    assert completed is not None
    assert completed.memory_type == "task_experience"
    assert completed.payload["status"] == "completed"
    assert completed.source_task_id == "task-ok"
    assert failed is not None
    assert failed.memory_type == "task_experience"
    assert failed.payload["status"] == "failed"
    assert failed.source_task_id == "task-bad"


def test_extract_field_convention_from_task_column_settings():
    candidate = extract_field_convention(
        {
            "task_id": "task-fields",
            "target_col": "bad_flag",
            "score_col": "prob",
            "split_col": "sample_type",
            "time_col": "apply_month",
            "channel_col": "channel_name",
        }
    )

    assert candidate is not None
    assert candidate.memory_type == "field_convention"
    assert candidate.source_task_id == "task-fields"
    assert candidate.payload == {
        "target_col": "bad_flag",
        "score_col": "prob",
        "split_col": "sample_type",
        "time_col": "apply_month",
        "channel_col": "channel_name",
    }


def test_extract_user_preference_from_explicit_remember_and_correction_messages():
    remember = extract_user_preference(
        {
            "message_id": "msg-1",
            "text": "请记住：以后报告里 PSI 统一写成稳定性指标。",
        }
    )
    correction = extract_user_preference(
        {
            "message_id": "msg-2",
            "text": "纠正一下：AUC 展示保留三位小数。",
        }
    )

    assert remember is not None
    assert remember.memory_type == "user_preference"
    assert remember.source_message_id == "msg-1"
    assert remember.payload["preference"] == "以后报告里 PSI 统一写成稳定性指标。"
    assert correction is not None
    assert correction.memory_type == "user_preference"
    assert correction.source_message_id == "msg-2"
    assert correction.payload["preference"] == "AUC 展示保留三位小数。"


def test_extract_user_preference_truncates_long_explicit_memory():
    candidate = extract_user_preference(
        {
            "message_id": "msg-long",
            "text": "请记住：" + "报告措辞保持克制。" * 40,
        }
    )

    assert candidate is not None
    assert len(candidate.summary) <= 203
    assert candidate.summary.endswith("...")
    assert candidate.payload["preference"] == candidate.summary


def test_skill_experience_reserved_does_not_create_active_runtime_candidates():
    candidates = extract_memory_candidates(
        task_result={
            "task_id": "task-skill",
            "status": "completed",
            "summary": "用户要求以后自动运行某个 skill runtime。",
            "skill_experience": {"name": "auto-validator", "workflow": ["run"]},
        },
        messages=[
            {
                "message_id": "msg-skill",
                "text": "请记住这个 skill：以后自动调用 auto-validator。",
            }
        ],
    )

    assert all(
        candidate.memory_type != "skill_experience_reserved" for candidate in candidates
    )
    assert all("skill_experience" not in candidate.payload for candidate in candidates)


def test_extract_user_preference_captures_marker_mid_sentence():
    # MEM-9: a marker later in the sentence ("好的，请记住：...") must not be
    # dropped just because it fails a hard text.startswith check.
    candidate = extract_user_preference(
        {
            "message_id": "msg-mid",
            "text": "好的，请记住：以后报告用英文。",
        }
    )

    assert candidate is not None
    assert candidate.memory_type == "user_preference"
    assert candidate.payload["preference"] == "以后报告用英文。"


def test_extract_user_preference_widened_trigger_words():
    # MEM-9: the trigger vocabulary widened beyond the original six literal
    # markers to cover other common phrasings ("记一下", "以后都").
    remember_short = extract_user_preference(
        {"message_id": "msg-short", "text": "记一下：以后报告统一用小数点两位。"}
    )
    from_now_on = extract_user_preference(
        {"message_id": "msg-future", "text": "以后都用中文写摘要。"}
    )

    assert remember_short is not None
    assert remember_short.payload["preference"] == "以后报告统一用小数点两位。"
    assert from_now_on is not None
    assert from_now_on.payload["preference"] == "用中文写摘要。"


def test_extract_user_preference_no_longer_vetoed_by_bare_runtime_substring():
    # MEM-9: this exact example from the review -- a lightgbm hyperparameter
    # named 'runtime' inside an explicit "please remember" instruction -- used
    # to be silently dropped because the whole message contained the
    # substring 'runtime'. The topic here is a training hyperparameter, not
    # the reserved skill/tool runtime, so it must now be captured.
    candidate = extract_user_preference(
        {
            "message_id": "msg-runtime-param",
            "text": "请记住：训练时用 lightgbm 的 runtime 参数 n_jobs=4",
        }
    )

    assert candidate is not None
    assert candidate.memory_type == "user_preference"
    assert "n_jobs=4" in candidate.payload["preference"]


def test_extract_user_preference_still_vetoes_genuine_skill_runtime_topic():
    # The narrowed rule must still reject a message whose actual topic is
    # invoking/running a skill or tool runtime (skill/runtime marker AND an
    # execute/run/invoke marker), matching
    # test_skill_experience_reserved_does_not_create_active_runtime_candidates.
    candidate = extract_user_preference(
        {
            "message_id": "msg-skill-run",
            "text": "请记住：以后自动执行这个 skill 的 runtime。",
        }
    )

    assert candidate is None


def test_classify_user_preference_capture_reports_reserved_topic_reason():
    from marvis.agent_memory.extractors import (
        USER_PREFERENCE_CAPTURED,
        USER_PREFERENCE_NO_MARKER,
        USER_PREFERENCE_RESERVED_TOPIC,
        classify_user_preference_capture,
    )

    reserved = classify_user_preference_capture(
        {"message_id": "msg-1", "text": "请记住这个 skill：以后自动调用 auto-validator。"}
    )
    captured = classify_user_preference_capture(
        {"message_id": "msg-2", "text": "请记住：以后报告用英文。"}
    )
    no_marker = classify_user_preference_capture(
        {"message_id": "msg-3", "text": "今天的报告看起来不错。"}
    )

    assert reserved == USER_PREFERENCE_RESERVED_TOPIC
    assert captured == USER_PREFERENCE_CAPTURED
    assert no_marker == USER_PREFERENCE_NO_MARKER
