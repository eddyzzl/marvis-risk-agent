"""Monitoring-schedule form filling sees platform choice lists, never data rows."""
from marvis.llm_outbound import OutboundContextError, _mint_prompt
from marvis.llm_outbound_instructions import user_instruction


FREQUENCIES = ("daily", "weekly", "monthly")
NOTIFY_CHANNELS = ("local_inbox",)
_MAX_CHOICES = 200
_MAX_COLUMNS = 2000


def _choice_ids(items, key="id"):
    if not isinstance(items, (list, tuple)) or len(items) > _MAX_CHOICES:
        raise OutboundContextError("调度表单可选项无效或超限，未发送请求。")
    ids = [item.get(key) for item in items if isinstance(item, dict)]
    if len(ids) != len(items) or any(not isinstance(value, str) or not value or len(value) > 200 for value in ids):
        raise OutboundContextError("调度表单可选项编号无效，未发送请求。")
    return ids


def _columns(options):
    names = sorted({column for item in options["datasets"] for column in item.get("columns", ())})
    if len(names) > _MAX_COLUMNS or any(not isinstance(name, str) or len(name) > 200 for name in names):
        raise OutboundContextError("调度表单字段列表无效或超限，未发送请求。")
    return names


def schedule_form_prompt(instruction, options, *, today):
    """``options`` is :func:`marvis.operations.conversation_schedule.schedule_form_options`."""
    _choice_ids(options.get("targets")), _choice_ids(options.get("datasets"))
    if not isinstance(today, str) or len(today) != 10:
        raise OutboundContextError("调度表单日期上下文无效，未发送请求。")
    return _mint_prompt({
        "stage": "monitoring_schedule_form",
        "instruction": str(user_instruction(instruction)),
        "today": today,
        "targets": [{"id": item["id"], "kind": item["kind"], "label": item["label"]} for item in options["targets"]],
        "datasets": [{"id": item["id"], "row_count": item["row_count"], "columns": list(item["columns"])}
                     for item in options["datasets"]],
        "frequencies": list(FREQUENCIES),
        "notify_channels": list(NOTIFY_CHANNELS),
        "instruction_rules": (
            "只按用户原话填写业务表单字段；用户没说的字段返回 null，不要猜。"
            "target_id、dataset_id、time_col 只能取给定列表里的值。start_date 用 YYYY-MM-DD，"
            "相对日期按 today 换算。不能创建调度、执行监控或计算指标；确认与创建由本地平台控制。"
        ),
    })


def installed_schedule_form_schema(bindings):
    options = bindings.get("options")
    if set(bindings) != {"options"} or not isinstance(options, dict):
        raise OutboundContextError("调度表单输出契约需要平台可选项，未发送请求。")
    targets, datasets = _choice_ids(options.get("targets")), _choice_ids(options.get("datasets"))
    columns = _columns(options)

    def nullable_enum(values):
        return {"type": ["string", "null"], "enum": [*values, None]}

    return {"name": "monitoring_schedule_form", "strict": False, "schema": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "target_id": nullable_enum(targets),
            "dataset_id": nullable_enum(datasets),
            "time_col": nullable_enum(columns),
            "frequency": nullable_enum(FREQUENCIES),
            "interval": {"type": ["integer", "null"], "minimum": 1, "maximum": 12},
            "start_date": {"type": ["string", "null"], "pattern": r"^\d{4}-\d{2}-\d{2}$"},
            "notify": nullable_enum(NOTIFY_CHANNELS),
            "clarify": {"type": ["string", "null"], "maxLength": 300},
        },
    }}
