"""Collections requests in conversation (V3 D-7).

Collections has no task type of its own: it lives in the strategy task's
collections reference workspace (Inspector → 催收参考工作台). A conversation
turn that asks for collections work is answered deterministically with where
that work happens and, plainly, which parts of the request the platform cannot
do. Nothing here creates a plan, runs a Tool or changes task state.
"""

from __future__ import annotations

import re

from marvis.domain import TASK_TYPE_STRATEGY
from marvis.domain import TaskRecord
from marvis.repositories.tasks import TaskRepository
from marvis.ui_locale import detect_text_language

# Only a strong collections intent counts: 催收 next to collections work
# (策略、分案、排队、批次…) or a verb that asks for that work, and in English
# "collections <work>" or "debt collection". A bare 贷后管理 / "collection"
# does not (mirrors js/home-intent.js).
_COLLECTIONS_RE = re.compile(
    r"催收.{0,4}(?:策略|分案|分配|排队|批次|案件|名单|队列|动作|计划|方案|工作台|流程|规则|外呼|话术|频次)"
    r"|(?:做|设计|制定|优化|搭建|安排|开展|处理)(?:一套|一版|一下|个)?.{0,6}催收(?!成本|费用|费率|评分|模型|卡|回款率)"
    r"|(?<![A-Za-z])(?:debt\s+collections?|collections\s+(?:strateg(?:y|ies)|polic(?:y|ies)|queues?|batch(?:es)?|cases?"
    r"|workflows?|campaigns?|actions?|assignments?|plans?|workspace))(?![A-Za-z])",
    re.IGNORECASE,
)
# A model, a recovery-rate target or a vintage/feature reading is modeling,
# feature or risk work that merely mentions collections.
_OTHER_WORK_RE = re.compile(
    r"模型|评分卡|[Cc]\s?卡|回款率|催收成本|费率|vintage|vtg|账龄|特征|变量"
    r"|(?<![A-Za-z])(?:model|models|scorecard|features?|vintage)(?![A-Za-z])",
    re.IGNORECASE,
)
# The conversation handler answers only where collections routing is otherwise
# ambiguous: the strategy task that owns the workspace, and data processing.
# Modeling, feature, validation and risk tasks keep their own routing.
GUIDED_TASK_TYPES = frozenset({TASK_TYPE_STRATEGY, "data_join"})

_UNSUPPORTED = (
    (
        re.compile(r"外呼|打电话|拨打|短信|联系客户|触达客户|话术|\b(?:call|sms|text message|script)s?\b", re.I),
        "联系客户（外呼、短信、话术）",
        "contacting customers (calls, SMS, scripts)",
    ),
    (
        re.compile(r"实时|对接.{0,6}(?:系统|平台)|上线催收|real[- ]?time|integrat", re.I),
        "对接外部催收系统或实时调度",
        "connecting to an external collections system or real-time dispatch",
    ),
    (
        re.compile(
            r"(?:优化|最优|自动).{0,6}催收|(?:预测|提升|保证).{0,4}回收|回收(?:率)?预测|催收效果|"
            r"optimi[sz]e|predict|guarantee",
            re.I,
        ),
        "自动优化催收动作或预测、保证回收效果",
        "optimizing collections actions automatically, or predicting or guaranteeing recoveries",
    ),
)


def is_collections_request(text: str | None) -> bool:
    value = str(text or "")
    return bool(_COLLECTIONS_RE.search(value)) and not _OTHER_WORK_RE.search(value)


def unsupported_parts(text: str | None) -> list[tuple[str, str]]:
    value = str(text or "")
    return [(zh, en) for pattern, zh, en in _UNSUPPORTED if pattern.search(value)]


def guidance_text(task_type: str, text: str) -> str:
    english = detect_text_language(text) == "en"
    unsupported = unsupported_parts(text)
    if english:
        parts = [
            (
                "Collections runs in this strategy task's Collections reference workspace: "
                "open the Inspector (top right) and choose it."
                if task_type == TASK_TYPE_STRATEGY
                else "Collections has no task of its own; it runs inside a strategy task. "
                "Go back to the home page, describe the collections work and create it as a strategy task, "
                "then open the Collections reference workspace in its Inspector."
            ),
            "There you can register cases, repayment schedules and cash flows, generate reference batches "
            "from your assignment rules, queue and run them locally after review, and reconcile costs and recoveries.",
        ]
        if unsupported:
            parts.append(
                "Not available yet: " + "; ".join(en for _, en in unsupported) + ". "
                "The platform does not contact customers, and there are no reviewed definitions yet "
                "for collections actions, costs, capacity or recoveries."
            )
        parts.append("I did not create a plan or run any tool.")
        return " ".join(parts)
    parts = [
        (
            "催收在当前策略任务的「催收参考工作台」里做：打开右上角「检查器」，选「催收参考工作台」。"
            if task_type == TASK_TYPE_STRATEGY
            else "催收不单独建任务，放在策略任务里：回到首页说一句催收需求，按策略任务创建，"
            "再打开该任务「检查器」里的「催收参考工作台」。"
        ),
        "在那里可以登记案件、还款计划和回款流水，按你写的分案规则生成参考批次，"
        "复核后做本地参考排队和执行，再核对成本与回收。",
    ]
    if unsupported:
        parts.append(
            "现在做不到：" + "、".join(zh for zh, _ in unsupported) + "。"
            "平台不联系客户，也还没有经过评审的催收动作、成本、产能和回收口径。"
        )
    parts.append("我没有创建计划，也没有执行工具。")
    return "".join(parts)


def maybe_handle_collections_request(
    repo: TaskRepository,
    task: TaskRecord,
    *,
    text: str,
) -> dict | None:
    """Answer a collections request with its entry; ``None`` when not one."""

    if task.task_type not in GUIDED_TASK_TYPES or not is_collections_request(text):
        return None
    repo.add_agent_message(
        task.id,
        role="user",
        stage="chat",
        content=text,
        metadata={"intent": "collections_guidance"},
    )
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=guidance_text(task.task_type, text),
        metadata={
            "intent": "collections_guidance",
            "kind": "guidance",
            "code": "collections_entry_guidance",
            "title": "催收需求",
            "entry": "collectionWorkspacePanel" if task.task_type == TASK_TYPE_STRATEGY else "strategy_task",
            "unsupported": [zh for zh, _ in unsupported_parts(text)],
            # Display-only quick action: opens the Inspector panel, sends nothing.
            **(
                {"inspector_actions": [{"label": "打开催收参考工作台", "item": "collectionWorkspacePanel"}]}
                if task.task_type == TASK_TYPE_STRATEGY
                else {}
            ),
        },
    )
    return {
        "task_id": task.id,
        # Same turn status as other no-plan replies, so callers treat it alike.
        "status": "clarification_required",
        "code": "collections_entry_guidance",
        "messages": repo.list_agent_messages(task.id),
    }
