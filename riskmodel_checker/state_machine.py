from riskmodel_checker.domain import TaskStatus


ALLOWED_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.CREATED: frozenset({TaskStatus.SCANNED, TaskStatus.FAILED}),
    TaskStatus.SCANNED: frozenset(
        {TaskStatus.SCANNED, TaskStatus.RUNNING, TaskStatus.FAILED}
    ),
    TaskStatus.RUNNING: frozenset(
        {TaskStatus.SCANNED, TaskStatus.EXECUTED, TaskStatus.FAILED}
    ),
    TaskStatus.EXECUTED: frozenset(
        {
            TaskStatus.SCANNED,
            TaskStatus.RUNNING,
            TaskStatus.COMPUTING_METRICS,
            TaskStatus.SUCCEEDED,
            TaskStatus.REVIEW_REQUIRED,
            TaskStatus.FAILED,
        }
    ),
    TaskStatus.COMPUTING_METRICS: frozenset(
        {TaskStatus.EXECUTED, TaskStatus.WRITING_ARTIFACTS, TaskStatus.FAILED}
    ),
    TaskStatus.WRITING_ARTIFACTS: frozenset(
        {
            TaskStatus.SCANNED,
            TaskStatus.RUNNING,
            TaskStatus.COMPUTING_METRICS,
            TaskStatus.SUCCEEDED,
            TaskStatus.REVIEW_REQUIRED,
            TaskStatus.FAILED,
        }
    ),
    TaskStatus.SUCCEEDED: frozenset(
        {
            TaskStatus.SCANNED,
            TaskStatus.RUNNING,
            TaskStatus.COMPUTING_METRICS,
            TaskStatus.REVIEW_REQUIRED,
            TaskStatus.FAILED,
        }
    ),
    TaskStatus.FAILED: frozenset(
        {
            TaskStatus.SCANNED,
            TaskStatus.RUNNING,
            TaskStatus.COMPUTING_METRICS,
            TaskStatus.FAILED,
        }
    ),
    TaskStatus.REVIEW_REQUIRED: frozenset(
        {
            TaskStatus.SCANNED,
            TaskStatus.RUNNING,
            TaskStatus.COMPUTING_METRICS,
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
        }
    ),
}


class IllegalTransition(Exception):
    def __init__(self, current: TaskStatus, target: TaskStatus) -> None:
        super().__init__(f"illegal transition: {current.value} -> {target.value}")
        self.current = current
        self.target = target


class ConflictError(Exception):
    pass


def assert_transition(current: TaskStatus, target: TaskStatus) -> None:
    if target not in ALLOWED_TRANSITIONS.get(current, frozenset()):
        raise IllegalTransition(current, target)
