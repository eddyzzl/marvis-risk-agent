from marvis.notebook_cancellation import NotebookCancellationRegistry


def test_pending_notebook_cancellation_can_be_cleared_before_later_retry():
    registry = NotebookCancellationRegistry()

    assert registry.request_cancel("task-1") is False
    registry.clear_pending("task-1")
    token = registry.register("task-1")

    assert token.is_cancelled() is False


def test_active_cancel_for_old_job_cannot_cancel_new_retry_token():
    registry = NotebookCancellationRegistry()
    old_token = registry.register("task-1", job_id="job-old")
    registry.unregister("task-1", old_token)
    retry_token = registry.register("task-1", job_id="job-new")

    delivered = registry.request_cancel(
        "task-1",
        allow_pending=False,
        expected_job_id="job-old",
    )

    assert delivered is False
    assert retry_token.is_cancelled() is False
