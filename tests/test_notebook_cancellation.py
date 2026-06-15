from marvis.notebook_cancellation import NotebookCancellationRegistry


def test_pending_notebook_cancellation_can_be_cleared_before_later_retry():
    registry = NotebookCancellationRegistry()

    assert registry.request_cancel("task-1") is False
    registry.clear_pending("task-1")
    token = registry.register("task-1")

    assert token.is_cancelled() is False
