"""Real-process end-to-end journey test (TST-3).

Unlike tests/test_frontend_playwright_smoke.py (a hand-written static server
fed canned JSON) and the TestClient-based integration tests, this test starts
a REAL uvicorn subprocess running the REAL FastAPI app (``marvis serve``)
against a temporary workspace, and drives a contracted six-journey-style
critical path purely over HTTP with no LLM configured (manual mode):

  1. create a task
  2. upload two CSV datasets (anchor + feature)
  3. propose -> confirm -> execute a JOIN (INV-3 gate; row-count invariant)
  4. create a standard_modeling plan on the joined dataset and drive it
     through every needs_confirmation gate (select experiment -> generate
     report -> post-training delivery actions) to completion
  5. assert a PMML artifact lands on disk in the task workspace

Kept intentionally small (~1500 rows, lr recipe, no tuning) so the whole
run finishes in a few minutes. Marked e2e + slow: excluded from the default
fast tier, run explicitly with ``pytest -m e2e``.
"""

from __future__ import annotations

import hashlib
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest


pytestmark = [pytest.mark.e2e, pytest.mark.slow]

ROWS = 1500
STARTUP_TIMEOUT_S = 60.0
STEP_POLL_TIMEOUT_S = 240.0
STEP_POLL_INTERVAL_S = 0.5


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_server(base_url: str, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"marvis serve exited early with code {proc.returncode}"
            )
        try:
            resp = httpx.get(f"{base_url}/api/branding", timeout=1.0)
            if resp.status_code == 200:
                return
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(0.2)
    raise RuntimeError(f"server did not become ready in time: {last_error}")


def _write_datasets(root: Path) -> tuple[Path, Path]:
    """Anchor (sample) + feature CSVs joinable on a phone-family key, matching
    the pattern the (Aligner-verified) data-join tests use for auto key
    detection: raw ``mobile`` on the anchor, md5(mobile) on the feature side.
    """
    rng = np.random.RandomState(11)
    phones = [f"138{i:08d}" for i in range(ROWS)]
    income = rng.normal(loc=8000, scale=3000, size=ROWS).clip(min=500)
    age = rng.randint(20, 60, size=ROWS)
    # Moderate (not perfect -- must stay well under the 0.95 leakage-correlation
    # guard) logistic relationship between income and bad_flag.
    logit = -1.5 + (-0.00025) * (income - 8000) / 1000.0
    prob = 1.0 / (1.0 + np.exp(-logit))
    bad_flag = (rng.rand(ROWS) < prob).astype(int)
    split = np.array(
        (["train"] * int(ROWS * 0.6))
        + (["test"] * int(ROWS * 0.25))
        + (["oot"] * (ROWS - int(ROWS * 0.6) - int(ROWS * 0.25)))
    )
    rng.shuffle(split)

    sample_path = root / "sample.csv"
    pd.DataFrame(
        {"mobile": phones, "bad_flag": bad_flag, "split": split}
    ).to_csv(sample_path, index=False)

    feature_path = root / "feature.csv"
    pd.DataFrame(
        {
            "phone_md5": [hashlib.md5(p.encode()).hexdigest() for p in phones],
            "income": income,
            "age": age,
        }
    ).to_csv(feature_path, index=False)
    return sample_path, feature_path


def _poll_plan_until(
    client: httpx.Client, plan_id: str, *, statuses: set[str], timeout: float
) -> dict:
    deadline = time.monotonic() + timeout
    last_plan: dict = {}
    while time.monotonic() < deadline:
        resp = client.get(f"/api/plans/{plan_id}")
        assert resp.status_code == 200, resp.text
        last_plan = resp.json()["plan"]
        if last_plan["status"] in statuses:
            return last_plan
        time.sleep(STEP_POLL_INTERVAL_S)
    raise AssertionError(
        f"plan {plan_id} did not reach {statuses} within {timeout}s; "
        f"last status={last_plan.get('status')!r}"
    )


def _confirm_gate_and_wait(client: httpx.Client, plan_id: str, step_id: str) -> dict:
    """Confirm a gate step and wait for the executor's background job to make
    real progress on it.

    The confirm endpoint returns 202 and schedules the run as a FastAPI
    BackgroundTask; the HTTP response can land before that task starts, so a
    plan poll immediately after 202 can still observe the *pre-confirm*
    snapshot (the just-confirmed step still shows "awaiting_confirm", and so
    does the plan). Wait for the confirmed step itself to leave
    "awaiting_confirm" before applying the caller's target-status wait --
    otherwise the stale snapshot is mistaken for the *next* gate.
    """
    resp = client.post(f"/api/plans/{plan_id}/steps/{step_id}/confirm")
    assert resp.status_code == 202, resp.text

    deadline = time.monotonic() + STEP_POLL_TIMEOUT_S
    plan: dict = {}
    while time.monotonic() < deadline:
        resp = client.get(f"/api/plans/{plan_id}")
        assert resp.status_code == 200, resp.text
        plan = resp.json()["plan"]
        confirmed_step = next(s for s in plan["steps"] if s["id"] == step_id)
        if confirmed_step["status"] != "awaiting_confirm":
            break
        time.sleep(STEP_POLL_INTERVAL_S)
    else:
        raise AssertionError(
            f"step {step_id} still awaiting_confirm after being confirmed "
            f"({STEP_POLL_TIMEOUT_S}s timeout)"
        )

    return _poll_plan_until(
        client,
        plan_id,
        statuses={"awaiting_confirm", "done", "failed", "review"},
        timeout=STEP_POLL_TIMEOUT_S,
    )


def _step_by_title(plan: dict, title: str) -> dict:
    for step in plan["steps"]:
        if step["title"] == title:
            return step
    raise AssertionError(f"step {title!r} not found in plan {plan['id']}")


@pytest.fixture
def server(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = dict(os.environ)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "marvis",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--workspace",
            str(workspace),
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_server(base_url, proc, STARTUP_TIMEOUT_S)
        yield base_url, workspace
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def test_real_server_join_and_modeling_journey(server, tmp_path: Path):
    base_url, workspace = server
    materials_dir = tmp_path / "materials"
    materials_dir.mkdir()
    sample_path, feature_path = _write_datasets(materials_dir)

    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        # 1. Create task. source_dir must resolve under an allowed material
        # root (workspace or $HOME); the workspace itself always qualifies.
        resp = client.post(
            "/api/tasks",
            json={
                "model_name": "e2e旅程",
                "validator": "qa",
                "source_dir": str(workspace),
                "task_type": "data_join",
                "run_mode": "manual",
            },
        )
        assert resp.status_code == 200, resp.text
        task_id = resp.json()["id"]

        # 2. Upload two CSV datasets.
        with sample_path.open("rb") as fh:
            resp = client.post(
                f"/api/tasks/{task_id}/datasets/upload",
                files={"file": ("sample.csv", fh, "text/csv")},
                data={"role": "sample"},
            )
        assert resp.status_code == 201, resp.text
        anchor_id = resp.json()["datasets"][0]["id"]
        anchor_rows = resp.json()["datasets"][0]["row_count"]
        assert anchor_rows == ROWS

        with feature_path.open("rb") as fh:
            resp = client.post(
                f"/api/tasks/{task_id}/datasets/upload",
                files={"file": ("feature.csv", fh, "text/csv")},
                data={"role": "feature"},
            )
        assert resp.status_code == 201, resp.text
        feature_id = resp.json()["datasets"][0]["id"]

        # 3. JOIN: propose -> confirm -> execute (INV-3 forced-confirm gate).
        resp = client.post(
            f"/api/tasks/{task_id}/joins/propose",
            json={"anchor_dataset_id": anchor_id, "feature_dataset_ids": [feature_id]},
        )
        assert resp.status_code == 201, resp.text
        join_plan = resp.json()
        join_plan_id = join_plan["join_plan_id"]
        assert join_plan["joins"], "join engine found no key pairs (mobile/phone_md5 alignment failed)"

        resp = client.post(
            f"/api/joins/{join_plan_id}/confirm",
            json={"feature_id": feature_id},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["joins"][0]["confirmed"] is True

        resp = client.post(f"/api/joins/{join_plan_id}/execute", json={})
        assert resp.status_code == 200, resp.text
        joined = resp.json()
        result_dataset_id = joined["result_dataset_id"]
        # INV-3: anchor row count preserved 1:1 through a left join.
        assert joined["anchor_rows"] == ROWS

        resp = client.get(f"/api/tasks/{task_id}/datasets")
        assert resp.status_code == 200, resp.text
        joined_dataset = next(
            d for d in resp.json()["datasets"] if d["id"] == result_dataset_id
        )
        assert joined_dataset["row_count"] == ROWS

        # 4. standard_modeling plan on the joined dataset, driven gate-by-gate.
        resp = client.post(
            f"/api/tasks/{task_id}/plans",
            json={
                "goal": "标准建模",
                "slots": {
                    "dataset_id": result_dataset_id,
                    "target_col": "bad_flag",
                    "feature_cols": ["income", "age"],
                    "split_col": "split",
                    "split_values": {"train": "train", "test": "test", "oot": "oot"},
                    "recipe": "lr",
                    "seed": 7,
                },
            },
        )
        assert resp.status_code == 201, resp.text
        plan = resp.json()["plan"]
        plan_id = plan["id"]
        assert plan["template_id"] == "standard_modeling"

        resp = client.post(f"/api/plans/{plan_id}/confirm")
        assert resp.status_code == 200, resp.text

        resp = client.post(f"/api/plans/{plan_id}/run")
        assert resp.status_code == 202, resp.text

        plan = _poll_plan_until(
            client,
            plan_id,
            statuses={"awaiting_confirm", "done", "failed", "review"},
            timeout=STEP_POLL_TIMEOUT_S,
        )
        assert plan["status"] == "awaiting_confirm", plan
        gate_step = _step_by_title(plan, "选择实验")
        assert gate_step["status"] == "awaiting_confirm", gate_step

        # Gate 1: 选择实验 (select_experiment) -> runs 生成模型开发报告, which is
        # itself a needs_confirmation gate -> pauses again.
        plan = _confirm_gate_and_wait(client, plan_id, gate_step["id"])
        assert plan["status"] == "awaiting_confirm", plan
        report_step = _step_by_title(plan, "生成模型开发报告")
        assert report_step["status"] == "awaiting_confirm", report_step

        # Gate 2: 生成模型开发报告 -> runs 模型交付动作 gate -> pauses again.
        plan = _confirm_gate_and_wait(client, plan_id, report_step["id"])
        assert plan["status"] == "awaiting_confirm", plan
        delivery_step = _step_by_title(plan, "模型交付动作")
        assert delivery_step["status"] == "awaiting_confirm", delivery_step

        # Gate 3: 模型交付动作 -> export_pmml + handoff_to_validation run for real.
        plan = _confirm_gate_and_wait(client, plan_id, delivery_step["id"])
        assert plan["status"] == "done", plan
        delivery_step = _step_by_title(plan, "模型交付动作")
        assert delivery_step["status"] == "done", delivery_step

        # 5. PMML artifact landed on disk in the task workspace.
        artifacts_dir = workspace / "tasks" / task_id / "modeling_artifacts"
        pmml_files = list(artifacts_dir.glob("*.pmml"))
        assert pmml_files, f"no .pmml file under {artifacts_dir}"
        assert pmml_files[0].stat().st_size > 0
