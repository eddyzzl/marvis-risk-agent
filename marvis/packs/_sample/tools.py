from __future__ import annotations

import time


def tool_echo(inputs: dict, _ctx) -> dict:
    return {"echoed": inputs["message"]}


def tool_fail(_inputs: dict, _ctx) -> dict:
    raise RuntimeError("sample failure")


def tool_bad_output(_inputs: dict, _ctx) -> dict:
    return {"wrong": True}


def tool_sleep(inputs: dict, _ctx) -> dict:
    time.sleep(float(inputs["seconds"]))
    return {"slept": True}
