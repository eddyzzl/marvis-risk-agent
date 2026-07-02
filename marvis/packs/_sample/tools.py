from __future__ import annotations

import random
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


def tool_random(_inputs: dict, ctx) -> dict:
    return {"value": random.random(), "seed": ctx.seed}


def tool_memory_hog(inputs: dict, _ctx) -> dict:
    megabytes = int(inputs["megabytes"])
    hold_seconds = float(inputs.get("hold_seconds", 5.0))
    # Touch every page so RSS actually reflects the allocation (a bare
    # bytearray(n) can stay lazily-mapped and under-report in RSS samples).
    buffer = bytearray(megabytes * 1024 * 1024)
    step = 4096
    for offset in range(0, len(buffer), step):
        buffer[offset] = 1
    time.sleep(hold_seconds)
    return {"allocated_mb": megabytes}
