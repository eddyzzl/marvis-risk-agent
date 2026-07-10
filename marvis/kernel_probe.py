"""Start a Python kernel and verify the Jupyter kernel_info handshake."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile

from jupyter_client import BlockingKernelClient
from jupyter_client.connect import write_connection_file


@dataclass(frozen=True)
class KernelProbeResult:
    ok: bool
    message: str
    returncode: int | None
    stdout_tail: str = ""
    stderr_tail: str = ""


def _kernel_environment(python_executable: Path) -> dict[str, str]:
    python_executable = Path(python_executable)
    runtime = python_executable.parent
    separator = ";" if python_executable.suffix.lower() == ".exe" else os.pathsep
    runtime_path = separator.join(
        str(path)
        for path in (
            runtime,
            runtime / "Scripts",
            runtime / "Library" / "bin",
        )
    )
    env = dict(os.environ)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env.update(
        {
            "CONDA_DEFAULT_ENV": str(runtime),
            "CONDA_PREFIX": str(runtime),
            "PATH": separator.join(filter(None, (runtime_path, env.get("PATH", "")))),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def probe_python_kernel(
    python_executable: Path,
    *,
    timeout: int = 30,
    cwd: Path | None = None,
) -> KernelProbeResult:
    python_executable = Path(python_executable)
    working_directory = Path(cwd) if cwd is not None else python_executable.parent
    client = None
    process = None
    probe_error: Exception | None = None
    handshake_succeeded = False

    with tempfile.TemporaryDirectory(prefix="marvis-kernel-probe-") as temp_dir:
        connection_file = str(Path(temp_dir) / "connection.json")
        write_connection_file(
            connection_file,
            ip="127.0.0.1",
            key=_connection_key(),
        )
        client = BlockingKernelClient(connection_file=connection_file)
        process = subprocess.Popen(
            [
                str(python_executable),
                "-m",
                "ipykernel_launcher",
                "-f",
                connection_file,
            ],
            cwd=str(working_directory),
            env=_kernel_environment(python_executable),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            client.load_connection_file()
            client.start_channels()
            client.wait_for_ready(timeout=max(1, int(timeout)))
            handshake_succeeded = True
        except Exception as exc:
            probe_error = exc
        finally:
            if client is not None:
                try:
                    client.stop_channels()
                except Exception:
                    pass

        stdout, stderr = _stop_and_collect(process)
        returncode = process.returncode

    stdout_tail = _tail(stdout)
    stderr_tail = _tail(stderr)
    if handshake_succeeded:
        return KernelProbeResult(
            ok=True,
            message="Jupyter kernel_info handshake succeeded",
            returncode=returncode,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
        )

    reason = (
        f"{type(probe_error).__name__}: {probe_error}"
        if probe_error is not None
        else f"kernel exited with code {returncode}"
    )
    details = stderr_tail or stdout_tail
    if details:
        reason = f"{reason}; kernel stderr: {_single_line_tail(details)}"
    return KernelProbeResult(
        ok=False,
        message=reason,
        returncode=returncode,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
    )


def _stop_and_collect(process: subprocess.Popen) -> tuple[str, str]:
    if process.poll() is None:
        process.terminate()
    try:
        return process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.communicate()


def _tail(value: str | None, *, limit: int = 4000) -> str:
    return "" if value is None else value[-limit:]


def _connection_key() -> bytes:
    return secrets.token_hex(32).encode("ascii")


def _single_line_tail(value: str, *, line_limit: int = 5) -> str:
    lines = [line.strip() for line in value.splitlines() if line.strip()][-line_limit:]
    return " | ".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify that a Python executable can answer Jupyter kernel_info."
    )
    parser.add_argument("--python", required=True, dest="python_executable")
    parser.add_argument("--cwd", default="")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args(argv)
    python_executable = Path(args.python_executable)
    if not python_executable.is_file():
        print(f"kernel Python executable does not exist: {python_executable}", file=sys.stderr)
        return 2
    result = probe_python_kernel(
        python_executable,
        timeout=args.timeout,
        cwd=Path(args.cwd) if args.cwd else None,
    )
    stream = sys.stdout if result.ok else sys.stderr
    print(result.message, file=stream)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
