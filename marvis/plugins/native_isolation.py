"""Fail-closed OS boundary for non-builtin, input-only compute workers.

This is a local macOS Seatbelt backend, not an institutional identity boundary.
Trusted platform packs retain their existing host filesystem/DB capabilities.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

TRUSTED_LOCAL = "trusted_local"
OS_SANDBOX = "os_sandbox"
MODES = (TRUSTED_LOCAL, OS_SANDBOX)
_SANDBOX = Path("/usr/bin/sandbox-exec")


class NativeIsolationError(RuntimeError):
    def __init__(self, code, *, capability=None):
        super().__init__(code)
        self.code = code
        self.capability = (capability or {}) | {
            "enforced": False,
            "execution_boundary": "not_started",
        }


@dataclass(frozen=True)
class WorkerCapability:
    mode: str
    platform: str
    backend: str
    available: bool
    status: str
    startup_checked: bool = False
    nonbuiltin_compute_only: bool = False
    builtin_boundary: str = "trusted_host"
    network_allowed: bool = True
    host_workspace_readable: bool = True
    ipc_allowed: bool = True
    fork_allowed: bool = True
    limitations: tuple[str, ...] = ()


def _payload(value):
    result = asdict(value)
    result["limitations"] = list(result["limitations"])
    return result


def configured_mode(workspace):
    from marvis.execution_environment import _settings_path

    path = _settings_path(Path(workspace))
    if not path.exists():
        return TRUSTED_LOCAL
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        mode = payload.get("worker_isolation_mode", TRUSTED_LOCAL)
    except (ValueError, OSError, AttributeError) as exc:
        raise NativeIsolationError("worker_isolation_settings_unreadable") from exc
    if mode not in MODES:
        raise NativeIsolationError("worker_isolation_mode_invalid")
    return mode


def capability(mode=TRUSTED_LOCAL, *, probe=False, python_executable=None):
    if mode not in MODES:
        raise NativeIsolationError("worker_isolation_mode_invalid")
    if mode == TRUSTED_LOCAL:
        return _payload(WorkerCapability(mode, sys.platform, "none", True, "trusted"))
    common = dict(
        mode=mode,
        platform=sys.platform,
        backend="macos_seatbelt",
        available=False,
        status="unsupported",
        nonbuiltin_compute_only=True,
        network_allowed=False,
        host_workspace_readable=False,
        ipc_allowed=False,
        fork_allowed=False,
        limitations=(
            "Only non-builtin input-only compute is supported; declared resource, file, network and process-spawn needs are refused.",
            "Builtin platform tools retain trusted host filesystem and database access.",
            "Runtime libraries and platform source are trusted read-only dependencies; directory metadata and root-directory names are visible.",
            "No Linux/Windows backend or hostile host-side same-UID process isolation is claimed.",
        ),
    )
    if sys.platform != "darwin":
        common["backend"] = "unavailable"
        return _payload(WorkerCapability(**common))
    if not _SANDBOX.is_file() or not os.access(_SANDBOX, os.X_OK):
        common["status"] = "missing_backend"
        return _payload(WorkerCapability(**common))
    common.update(available=True, status="available")
    if probe:
        common["startup_checked"] = True
        try:
            with tempfile.TemporaryDirectory(prefix="marvis-worker-preflight-") as tmp:
                root = Path(tmp).resolve()
                runtime = _runtime(python_executable or sys.executable)
                profile = _profile(runtime, root, (), ())
                script = (
                    "import ctypes; c=ctypes.CDLL(None,use_errno=True); "
                    "assert c.fork()==-1 and ctypes.get_errno()==1; "
                    "s=c.socket(2,1,0); a=ctypes.create_string_buffer(bytes([16,2,0,9,127,0,0,1])+bytes(8)); "
                    "assert c.connect(s,a,16)==-1 and ctypes.get_errno()==1; "
                    "c.close(s); print('boundary-ready')"
                )
                completed = subprocess.run(
                    [
                        str(_SANDBOX),
                        "-p",
                        profile,
                        runtime["python"],
                        "-I",
                        "-B",
                        "-c",
                        script,
                    ],
                    cwd=root,
                    env=_environment(root, runtime, None),
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if completed.returncode or completed.stdout.strip() != "boundary-ready":
                    raise NativeIsolationError("worker_isolation_preflight_failed")
        except (OSError, subprocess.SubprocessError, NativeIsolationError):
            common.update(available=False, status="startup_failed")
    return _payload(WorkerCapability(**common))


def _runtime(python_executable):
    python = Path(python_executable).resolve(strict=True)
    if not python.is_file() or not os.access(python, os.X_OK):
        raise NativeIsolationError("worker_isolation_interpreter_invalid")
    script = "import json,sys; print(json.dumps({'prefix':sys.prefix,'base_prefix':sys.base_prefix,'python':sys.executable}))"
    completed = subprocess.run(
        [str(python), "-I", "-B", "-c", script],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    try:
        result = json.loads(completed.stdout)
        roots = tuple(
            dict.fromkeys(
                str(Path(result[key]).resolve(strict=True))
                for key in ("prefix", "base_prefix")
            )
        )
        if any(path == "/" for path in roots):
            raise ValueError("unbounded interpreter")
        return {"python": str(python), "roots": roots}
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise NativeIsolationError("worker_isolation_interpreter_invalid") from exc


def _profile(runtime, scratch, package_roots, files):
    read_roots = [
        *runtime["roots"],
        "/System/Library",
        "/usr/lib",
        "/usr/bin",
        "/bin",
        "/usr/share",
        str(scratch),
        *package_roots,
    ]
    profile = [
        "(version 1)",
        "(deny default)",
        "(allow process-exec)",
        "(allow sysctl-read)",
        '(allow file-read-data (literal "/"))',
        '(allow file-read* file-write* (literal "/dev/null"))',
        '(allow file-read* (literal "/dev/urandom") (literal "/dev/random"))',
    ]
    # No import of Apple private system profiles: they add implicit IPC rights.
    for value in read_roots:
        profile.append("(allow file-read* (subpath " + json.dumps(str(value)) + "))")
    # Python's FileFinder lists its import-root directory before reading the
    # selected package. This permits directory names, never sibling contents.
    for value in package_roots:
        profile.append(
            "(allow file-read-data (literal "
            + json.dumps(str(Path(value).parent))
            + "))"
        )
    for value in files:
        profile.append("(allow file-read* (literal " + json.dumps(str(value)) + "))")
    for value in [runtime["python"], *read_roots, *files]:
        for parent in Path(value).parents:
            profile.append(
                "(allow file-read-metadata (literal " + json.dumps(str(parent)) + "))"
            )
    return "\n".join(profile)


def _environment(root, runtime, project_root):
    env = {
        "LANG": "en_US.UTF-8",
        "LC_CTYPE": "UTF-8",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "HOME": str(root / "home"),
        "TMPDIR": str(root / "tmp"),
        "TMP": str(root / "tmp"),
        "TEMP": str(root / "tmp"),
        "PATH": str(Path(runtime["python"]).parent) + ":/usr/bin:/bin",
    }
    if project_root is not None:
        env["PYTHONPATH"] = str(project_root)
    return env


def verify_worker_boundary(job):
    if job.get("worker_isolation_mode", TRUSTED_LOCAL) != OS_SANDBOX or job.get(
        "builtin"
    ):
        return
    if sys.platform != "darwin":
        raise NativeIsolationError("worker_os_boundary_not_active")
    import ctypes

    sandbox = ctypes.CDLL("/usr/lib/libsandbox.dylib")
    for operation in (
        b"process-fork",
        b"network-outbound",
        b"mach-lookup",
        b"file-write-data",
    ):
        if sandbox.sandbox_check(os.getpid(), operation, 0) != 1:
            raise NativeIsolationError("worker_os_boundary_not_active")


@dataclass(frozen=True)
class Launch:
    args: list[str]
    job: dict
    env: dict
    cwd: str | None
    evidence: dict


@contextmanager
def prepare_launch(python_executable, job, *, env, rss_limit_mb):
    mode = job.get("worker_isolation_mode", TRUSTED_LOCAL)
    current = capability(mode)
    if not current["available"]:
        raise NativeIsolationError("worker_isolation_unavailable", capability=current)
    args = [python_executable, "-m", "marvis.plugins.subprocess_worker"]
    if mode == TRUSTED_LOCAL or job.get("builtin") is True:
        evidence = current | {
            "enforced": False,
            "execution_boundary": "trusted_builtin_host"
            if job.get("builtin")
            else "trusted_local",
        }
        if job.get("builtin"):
            evidence.update(
                backend="none",
                status="trusted_host",
                network_allowed=True,
                host_workspace_readable=True,
                ipc_allowed=True,
                fork_allowed=True,
            )
        yield Launch(args, job, env, None, evidence)
        return
    needs = tuple(job.get("isolation_needs") or job.get("side_effects") or ())
    if needs:
        raise NativeIsolationError(
            "worker_isolation_needs_not_supported",
            capability=current | {"refused_needs": list(needs)},
        )
    if rss_limit_mb is None or int(rss_limit_mb) <= 0:
        raise NativeIsolationError(
            "worker_isolation_rss_monitor_required", capability=current
        )
    import marvis.resource_monitor as resources

    if resources.psutil is None:
        raise NativeIsolationError(
            "worker_isolation_rss_monitor_unavailable", capability=current
        )
    current = capability(mode, probe=True, python_executable=python_executable)
    if not current["available"]:
        raise NativeIsolationError(
            "worker_isolation_preflight_failed", capability=current
        )
    runtime = _runtime(python_executable)
    workspace = Path(job["workspace"]).resolve()
    project_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="marvis-native-worker-") as temporary:
        root = Path(temporary).resolve()
        for name in ("workspace", "datasets", "home", "tmp"):
            (root / name).mkdir()
        packages = [str(project_root / "marvis")]
        files = []
        if job.get("module_path"):
            file = Path(job["module_path"]).resolve(strict=True)
            if ".blind_oot" in file.parts or not file.is_file() or file.suffix != ".py":
                raise NativeIsolationError(
                    "worker_isolation_source_invalid", capability=current
                )
            files.append(str(file))
        else:
            package_name = str(job.get("module", "")).split(".")[0]
            candidates = [
                (Path(value) / package_name).resolve()
                for value in job.get("plugin_paths", ())
            ]
            matches = [path for path in candidates if path.is_dir()]
            if len(matches) != 1 or ".blind_oot" in matches[0].parts:
                raise NativeIsolationError(
                    "worker_isolation_package_invalid", capability=current
                )
            packages.append(str(matches[0]))
        # Trusted dependencies cannot accidentally grant the live control plane.
        if any(
            workspace == Path(value) or workspace.is_relative_to(Path(value))
            for value in [*runtime["roots"], *packages]
        ):
            raise NativeIsolationError(
                "worker_isolation_mount_overlaps_workspace", capability=current
            )
        sealed = dict(job)
        sealed.update(
            workspace=str(root / "workspace"),
            datasets_root=str(root / "datasets"),
            worker_isolation_mode=OS_SANDBOX,
        )
        sealed.pop("progress_path", None)
        evidence = current | {
            "enforced": True,
            "execution_boundary": "input_only_compute",
            "scratch_write_allowed": False,
        }
        sealed["worker_isolation_evidence"] = evidence
        try:
            if (
                len(json.dumps(sealed, ensure_ascii=False, allow_nan=False).encode())
                > 2 * 1024 * 1024
            ):
                raise ValueError("input budget")
        except (ValueError, TypeError) as exc:
            raise NativeIsolationError(
                "worker_isolation_input_budget_or_shape", capability=current
            ) from exc
        profile = _profile(runtime, root, packages, files)
        yield Launch(
            [
                str(_SANDBOX),
                "-p",
                profile,
                runtime["python"],
                "-B",
                "-m",
                "marvis.plugins.subprocess_worker",
            ],
            sealed,
            _environment(root, runtime, project_root),
            str(root),
            evidence,
        )
