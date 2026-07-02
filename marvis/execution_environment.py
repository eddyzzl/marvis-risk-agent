from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import subprocess
import sys

from marvis.files import write_json_atomic


logger = logging.getLogger(__name__)

SETTINGS_FILE = "execution_environment.json"


@dataclass(frozen=True)
class ExecutionEnvironmentSettings:
    execution_mode: str = "jupyter_kernel"
    kernel_name: str = "python3"
    conda_env_name: str = ""
    python_executable: str = ""


@dataclass(frozen=True)
class ExecutionEnvironmentValidation:
    ok: bool
    message: str
    kernel_name: str
    python_version: str = ""


@dataclass(frozen=True)
class ExecutionEnvironmentOption:
    id: str
    label: str
    execution_mode: str
    kernel_name: str
    conda_env_name: str = ""
    python_executable: str = ""
    python_version: str = ""
    source: str = ""
    available: bool = True
    note: str = ""


def load_execution_environment(workspace: Path) -> ExecutionEnvironmentSettings:
    path = _settings_path(workspace)
    if not path.exists():
        return ExecutionEnvironmentSettings()
    try:
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            raise json.JSONDecodeError("empty file", raw, 0)
        payload = json.loads(raw)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        logger.warning(
            "execution_environment: failed to read %s (%s); falling back to defaults",
            path,
            exc,
        )
        return ExecutionEnvironmentSettings()
    return ExecutionEnvironmentSettings(
        execution_mode=str(payload.get("execution_mode") or "jupyter_kernel"),
        kernel_name=str(payload.get("kernel_name") or "python3"),
        conda_env_name=str(payload.get("conda_env_name") or ""),
        python_executable=str(payload.get("python_executable") or ""),
    )


def save_execution_environment(
    workspace: Path,
    settings: ExecutionEnvironmentSettings,
) -> ExecutionEnvironmentSettings:
    path = _settings_path(workspace)
    write_json_atomic(path, asdict(settings), ensure_ascii=False, indent=2)
    return settings


def available_kernel_names() -> list[str]:
    return sorted(available_kernel_specs())


def available_kernel_specs() -> dict[str, dict]:
    try:
        from jupyter_client.kernelspec import KernelSpecManager

        specs = KernelSpecManager().get_all_specs()
        return {
            name: {
                "display_name": (payload.get("spec") or {}).get("display_name")
                or name,
                "argv": list((payload.get("spec") or {}).get("argv") or []),
            }
            for name, payload in sorted(specs.items())
        }
    except Exception:
        return {}


def detect_execution_environment_options() -> list[ExecutionEnvironmentOption]:
    kernel_specs = available_kernel_specs()
    kernel_by_python = _kernel_by_python_path(kernel_specs)
    options: list[ExecutionEnvironmentOption] = []
    seen_ids: set[str] = set()

    for kernel_name, spec in kernel_specs.items():
        option = _kernel_option(kernel_name, spec)
        options.append(option)
        seen_ids.add(option.id)

    for env_path in _conda_environment_paths():
        option = _conda_environment_option(env_path, kernel_by_python, kernel_specs)
        if option and option.id not in seen_ids:
            options.append(option)
            seen_ids.add(option.id)

    current = _current_python_option(kernel_by_python, kernel_specs)
    if current and current.id not in seen_ids:
        options.append(current)

    return options


def validate_execution_environment(
    settings: ExecutionEnvironmentSettings,
) -> ExecutionEnvironmentValidation:
    if settings.execution_mode == "jupyter_kernel":
        kernels = available_kernel_names()
        if settings.kernel_name not in kernels:
            return ExecutionEnvironmentValidation(
                ok=False,
                message=(
                    f"Jupyter kernel {settings.kernel_name!r} is not available; "
                    f"available kernels: {', '.join(kernels) or 'none'}"
                ),
                kernel_name=settings.kernel_name,
            )
        return ExecutionEnvironmentValidation(
            ok=True,
            message=f"Jupyter kernel {settings.kernel_name!r} is available",
            kernel_name=settings.kernel_name,
        )
    if settings.execution_mode == "conda_env":
        if not settings.conda_env_name:
            return ExecutionEnvironmentValidation(
                ok=False,
                message="conda_env_name is required for conda_env execution mode",
                kernel_name=settings.kernel_name,
            )
        validation = _run_version_check(
            ["conda", "run", "-n", settings.conda_env_name, "python", "-V"],
            kernel_name=settings.kernel_name or settings.conda_env_name,
        )
        return _with_available_kernel_check(validation)
    if settings.execution_mode == "python_executable":
        if not settings.python_executable:
            return ExecutionEnvironmentValidation(
                ok=False,
                message="python_executable is required for python_executable execution mode",
                kernel_name=settings.kernel_name,
            )
        validation = _run_version_check(
            [settings.python_executable, "-V"],
            kernel_name=settings.kernel_name,
        )
        return _with_available_kernel_check(validation)
    return ExecutionEnvironmentValidation(
        ok=False,
        message=f"unsupported execution mode: {settings.execution_mode}",
        kernel_name=settings.kernel_name,
    )


def _run_version_check(
    command: list[str],
    *,
    kernel_name: str,
) -> ExecutionEnvironmentValidation:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        return ExecutionEnvironmentValidation(
            ok=False,
            message=f"environment validation failed: {type(exc).__name__}: {exc}",
            kernel_name=kernel_name,
        )
    version = (completed.stdout or completed.stderr).strip()
    return ExecutionEnvironmentValidation(
        ok=True,
        message=version,
        kernel_name=kernel_name,
        python_version=version,
    )


def _with_available_kernel_check(
    validation: ExecutionEnvironmentValidation,
) -> ExecutionEnvironmentValidation:
    if not validation.ok:
        return validation
    kernels = available_kernel_names()
    if validation.kernel_name not in kernels:
        return ExecutionEnvironmentValidation(
            ok=False,
            message=(
                f"{validation.message}; Jupyter kernel {validation.kernel_name!r} "
                f"is not available; available kernels: {', '.join(kernels) or 'none'}"
            ),
            kernel_name=validation.kernel_name,
            python_version=validation.python_version,
        )
    return validation


def _kernel_option(kernel_name: str, spec: dict) -> ExecutionEnvironmentOption:
    display_name = str(spec.get("display_name") or kernel_name)
    python_path = _kernel_python_path(spec)
    if display_name == kernel_name:
        label = f"Jupyter Kernel · {kernel_name}"
    else:
        label = f"Jupyter Kernel · {display_name} ({kernel_name})"
    return ExecutionEnvironmentOption(
        id=f"kernel:{kernel_name}",
        label=label,
        execution_mode="jupyter_kernel",
        kernel_name=kernel_name,
        python_executable=python_path,
        source="jupyter",
        note=python_path,
    )


def _conda_environment_option(
    env_path: Path,
    kernel_by_python: dict[str, str],
    kernel_specs: dict[str, dict],
) -> ExecutionEnvironmentOption | None:
    python_path = _python_path_for_environment(env_path)
    if python_path is None:
        return None
    env_name = _conda_environment_name(env_path)
    resolved_python = _safe_resolve(python_path)
    kernel_name = kernel_by_python.get(resolved_python)
    if not kernel_name and env_name in kernel_specs:
        kernel_name = env_name
    available = bool(kernel_name)
    note = (
        f"Kernel: {kernel_name}"
        if available
        else "缺少匹配的 Jupyter Kernel，需在该环境安装并注册 ipykernel。"
    )
    return ExecutionEnvironmentOption(
        id=f"conda:{env_name}:{resolved_python}",
        label=f"Conda · {env_name}",
        execution_mode="conda_env",
        kernel_name=kernel_name or "",
        conda_env_name=env_name,
        python_executable=str(python_path),
        source="conda",
        available=available,
        note=note,
    )


def _current_python_option(
    kernel_by_python: dict[str, str],
    kernel_specs: dict[str, dict],
) -> ExecutionEnvironmentOption | None:
    current_python = Path(sys.executable)
    if not current_python.is_file():
        return None
    resolved_python = _safe_resolve(current_python)
    kernel_name = kernel_by_python.get(resolved_python)
    if not kernel_name and "python3" in kernel_specs:
        kernel_name = "python3"
    available = bool(kernel_name)
    return ExecutionEnvironmentOption(
        id=f"python:{resolved_python}",
        label=f"当前 Python · {current_python.name}",
        execution_mode="python_executable",
        kernel_name=kernel_name or "",
        python_executable=str(current_python),
        source="current",
        available=available,
        note=(
            f"Kernel: {kernel_name}"
            if available
            else "缺少可用于 Notebook 的 Jupyter Kernel。"
        ),
    )


def _kernel_by_python_path(kernel_specs: dict[str, dict]) -> dict[str, str]:
    result: dict[str, str] = {}
    for kernel_name, spec in kernel_specs.items():
        python_path = _kernel_python_path(spec)
        if not python_path:
            continue
        path = Path(python_path).expanduser()
        if not path.is_absolute():
            continue
        result.setdefault(_safe_resolve(path), kernel_name)
    return result


def _kernel_python_path(spec: dict) -> str:
    argv = spec.get("argv") or []
    if not argv:
        return ""
    return str(argv[0])


def _conda_environment_paths() -> list[Path]:
    try:
        completed = subprocess.run(
            ["conda", "env", "list", "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return []
    return [Path(path) for path in payload.get("envs", []) if path]


def _python_path_for_environment(env_path: Path) -> Path | None:
    if sys.platform.startswith("win"):
        candidates = [env_path / "python.exe"]
    else:
        candidates = [env_path / "bin" / "python", env_path / "python"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _conda_environment_name(env_path: Path) -> str:
    if env_path.parent.name == "envs":
        return env_path.name
    return "base"


def _safe_resolve(path: Path) -> str:
    try:
        return str(path.expanduser().resolve())
    except OSError:
        return str(path.expanduser())


def _settings_path(workspace: Path) -> Path:
    return Path(workspace) / "settings" / SETTINGS_FILE
