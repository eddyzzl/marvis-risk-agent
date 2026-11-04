"""Explicit local execution modes and OS-enforced socket boundaries.

The loopback policy confines the launched process tree, including native code
and exec descendants. Filesystem access and delegated local/IPC services are
outside this boundary; this is not a complete sandbox for hostile code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import subprocess
import sys

TRUSTED_LOCAL = "trusted_local"
LOOPBACK_ONLY = "loopback_only"
NETWORK_MODES = (TRUSTED_LOCAL, LOOPBACK_ONLY)
# Not a mode: the settings file exists but could not be read, so the user's
# choice is unknown. Every execution path refuses it (fail closed) instead of
# silently assuming trusted_local.
SETTINGS_UNREADABLE = "settings_unreadable"
SETTINGS_UNREADABLE_MESSAGE = "执行环境设置文件存在但无法读取，Notebook 网络模式未知；为安全起见未启动 Notebook。"
SETTINGS_UNREADABLE_NEXT_ACTION = (
    "请在「设置 → 执行环境」中依次重新选择 Python 环境和 Notebook 网络模式，"
    "再在「第三方工具执行边界」点「确认执行边界」（三项都明确后才重写设置文件），"
    "或修复工作区 settings/execution_environment.json 后重试。"
)
# Credentials of the local API never enter a loopback Notebook tree.
KERNEL_WITHHELD_ENV = frozenset({"MARVIS_LOCAL_TOKEN"})
_SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
_LIMITATIONS = (
    "不隔离文件系统；本地代理与委托 IPC 服务可能转发数据；"
    "不是完整的不可信代码沙箱。"
)


class LocalExecutionIsolationError(RuntimeError):
    pass


@dataclass(frozen=True)
class LocalExecutionCapability:
    mode: str
    platform: str
    backend: str
    available: bool
    status: str
    network_restricted: bool
    filesystem_isolated: bool = False
    delegated_ipc_protected: bool = False
    startup_checked: bool = False
    message: str = ""
    limitations: str = _LIMITATIONS
    next_action: str = ""


_LOOPBACK_PROFILE = """(version 1)
(allow default)
(deny network*)
(allow network-outbound (remote ip "localhost:*"))
(allow network-inbound (local ip "localhost:*"))
(deny file-read* file-write* (regex #"/\\.blind_oot(/|$)"))
"""
# The last rule keeps sealed blind OOT custody (marvis.modeling_holdout) out of
# loopback Notebook trees. trusted_local processes are not covered.

# macOS 26 Seatbelt does not match IPv4-mapped IPv6 against localhost.
# The JVM uses those sockets even for 127.0.0.1 unless IPv4 is requested.
# Set this inside the confined process, then replace it with the real command:
# all descendants inherit the compatibility option, while user option values
# stay out of argv and the host environment is untouched. The OS policy above
# still enforces the boundary; this option does not grant network access.
_LOOPBACK_EXEC_BOOTSTRAP = """
import os, sys
option = "-Djava.net.preferIPv4Stack=true"
existing = os.environ.get("JAVA_TOOL_OPTIONS", "")
if existing.split()[-1:] != [option]:
    os.environ["JAVA_TOOL_OPTIONS"] = existing + " " + option if existing else option
os.execvpe(sys.argv[1], sys.argv[1:], os.environ)
"""


def validate_network_mode(mode: str) -> str:
    if mode == SETTINGS_UNREADABLE:
        raise LocalExecutionIsolationError(f"{SETTINGS_UNREADABLE_MESSAGE} {SETTINGS_UNREADABLE_NEXT_ACTION}")
    if mode not in NETWORK_MODES:
        raise LocalExecutionIsolationError("Notebook 网络模式无效，请选择可信本地运行或仅本机网络。")
    return mode


def execution_capability(mode: str = TRUSTED_LOCAL, *, probe: bool = False) -> LocalExecutionCapability:
    """Describe support; optional startup probe contains no notebook or data."""
    validate_network_mode(mode)
    if mode == TRUSTED_LOCAL:
        return LocalExecutionCapability(
            mode=mode, platform=sys.platform, backend="none", available=True,
            status="trusted", network_restricted=False,
            message="可信本地运行：保留既有 Notebook 契约，不限制网络；仅执行可信材料。",
        )
    backend = "macos_seatbelt" if sys.platform == "darwin" else "unavailable"
    common = dict(mode=mode, platform=sys.platform, backend=backend, network_restricted=False)
    if sys.platform != "darwin":
        return LocalExecutionCapability(
            **common, available=False, status="unsupported",
            message="此系统尚无经过原生验证的仅本机网络 backend；未启动 Notebook。",
            next_action="使用已验证的 macOS 环境，或在确认材料可信后明确选择可信本地运行。",
        )
    if not _SANDBOX_EXEC.is_file() or not os.access(_SANDBOX_EXEC, os.X_OK):
        return LocalExecutionCapability(
            **common, available=False, status="missing_tool",
            message="本机缺少可执行的 sandbox-exec；未启动 Notebook。",
            next_action="检查 macOS 隔离工具，或在确认材料可信后明确选择可信本地运行。",
        )
    if probe:
        try:
            completed = subprocess.run(
                [str(_SANDBOX_EXEC), "-p", _LOOPBACK_PROFILE + _inherited_rules(), "/usr/bin/true"],
                capture_output=True, timeout=5, check=False,
            )
            if completed.returncode != 0:
                raise LocalExecutionIsolationError("startup probe failed")
        except (OSError, subprocess.SubprocessError, LocalExecutionIsolationError):
            return LocalExecutionCapability(
                **common, available=False, status="startup_failed", startup_checked=True,
                message="本机网络策略启动预检失败；未启动 Notebook。",
                next_action="检查系统是否允许 sandbox-exec 启动；修复后重新验证执行环境。",
            )
    return LocalExecutionCapability(
        mode=mode, platform=sys.platform, backend=backend, available=True,
        status="available", network_restricted=True, startup_checked=probe,
        message="仅本机网络：Notebook 进程树允许 loopback，拒绝其他 socket 网络。",
        next_action="Notebook 的网络依赖应预先准备为本地材料；启动预检不替代原生网络验收。",
    )


def _inherited_rules() -> str:
    # A probe nested in a marked launch must keep that launch's rules, or
    # macOS refuses the nested sandbox_apply.
    from marvis.local_peer import inherited_marker_rules

    return inherited_marker_rules()


def preflight_execution(mode: str = TRUSTED_LOCAL) -> dict:
    capability = execution_capability(mode, probe=True)
    if not capability.available:
        raise LocalExecutionIsolationError(f"{capability.message} {capability.next_action}")
    return asdict(capability)


def execution_command(command: list[str], mode: str = TRUSTED_LOCAL) -> list[str]:
    validate_network_mode(mode)
    if not command or not all(isinstance(value, str) and value for value in command):
        raise LocalExecutionIsolationError("local execution command is invalid")
    if mode == TRUSTED_LOCAL:
        return list(command)
    capability = execution_capability(mode)
    if not capability.available:
        raise LocalExecutionIsolationError(f"{capability.message} {capability.next_action}")
    from marvis.local_peer import launch_marker_rules

    # Each launch carries its own marker rule. The sandbox (and so the
    # marker) is inherited by every descendant and cannot be dropped, which
    # lets the API recognise this launch's processes (marvis.local_peer).
    return [
        str(_SANDBOX_EXEC), "-p", _LOOPBACK_PROFILE + launch_marker_rules(),
        # The bootstrap needs only stdlib; ignore host Python configuration and
        # site hooks without changing the environment inherited by the target.
        sys.executable, "-I", "-S", "-c", _LOOPBACK_EXEC_BOOTSTRAP, *command,
    ]


def loopback_only_command(command: list[str]) -> list[str]:
    return execution_command(command, LOOPBACK_ONLY)


def notebook_kernel_manager_class(mode: str = LOOPBACK_ONLY):
    from jupyter_client.manager import AsyncKernelManager
    from jupyter_client.provisioning.local_provisioner import LocalProvisioner

    validate_network_mode(mode)
    if mode == TRUSTED_LOCAL:
        return AsyncKernelManager

    class LoopbackKernelManager(AsyncKernelManager):
        async def _async_pre_start_kernel(self, **kwargs):
            # Reject custom/remote provisioners before their hooks run or get
            # notebook execution context. Always use the installed local class.
            preflight_execution(mode)
            config = self.kernel_spec.metadata.get("kernel_provisioner", {})
            if (not isinstance(config, dict)
                    or config.get("provisioner_name", "local-provisioner") != "local-provisioner"
                    or config.get("config")):
                raise LocalExecutionIsolationError("Notebook 仅支持受网络隔离的本地 kernel。")
            if self.provisioner is None:
                import uuid

                self.kernel_id = self.kernel_id or kwargs.pop("kernel_id", str(uuid.uuid4()))
                self.provisioner = LocalProvisioner(
                    kernel_id=self.kernel_id, kernel_spec=self.kernel_spec, parent=self,
                )
            elif type(self.provisioner) is not LocalProvisioner:
                raise LocalExecutionIsolationError("Notebook kernel provisioner 未通过本地隔离校验。")
            kwargs["env"] = {key: value for key, value in (kwargs.get("env") or os.environ).items()
                             if key not in KERNEL_WITHHELD_ENV}
            return await super()._async_pre_start_kernel(**kwargs)

        def format_kernel_cmd(self, extra_arguments=None):
            return execution_command(super().format_kernel_cmd(extra_arguments), mode)

    return LoopbackKernelManager
