"""OS-enforced network boundary for local, data-bearing code execution.

This confines the launched process tree's sockets, including native libraries
and exec descendants. It does not claim to be a machine-wide firewall or a
complete sandbox for hostile code, filesystem access, or delegated IPC services.
"""

from __future__ import annotations

from pathlib import Path
import sys


class LocalExecutionIsolationError(RuntimeError):
    pass


_LOOPBACK_PROFILE = """(version 1)
(allow default)
(deny network*)
(allow network-outbound (remote ip "localhost:*"))
(allow network-inbound (local ip "localhost:*"))
"""


def loopback_only_command(command: list[str]) -> list[str]:
    """Fail closed when the host cannot enforce the required socket boundary."""
    if not command or not all(isinstance(value, str) and value for value in command):
        raise LocalExecutionIsolationError("local execution command is invalid")
    sandbox = Path("/usr/bin/sandbox-exec")
    if sys.platform != "darwin" or not sandbox.is_file():
        raise LocalExecutionIsolationError(
            "此主机尚无已支持的 Notebook 进程网络隔离；未启动含业务数据的代码。"
        )
    return [str(sandbox), "-p", _LOOPBACK_PROFILE, *command]


def notebook_kernel_manager_class():
    # Keep Jupyter imports out of ordinary tool workers and CLI probes.
    from jupyter_client.manager import AsyncKernelManager
    from jupyter_client.provisioning.local_provisioner import LocalProvisioner

    class LoopbackKernelManager(AsyncKernelManager):
        async def _async_pre_start_kernel(self, **kwargs):
            # A remote/custom provisioner may never call format_kernel_cmd.
            # Choose the installed local implementation explicitly, before any
            # provisioner hook can run or receive notebook execution context.
            loopback_only_command([sys.executable])
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
            return await super()._async_pre_start_kernel(**kwargs)

        def format_kernel_cmd(self, extra_arguments=None):
            return loopback_only_command(super().format_kernel_cmd(extra_arguments))

    return LoopbackKernelManager
