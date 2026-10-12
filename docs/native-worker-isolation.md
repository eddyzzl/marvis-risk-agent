# Native worker isolation contract

`worker_isolation_mode` is an explicit execution-environment setting. Its default is `trusted_local`, preserving existing V1/V2 behavior. The additional `os_sandbox` mode governs **non-builtin, input-only compute workers** on the locally verified macOS backend. It does not place built-in platform tools, Notebook kernels, the API server or external host processes inside that boundary.

The settings API is the existing `/api/settings/execution-environment` GET/PUT surface. The dedicated execution-settings panel shows support, startup state, limitations and the next action. A save that requests an unavailable backend is rejected. Every subsequent non-builtin invocation reads the saved mode and performs an OS startup probe before launching code. Malformed or unknown worker settings fail closed, instead of selecting the compatibility mode.

## Supported strong boundary

The host launches the existing worker through a deny-default Seatbelt profile using the installed `/usr/bin/sandbox-exec`. The profile grants read-only access to the selected trusted interpreter/runtime libraries, MARVIS source, and the exact selected package or source file. Python import-root directory names and ancestor metadata remain visible. The worker receives separate empty workspace, dataset, home and temporary directories; none of the real workspace, SQLite database, administrator credential, private holdout or another task's material is mounted. The empty directories cannot be written by worker code.

Internet and loopback TCP/UDP, IPv4/IPv6, external Unix socket connections, Mach service lookup and named POSIX shared memory are denied. The only host communication is the inherited worker protocol pipes. Fork and process-spawn are denied by the OS. A same-PID native `exec` is permitted and retains the sandbox and resource limits. A worker cannot detach a daemon or grandchild from host monitoring because creation of such processes is refused. A declared `process:spawn` requirement is unsupported in this mode.

Resource controls retain the original host wall-clock timeout, cancellation, process cleanup and sampled RSS monitor, with finite RSS monitoring required. CPU and file-size limits become hard ceilings for this worker. Host-side native output collection is bounded to 8 MiB stdout and 1 MiB stderr; the complete job/input envelope is limited to 2 MiB. These byte budgets also cover raw native writes that bypass Python's output redirection. A killed or rejected worker returns a typed failure; it does not silently rerun in a less restrictive mode.

The worker independently checks that OS policy is active before it imports tool code. Host-issued execution evidence appears under the existing result/audit `resource_limits.worker_isolation` field. Worker output cannot change that evidence. The protocol version is 7, so an older worker cannot silently accept the new contract.

## Explicit unsupported needs and compatibility

Non-builtin tools declaring resource/file reads, persistent writes, network, LLM access or process-spawn permissions are refused before dispatch. The first supported profile computes only from the caller's submitted JSON and returns structured JSON. Dataset/artifact brokering and durable writes require a future exact actor/task/approval-bound mount contract; this mode does not grant a whole workspace or dataset directory to approximate that contract.

Built-in packs retain their existing trusted host filesystem and database capabilities, including their normal model/workflow requirements. Their execution evidence explicitly says `trusted_builtin_host` and `enforced=false`. An arbitrary-code model/Notebook loader inside a built-in workflow is consequently outside this strong compute boundary. Notebook network controls remain separate. Trusted-local mode remains available for existing trusted workflows and their declared permissions.

Linux and Windows have no verified backend here and are reported as unsupported. Local tests that cannot run a native platform probe remain skipped/unavailable; they are not acceptance evidence for that platform. The profile uses macOS system interfaces, so availability is checked on the actual host rather than inferred from a platform name.

This boundary protects the launched worker. It does not prevent a separate hostile process already running under the same host account from modifying control-plane files, trusted runtime installations or application memory. It also does not establish institutional authentication, distributed instance isolation or publisher identity outside the administrator's public-key vetting.

## Local verification

The native suite executes the application's actual ToolRunner and signed on-disk packages. It uses `ctypes`/libSystem calls to bypass Python file/network/process patches: native open/read/write against synthetic private material and real control-plane paths without reading credential bodies; IPv4/IPv6 TCP and UDP; Unix services; a host-created POSIX shared-memory object; a host-resolvable Mach service; fork/posix_spawn; same-PID exec; native allocation; raw stdout flooding; hard CPU termination; timeout/reap and preflight rejection. Separate tests verify the settings API, explicit unsupported platforms and declared-needs rejection. Real browser verification is tracked separately from these local API/native checks.
