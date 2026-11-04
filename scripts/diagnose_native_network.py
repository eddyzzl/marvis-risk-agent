"""Bounded native socket diagnostics for a failed macOS isolation CI lane.

Uses synthetic local listeners and documentation-only public IP ranges. It
does not change the policy, environment, or original pytest verdict. Output
contains only platform facts, explicit probe options, and probe subprocesses.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from marvis.local_execution import _LOOPBACK_PROFILE, loopback_only_command  # noqa: E402


def _run(command: list[str], timeout: float = 8) -> dict:
    started = time.monotonic()
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate(timeout=3)
    # The JVM can echo environment-supplied options before its own output.
    # Their presence is useful; their values can contain operator secrets.
    options_echo = r"(?m)^((?:NOTE: )?Picked up (?:JAVA_TOOL_OPTIONS|JDK_JAVA_OPTIONS|_JAVA_OPTIONS):).*$"
    stdout = re.sub(options_echo, r"\1 <redacted>", stdout)
    stderr = re.sub(options_echo, r"\1 <redacted>", stderr)
    return {
        "returncode": process.returncode, "timed_out": timed_out,
        "seconds": round(time.monotonic() - started, 3),
        "stdout": stdout[-8000:], "stderr": stderr[-8000:],
    }


def _local_probe(family: int, socktype: int, host: str, build_command) -> dict:
    stop = threading.Event()
    received = []
    with socket.socket(family, socktype) as server:
        server.bind((host, 0))
        server.settimeout(0.1)
        if socktype == socket.SOCK_STREAM:
            server.listen(1)

        def serve():
            while not stop.is_set():
                try:
                    if socktype == socket.SOCK_STREAM:
                        conn, _ = server.accept()
                        with conn:
                            conn.settimeout(2)
                            received.append(conn.recv(32).decode())
                            conn.sendall(b"local-ok")
                    else:
                        data, address = server.recvfrom(32)
                        received.append(data.decode())
                        server.sendto(b"local-ok", address)
                    return
                except TimeoutError:
                    continue
                except OSError:
                    return

        worker = threading.Thread(target=serve, daemon=True)
        worker.start()
        try:
            result = _run(loopback_only_command(build_command(server.getsockname()[1])))
        finally:
            stop.set()
            worker.join(timeout=3)
    return {**result, "listener_received": received,
            "loopback_ok": result["returncode"] == 0 and received == ["probe"]
            and "local-ok" in result["stdout"]}


_JAVA_SOURCE = """
import java.net.*;
class LocalNetworkDiagnostic {
    public static void main(String[] args) throws Exception {
        try (Socket remote = new Socket()) {
            remote.connect(new InetSocketAddress("203.0.113.1", 443), 1500);
            throw new AssertionError("external connect was allowed");
        } catch (SocketException ex) {
            String message = ex.getMessage().toLowerCase();
            if (!message.contains("permitted") && !message.contains("permission")) throw ex;
            System.out.println("java-external-denied: " + ex);
        }
        try (Socket local = new Socket("127.0.0.1", Integer.parseInt(args[0]))) {
            local.setSoTimeout(2000);
            local.getOutputStream().write("probe".getBytes());
            byte[] answer = local.getInputStream().readNBytes(8);
            if (!new String(answer).equals("local-ok")) throw new AssertionError("wrong local reply");
            System.out.println("local-ok");
        }
    }
}
"""


def diagnose() -> dict:
    report = {"platform": platform.platform(), "python": sys.version,
              "python_executable": sys.executable, "base_network_policy": _LOOPBACK_PROFILE,
              "policy_note": "Every probe uses loopback_only_command including launch-marker rules.",
              "scope": "Failure diagnostics only; does not replace or alter native-test verdicts.",
              "probes": []}
    if sys.platform != "darwin":
        report["unsupported"] = "This diagnostic requires macOS Seatbelt."
        return report

    def record(name, result):
        report["probes"].append({"name": name, **result})
        print(json.dumps(report["probes"][-1], ensure_ascii=False), flush=True)

    addresses = [
        ("ipv4", socket.AF_INET, "127.0.0.1", socket.AF_INET, "127.0.0.1"),
        ("ipv6", socket.AF_INET6, "::1", socket.AF_INET6, "::1"),
        ("ipv4_mapped", socket.AF_INET6, "::ffff:127.0.0.1", socket.AF_INET, "127.0.0.1"),
        ("localhost_name", socket.AF_INET, "localhost", socket.AF_INET, "127.0.0.1"),
    ]
    for name, family, host, listener_family, listener_host in addresses:
        for kind, socktype in (("tcp", socket.SOCK_STREAM), ("udp", socket.SOCK_DGRAM)):
            def command(port, family=family, host=host, socktype=socktype):
                source = (
                    f"import socket; s=socket.socket({int(family)}, {int(socktype)}); "
                    f"s.settimeout(2); s.connect(({host!r}, {port})); "
                    "s.sendall(b'probe'); print(s.recv(32).decode())"
                )
                return [sys.executable, "-c", source]
            record(f"native_{name}_{kind}", _local_probe(listener_family, socktype, listener_host, command))

    for family, host in ((socket.AF_INET, "203.0.113.1"),
                         (socket.AF_INET6, "2001:db8::1"),
                         (socket.AF_INET6, "::ffff:203.0.113.1")):
        for kind, socktype in (("tcp", socket.SOCK_STREAM), ("udp", socket.SOCK_DGRAM)):
            operation = f"s.connect(({host!r}, 443))" if kind == "tcp" else f"s.sendto(b'probe', ({host!r}, 53))"
            source = (
                f"import socket,json; s=socket.socket({int(family)}, {int(socktype)}); s.settimeout(2)\n"
                f"try:\n {operation}\nexcept OSError as exc:\n"
                f" print(json.dumps({{'denied': exc.errno in ({errno.EPERM}, {errno.EACCES}), "
                "'errno':exc.errno,'error':str(exc)}))\nelse:\n print('EXTERNAL_ALLOWED')\n"
            )
            record(f"external_{host}_{kind}", _run(loopback_only_command([sys.executable, "-c", source])))

    java, javac = shutil.which("java"), shutil.which("javac")
    report["java_executable"] = java
    if java and javac:
        report["java_version"] = _run([java, "-version"])
        with tempfile.TemporaryDirectory(prefix="marvis-native-diagnostic-") as tmp:
            source = Path(tmp) / "LocalNetworkDiagnostic.java"
            source.write_text(_JAVA_SOURCE)
            compiled = _run([javac, str(source)], timeout=15)
            report["javac"] = compiled
            if compiled["returncode"] == 0:
                # The launcher now supplies IPv4 compatibility for the JVM.
                # An explicit false is a diagnostic control for the original
                # macOS 26 IPv4-mapped socket failure, not an acceptance test.
                for name, options in (
                    ("launch_default", []),
                    ("forced_dual_stack", ["-Djava.net.preferIPv4Stack=false"]),
                    ("ipv4_stack", ["-Djava.net.preferIPv4Stack=true"]),
                ):
                    def command(port, options=options):
                        child = [java, *options, "-cp", tmp, "LocalNetworkDiagnostic", str(port)]
                        return [sys.executable, "-c", "import subprocess,sys; sys.exit(subprocess.call(" + repr(child) + "))"]
                    result = _local_probe(socket.AF_INET, socket.SOCK_STREAM, "127.0.0.1", command)
                    record(f"java_{name}", {"jvm_options": options, **result})
    else:
        report["java_unavailable"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
