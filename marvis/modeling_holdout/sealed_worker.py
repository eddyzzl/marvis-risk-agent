"""Read the sealed holdout and score the frozen model inside an OS-isolated process.

The API process never decodes sealed rows at evaluation time when the macOS
Seatbelt backend is available. The worker may read only the interpreter, the
platform source, the sealed original (one literal path) and the pinned frozen
model files; it cannot write files, open sockets, use IPC or fork. Metrics are
computed by the same deterministic platform functions as before.
"""

from __future__ import annotations

import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile

import numpy as np

from marvis.modeling_holdout.contracts import HoldoutError

ISOLATED_WORKER = 'isolated_worker'
API_PROCESS = 'api_process'
_TIMEOUT_SECONDS = 900
_READY_TIMEOUT_SECONDS = 300


def sealed_metrics(frame, contract, classes, scorer):
    """Deterministic platform metrics; unchanged by isolation or fact gating."""
    target = frame[contract['target_col']].to_numpy()
    if len(target) < 2 or frame[contract['target_col']].isna().any():
        raise HoldoutError('holdout_labels_invalid')
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, log_loss, roc_auc_score
    target_type = contract['target_type']
    if target_type == 'multiclass':
        probabilities = scorer.predict_proba(frame)
        return {'rows': len(frame), 'logloss': float(log_loss(target, probabilities, labels=classes)),
                'macro_auc': float(roc_auc_score(target, probabilities, multi_class='ovr', average='macro', labels=classes))}
    scores = np.asarray(scorer.score(frame), dtype=float)
    if not np.isfinite(scores).all():
        raise HoldoutError('holdout_scores_invalid')
    if target_type == 'continuous':
        return {'rows': len(frame), 'rmse': float(np.sqrt(mean_squared_error(target, scores))),
                'mae': float(mean_absolute_error(target, scores)), 'r2': float(r2_score(target, scores))}
    if not set(target) == {0, 1}:
        raise HoldoutError('holdout_binary_labels_invalid')
    from marvis.feature.metrics import feature_ks, feature_auc
    return {'rows': len(frame), 'ks': float(feature_ks(scores, target)), 'auc': float(feature_auc(scores, target))}


def _scorer(artifact_data, runtime_dir):
    from marvis.packs.modeling.contracts import ModelArtifact
    from marvis.packs.modeling.scoring import _ModelArtifactScorer
    data = dict(artifact_data)
    for name in ('feature_list', 'feature_importance', 'scorecard_table'):
        data[name] = tuple(data[name])
    return _ModelArtifactScorer(ModelArtifact(**data), base_dir=Path(runtime_dir), replay_preprocessing=True)


def available_boundary():
    """Boundary evaluation will use on this host; probes without sealed data."""
    from marvis.plugins.native_isolation import OS_SANDBOX, NativeIsolationError, capability
    try:
        current = capability(OS_SANDBOX, probe=True, python_executable=sys.executable)
    except (NativeIsolationError, OSError, subprocess.SubprocessError):
        return API_PROCESS
    return ISOLATED_WORKER if current['available'] else API_PROCESS


class IsolatedSession:
    """Sandboxed worker prepared before the read budget is consumed.

    The worker imports its libraries and loads the frozen scorer, then
    reports ready on a dedicated pipe and waits. It reads sealed rows only
    after ``score()`` sends ``GO``; any failure before that is a pre-read
    failure and leaves the budget intact. If the parent exits instead, the
    worker sees EOF and quits without reading.
    """

    def __init__(self, sealed_path, sealed_hash, runtime_dir, artifact_data, contract):
        from marvis.plugins import native_isolation as isolation
        runtime = isolation._runtime(sys.executable)
        project_root = Path(__file__).resolve().parents[2]
        runtime_dir = Path(runtime_dir).resolve()
        files = [str(Path(sealed_path).resolve(strict=True))]
        files += sorted(str(path) for path in runtime_dir.rglob('*') if path.is_file())
        self._temporary = tempfile.TemporaryDirectory(prefix='marvis-sealed-eval-')
        self.process = self._reader = None
        try:
            root = Path(self._temporary.name).resolve()
            for name in ('home', 'tmp'):
                (root / name).mkdir()
            profile = isolation._profile(runtime, root, [str(project_root / 'marvis')], files)
            read_fd, write_fd = os.pipe()
            self._reader = os.fdopen(read_fd, 'rb')
            try:
                self.process = subprocess.Popen(
                    [str(isolation._SANDBOX), '-p', profile, runtime['python'], '-B', '-m', 'marvis.modeling_holdout.sealed_worker'],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, pass_fds=(write_fd,),
                    cwd=root, env=isolation._environment(root, runtime, project_root),
                )
            finally:
                os.close(write_fd)
            job = {'sealed_path': files[0], 'sealed_hash': sealed_hash, 'runtime_dir': str(runtime_dir),
                   'artifact': artifact_data, 'contract': contract, 'result_fd': write_fd}
            self.process.stdin.write((json.dumps(job) + '\n').encode())
            self.process.stdin.flush()
            message = self._message(_READY_TIMEOUT_SECONDS)
            if message != {'stage': 'ready'}:
                raise HoldoutError(str((message or {}).get('code') or 'holdout_isolated_worker_not_ready'))
        except HoldoutError:
            self.close()
            raise
        except (OSError, ValueError) as exc:
            self.close()
            raise HoldoutError('holdout_isolated_worker_not_ready') from exc

    def _message(self, timeout):
        if not select.select([self._reader], [], [], timeout)[0]:
            return None
        line = self._reader.readline()
        try:
            return json.loads(line) if line else None
        except ValueError:
            return None

    def score(self):
        try:
            self.process.stdin.write(b'GO\n')
            self.process.stdin.close()
            message = self._message(_TIMEOUT_SECONDS)
            if message is None:
                raise HoldoutError('holdout_isolated_evaluation_timeout')
            # Exactly one result, then a clean exit; a second line or a
            # non-zero exit invalidates the reply.
            extra = self._reader.read()
            returncode = self.process.wait(timeout=30)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            raise HoldoutError('holdout_isolated_evaluation_failed') from exc
        if message.get('stage') != 'result' or extra.strip() or returncode != 0:
            raise HoldoutError('holdout_isolated_evaluation_failed')
        if not message.get('ok'):
            raise HoldoutError(str(message.get('code') or 'holdout_isolated_evaluation_failed'))
        return message['metrics']

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        if self.process is not None and self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        if self._reader is not None:
            self._reader.close()
        self._temporary.cleanup()


def _worker_boundary_active():
    import ctypes
    sandbox = ctypes.CDLL('/usr/lib/libsandbox.dylib')
    return all(sandbox.sandbox_check(os.getpid(), operation, 0) == 1
               for operation in (b'process-fork', b'network-outbound', b'file-write-data'))


def _send(stream, message):
    stream.write(json.dumps(message, allow_nan=False) + '\n')
    stream.flush()


def main():
    job = json.loads(sys.stdin.readline())
    out = os.fdopen(job['result_fd'], 'w')
    try:
        # Everything that can fail without sealed data happens before "ready".
        if sys.platform != 'darwin' or not _worker_boundary_active():
            raise HoldoutError('holdout_worker_boundary_not_active')
        import pyarrow.parquet as pq
        import sklearn.metrics  # noqa: F401 -- load metric code before the read
        from marvis.feature.metrics import feature_auc, feature_ks
        feature_ks(np.array([0.1, 0.9]), np.array([0, 1]))
        feature_auc(np.array([0.1, 0.9]), np.array([0, 1]))
        scorer = _scorer(job['artifact'], job['runtime_dir'])
    except HoldoutError as exc:
        _send(out, {'stage': 'failed', 'code': exc.code})
        return
    except Exception:  # noqa: BLE001 -- the sandboxed child reports only a fixed code
        _send(out, {'stage': 'failed', 'code': 'holdout_isolated_worker_not_ready'})
        return
    _send(out, {'stage': 'ready'})
    if sys.stdin.readline().strip() != 'GO':
        return
    try:
        data = Path(job['sealed_path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != job['sealed_hash']:
            raise HoldoutError('holdout_sealed_integrity_failed')
        frame = pq.read_table(BytesIO(data)).to_pandas()
        del data
        metrics = sealed_metrics(frame, job['contract'], job['artifact']['params'].get('classes'), scorer)
        reply = {'stage': 'result', 'ok': True, 'metrics': metrics}
    except HoldoutError as exc:
        reply = {'stage': 'result', 'ok': False, 'code': exc.code}
    except Exception:  # noqa: BLE001
        reply = {'stage': 'result', 'ok': False, 'code': 'holdout_isolated_evaluation_failed'}
    _send(out, reply)


if __name__ == '__main__':
    main()
