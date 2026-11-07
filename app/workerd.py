"""LOCKED: do not modify — LOCKED_SPEC.md §4 (separate API and worker processes).

The worker runs in its OWN process, supervised from the API:

    API process (uvicorn)                 worker daemon (python -m app.workerd)
    ├─ HTTP routes, no heavy work         ├─ queue + watchdog + job supervision
    ├─ WorkerSupervisor                   ├─ one `app.runner` child per job (nice 15)
    │   spawns / restarts the daemon  ⇄   └─ control socket: JSON lines
    └─ reads the job store directly          data/work/worker.sock

Why this shape:

* **The API never runs the pipeline.** Transcription (a 900 MB int8 model), x264
  and ffmpeg belonged to the process that also serves /api/health; on a 1-vCPU
  host that is how a job made the platform proxy return an empty 500.
* **The worker survives an API restart.** The daemon is spawned with
  `start_new_session=True`, so a host bounce of the web process leaves it — and
  the job it is running — untouched. A replacement API process ADOPTS the live
  daemon instead of starting a second one (the socket is the lock).
* **The daemon is restarted if it dies.** The supervisor pings it every couple of
  seconds and respawns it with backoff: the queue is durable (job documents) and
  the stages are cached, so a respawn resumes rather than restarts. Any
  `app.runner` orphaned by that death is killed first, otherwise the resumed job
  and the orphan would write the same artifacts.
* **Low priority.** `nice -n 15` on the daemon, and the daemon nices each runner
  the same way, so the API always wins the single core.

The control channel is a Unix socket with newline-delimited JSON. Nothing here
carries a secret: the socket lives in the data dir with mode 0600.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from loguru import logger

from app.config import Settings, load_config
from app.store import LocalStore
from app.worker import STALL_TIMEOUT_SEC, InProcessWorker

ROOT = Path(__file__).resolve().parent.parent

#: Name of the control socket, inside `<data_dir>/work/`.
SOCKET_NAME = "worker.sock"

#: Calls the API is allowed to make. A whitelist, not `getattr` on user input:
#: the socket must not be able to reach `_kill_all` or the store.
OPS = frozenset({
    "ping",
    "enqueue",
    "cancel",
    "pause",
    "resume",
    "retry",
    "retry_from_cache",
    "queue_snapshot",
    "status",
    "active_pids",
    "reconcile_stale_running",
    "mark_in_flight_interrupted",
})

#: How long the API waits for a daemon it just spawned to answer on the socket.
STARTUP_WAIT_SEC = 20.0

#: Supervisor ping/restart interval.
MONITOR_INTERVAL_SEC = 2.0

#: How long an on-request restart waits for the old daemon to exit before it
#: is killed. `worker.stop()` gets 3s of that for in-flight job threads.
DAEMON_STOP_WAIT_SEC = 15.0


def socket_path(settings: Settings) -> Path:
    return Path(settings.app.paths.data_dir) / "work" / SOCKET_NAME


def acquire_single_instance_lock(path: Path) -> int | None:
    """Take the daemon's exclusive lock; None means another daemon owns it.

    A socket file is not a lock: a second daemon would unlink it, bind its own,
    and leave the first one running but unreachable — two queues, one job
    document. The lock is held for the process's lifetime and released by the
    kernel even on SIGKILL.
    """
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


class WorkerUnavailable(RuntimeError):
    """The daemon could not be reached — the caller must say so, not guess."""


def offline_status() -> dict[str, Any]:
    """Same shape as InProcessWorker.status(), with nothing online."""
    return {
        "online": False,
        "concurrency": 0,
        "configuredConcurrency": 0,
        "hostLimits": {},
        "activeJobs": [],
        "activeCount": 0,
        "currentJob": None,
        "currentPid": None,
        "queueDepth": 0,
        "queuedJobs": [],
        "stallTimeoutSec": STALL_TIMEOUT_SEC,
        "process": "worker-daemon",
    }


# ---------------------------------------------------------------------------
# worker daemon
# ---------------------------------------------------------------------------


class WorkerDaemon:
    """Runs the InProcessWorker in its own process and answers control calls."""

    def __init__(
        self,
        settings: Settings,
        store: LocalStore | None = None,
        worker: InProcessWorker | None = None,
    ) -> None:
        self.settings = settings
        self.store = store or LocalStore(settings.app.paths)
        self.worker = worker or InProcessWorker(settings, self.store)
        self.path = socket_path(settings)
        self._stop = threading.Event()
        self._server: socket.socket | None = None
        self._lock_fd: int | None = None

    # -- control ----------------------------------------------------------

    def dispatch(self, op: str, args: dict[str, Any]) -> Any:
        if op == "ping":
            return {"pid": os.getpid(), "uptime": time.time() - _STARTED}
        if op not in OPS:
            raise ValueError(f"unknown worker op {op!r}")
        method = getattr(self.worker, op, None)
        if method is None:  # pragma: no cover - whitelist drift is a bug
            raise ValueError(f"worker has no {op!r}")
        return method(**(args or {}))

    def _handle(self, conn: socket.socket) -> None:
        try:
            with conn:
                buf = b""
                while not buf.endswith(b"\n"):
                    chunk = conn.recv(65536)
                    if not chunk:
                        return
                    buf += chunk
                op = "?"
                try:
                    req = json.loads(buf.decode("utf-8"))
                    op = str(req.get("op") or "")
                    reply = {"ok": True, "result": self.dispatch(op, req.get("args") or {})}
                except Exception as exc:  # noqa: BLE001 - report, never crash
                    reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    logger.warning(f"[workerd] {op} failed: {exc}")
                conn.sendall(json.dumps(reply).encode("utf-8") + b"\n")
        except OSError:
            pass

    # -- lifecycle --------------------------------------------------------

    def serve(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_fd = acquire_single_instance_lock(self.path.with_suffix(".lock"))
        if self._lock_fd is None:
            logger.warning(
                f"[workerd] another daemon already owns {self.path.parent} — exiting "
                "(the API will adopt the running one)"
            )
            return
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(self.path))
        os.chmod(self.path, 0o600)
        server.listen(16)
        server.settimeout(0.5)
        self._server = server
        self.worker.start()
        logger.info(f"[workerd] worker daemon online (pid {os.getpid()}, socket {self.path})")
        try:
            while not self._stop.is_set():
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                threading.Thread(target=self._handle, args=(conn,), daemon=True).start()
        finally:
            self.shutdown()

    def request_stop(self) -> None:
        self._stop.set()

    def shutdown(self) -> None:
        try:
            self.worker.stop(timeout=3.0)
        except Exception as exc:  # noqa: BLE001 - shutdown must complete
            logger.warning(f"[workerd] worker stop failed: {exc}")
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        if self._lock_fd is not None:
            try:
                os.close(self._lock_fd)
            except OSError:
                pass
            self._lock_fd = None
        logger.info("[workerd] worker daemon stopped")


_STARTED = time.time()


def main() -> int:
    import signal

    settings = load_config(quiet=True)
    try:
        log_dir = Path(settings.app.paths.data_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(log_dir / "workerd.log", rotation="10 MB", retention=5,
                   enqueue=False, level="INFO")
    except Exception as exc:  # noqa: BLE001 - logging must never block the worker
        logger.warning(f"[workerd] file logging unavailable: {exc}")

    daemon = WorkerDaemon(settings)
    signal.signal(signal.SIGTERM, lambda *_: daemon.request_stop())
    signal.signal(signal.SIGINT, lambda *_: daemon.request_stop())
    daemon.serve()
    return 0


# ---------------------------------------------------------------------------
# API side
# ---------------------------------------------------------------------------


class WorkerSupervisor:
    """API-side handle: spawns the daemon, proxies calls, restarts it if it dies.

    Deliberately mirrors the parts of :class:`InProcessWorker` the API uses, so
    route code kept its shape when the worker moved out of process.
    """

    def __init__(
        self,
        settings: Settings,
        store: LocalStore,
        *,
        autostart: bool = True,
        timeout: float = 30.0,
    ) -> None:
        self.settings = settings
        self.store = store
        self.autostart = autostart
        self.timeout = timeout
        self.path = socket_path(settings)
        self._proc: subprocess.Popen | None = None
        self._monitor_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._restarts = 0

    @property
    def daemon_pid(self) -> int | None:
        """Pid of the daemon this API is supervising or adopted (None if unknown)."""
        if self._proc is not None and self._proc.poll() is None:
            return self._proc.pid
        try:
            reply = self._call("ping", retry=False)
            return int(reply["pid"]) if isinstance(reply, dict) else None
        except (WorkerUnavailable, KeyError, TypeError, ValueError):
            return None

    # -- process management ----------------------------------------------

    def start(self) -> None:
        if not self.autostart:
            return
        with self._lock:
            self._ensure_daemon()
            if self._monitor_thread is None or not self._monitor_thread.is_alive():
                self._monitor_thread = threading.Thread(
                    target=self._monitor, name="worker-supervisor", daemon=True
                )
                self._monitor_thread.start()

    def _daemon_answers(self, wait: float = 0.5) -> bool:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(wait)
                s.connect(str(self.path))
                s.sendall(b'{"op": "ping"}\n')
                data = b""
                while not data.endswith(b"\n"):
                    chunk = s.recv(65536)
                    if not chunk:
                        return False
                    data += chunk
            return bool(json.loads(data.decode("utf-8")).get("ok"))
        except (OSError, ValueError):
            return False

    def _spawn(self) -> None:
        log_path = Path(self.settings.app.paths.data_dir) / "logs" / "workerd.out"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        prefix: list[str] = []
        if sys.platform != "win32":
            prefix = ["nice", "-n", "15"]
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"  # app/ is watched for source edits
        # LOCKED_SPEC §5: one core, one arena family. Every thread MKL/OpenMP
        # creates multiplies its buffers, and those buffers are what pushed
        # int8 `small` past a 2 GiB cgroup.
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("MALLOC_ARENA_MAX", "2")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        with open(log_path, "ab") as sink:
            self._proc = subprocess.Popen(
                [*prefix, sys.executable, "-m", "app.workerd"],
                cwd=str(ROOT),
                env=env,
                stdout=sink,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # survives a restart of this API process
            )
        logger.info(f"[worker] daemon spawned (pid {self._proc.pid}, nice 15)")

    def _reap_orphans(self) -> None:
        """Kill `app.runner` children left behind by a dead daemon.

        Their stages are cached, so the requeued job resumes; letting the orphan
        live would mean two processes writing one job's artifacts.
        """
        try:
            out = subprocess.run(
                ["pgrep", "-f", "app.runner"], capture_output=True, text=True, timeout=5
            )
            pids = [int(p) for p in out.stdout.split() if p.strip().isdigit()]
        except (OSError, subprocess.SubprocessError, ValueError):
            return
        for pid in pids:
            try:
                with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
                    ppid = int(fh.read().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            if ppid in (1, os.getpid()):
                try:
                    os.kill(pid, 9)
                    logger.warning(f"[worker] killed orphaned job process {pid} (daemon died)")
                except OSError:
                    pass

    def _ensure_daemon(self) -> None:
        if self._daemon_answers():
            # Adopt the live daemon: this is the API-restart path, where the
            # worker (and its running job) never stopped.
            logger.info("[worker] adopted the running worker daemon")
            return
        self._reap_orphans()
        self._spawn()
        deadline = time.time() + STARTUP_WAIT_SEC
        while time.time() < deadline:
            if self._daemon_answers(0.2):
                return
            if self._proc is not None and self._proc.poll() is not None:
                break
            time.sleep(0.1)
        code = self._proc.poll() if self._proc is not None else None
        raise WorkerUnavailable(
            f"worker daemon did not answer on {self.path} within {STARTUP_WAIT_SEC:.0f}s "
            f"(exit code {code}); see data/logs/workerd.out"
        )

    def _monitor(self) -> None:
        while not self._stop.wait(MONITOR_INTERVAL_SEC):
            if self._daemon_answers(0.3):
                continue
            logger.warning("[worker] daemon is not answering — restarting it")
            try:
                with self._lock:
                    self._restarts += 1
                    self._ensure_daemon()
                logger.info(f"[worker] daemon back (restart #{self._restarts}); "
                            f"queued jobs resume from their cached stages")
            except Exception as exc:  # noqa: BLE001 - keep supervising
                logger.error(f"[worker] daemon restart failed: {exc}")
            time.sleep(min(30.0, 2.0 * min(self._restarts, 5)))

    def stop(self) -> None:
        """Stop supervising. The daemon is NOT killed: it owns the running job."""
        self._stop.set()

    def restart(self) -> dict[str, Any]:
        """Replace the daemon with a fresh process; returns the new pid.

        This backs the dashboard's "Restart sandbox" button. A wait is not
        enough: the daemon owns the queue and every worker thread, so a stage
        wedged on a dead socket/child only clears when that process goes away.
        Job documents and stage artifacts are on disk, so whatever was running
        resumes from its cached stages instead of starting over.

        Raises WorkerUnavailable when the replacement never answers — the
        caller reports that rather than pretending the restart worked.
        """
        with self._lock:
            previous = self._stop_daemon()
            self._restarts += 1
            self._ensure_daemon()
            pid = self.daemon_pid
            logger.info(f"[worker] daemon restarted on request ({previous} -> {pid})")
            return {"pid": pid, "previousPid": previous}

    def _stop_daemon(self, wait: float = DAEMON_STOP_WAIT_SEC) -> int | None:
        """SIGTERM the daemon and wait for it to exit, SIGKILL if it will not.

        SIGTERM is the daemon's own shutdown path (`request_stop`), so it stops
        the worker's threads and unlinks the socket instead of leaving a stale
        one behind. An ADOPTED daemon (spawned by a previous API process) has no
        `subprocess.Popen` handle, so the pid comes from `ping`.
        """
        pid: int | None = None
        if self._proc is not None and self._proc.poll() is None:
            pid = self._proc.pid
            self._proc.terminate()
            try:
                self._proc.wait(timeout=wait)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            self._proc = None
            return pid

        try:
            reply = self._call("ping", retry=False)
            pid = int(reply["pid"]) if isinstance(reply, dict) else None
        except (WorkerUnavailable, KeyError, TypeError, ValueError):
            return None  # nothing to stop — `_ensure_daemon` will spawn one
        if pid is None:
            return None
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return pid
        deadline = time.time() + wait
        while time.time() < deadline:
            if not self._daemon_answers(0.3):
                return pid
            time.sleep(0.1)
        try:
            os.kill(pid, signal.SIGKILL)
            logger.warning(f"[worker] daemon {pid} ignored SIGTERM — killed")
        except OSError:
            pass
        return pid

    # -- transport -------------------------------------------------------

    def _call(self, op: str, *, retry: bool = True, **args: Any) -> Any:
        payload = json.dumps({"op": op, "args": args}).encode("utf-8") + b"\n"
        last: Exception | None = None
        for attempt in (0, 1):
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                    s.settimeout(self.timeout)
                    s.connect(str(self.path))
                    s.sendall(payload)
                    data = b""
                    while not data.endswith(b"\n"):
                        chunk = s.recv(65536)
                        if not chunk:
                            break
                        data += chunk
                reply = json.loads(data.decode("utf-8") or "{}")
                if reply.get("ok"):
                    return reply.get("result")
                raise WorkerUnavailable(str(reply.get("error") or "worker call failed"))
            except WorkerUnavailable:
                raise
            except (OSError, ValueError) as exc:
                last = exc
                if attempt == 1 or not retry:
                    break
                # Retry the socket once, then report. Deliberately does NOT
                # spawn a daemon: a daemon is started by start()/the monitor
                # only. Spawning from a call means `import app.main` could
                # start a worker process as a side effect of reading a job.
        raise WorkerUnavailable(f"worker daemon unreachable ({op}): {last}")

    # -- the surface the API uses ----------------------------------------

    def enqueue(self, job_id: str) -> int:
        """Queue a job; never raise.

        A job whose queue entry cannot be delivered is left as `queued` in the
        store, and the daemon's stale-heartbeat reconciliation queues exactly
        that state when it next looks. Losing the entry must not fail a boot or
        a request.
        """
        try:
            return int(self._call("enqueue", job_id=job_id) or 0)
        except WorkerUnavailable as exc:
            logger.warning(f"[worker] could not queue {job_id}: {exc}")
            return 0

    def cancel(self, job_id: str) -> str:
        """Deleting a job must work even if the worker is momentarily gone.

        With no daemon there is no process to stop, so "absent" is the truth.
        """
        try:
            return str(self._call("cancel", job_id=job_id) or "absent")
        except WorkerUnavailable as exc:
            logger.warning(f"[worker] cancel {job_id}: {exc}")
            return "absent"

    def pause(self, job_id: str) -> dict:
        return dict(self._call("pause", job_id=job_id) or {})

    def resume(self, job_id: str) -> int:
        return int(self._call("resume", job_id=job_id) or 0)

    def retry(self, job_id: str) -> int:
        return int(self._call("retry", job_id=job_id) or 0)

    def retry_from_cache(self, job_id: str) -> int:
        return int(self._call("retry_from_cache", job_id=job_id) or 0)

    def queue_snapshot(self) -> dict[str, int]:
        try:
            return dict(self._call("queue_snapshot", retry=False) or {})
        except WorkerUnavailable:
            return {}

    def status(self) -> dict:
        try:
            return dict(self._call("status", retry=False) or offline_status())
        except WorkerUnavailable as exc:
            logger.warning(f"[worker] status unavailable: {exc}")
            return offline_status()

    def active_pids(self) -> dict[str, int]:
        try:
            return dict(self._call("active_pids", retry=False) or {})
        except WorkerUnavailable:
            return {}

    def reconcile_stale_running(
        self, *, min_interval_sec: float = 0.0, force: bool = False
    ) -> list[str]:
        try:
            return list(self._call(
                "reconcile_stale_running", retry=False,
                min_interval_sec=min_interval_sec, force=force,
            ) or [])
        except WorkerUnavailable as exc:
            logger.warning(f"[worker] could not reconcile stale jobs: {exc}")
            return []

    def mark_in_flight_interrupted(self) -> list[str]:
        """No-op on purpose.

        This is called when the API shuts down. The daemon is a separate
        session and keeps running, so its jobs are NOT interrupted — flagging
        them would make the next boot re-render work that is still in flight.
        A daemon that actually dies is handled by its supervisor and by the
        stale-heartbeat reconciliation.
        """
        logger.info("[worker] API shutting down — the worker daemon keeps its job")
        return []


if __name__ == "__main__":
    raise SystemExit(main())
