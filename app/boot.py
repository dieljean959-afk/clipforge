"""Boot supervisor: the port opens first, everything heavy happens after.

The API used to run its recovery, share backfill, sweeps and worker spawn inside
import time / lifespan startup. uvicorn binds its port *after* lifespan startup
finishes, so any slow phase there is indistinguishable from "the server is down":
the browser shows "booting up preview… 97%" and the supervisor proxy answers 503
"API restarting".

This module makes the split explicit and measurable:

* :class:`Boot` runs named phases on a daemon thread, each with a timeout.
* A phase failure is recorded and reported, never raised into the request path.
* ``/api/health`` answers from memory only, so it is 200 the moment the port is
  open. ``/api/status`` carries the phase table plus per-dependency probes.
* :class:`InstanceLock` keeps two servers (or two workers) from fighting over
  the port and the same job files; a stale lock (dead pid) is reclaimed.
* :func:`record_fatal` writes ``data/boot_failure.json`` so a real boot failure
  is shown with its cause and fix instead of an endless spinner.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from loguru import logger

#: Phase names are stable strings: the dashboard and doctor.py key off them.
PHASES = (
    "worker_spawn",
    "worker_purge",
    "reclaim_watchdog",
    "crash_sweep",
    "share_backfill",
    "job_recovery",
    "model_preload",
    "retention_sweep",
    "log_hygiene",
    "invariants",
    "update_checks",
    "health_snapshot",
)


@dataclass
class Phase:
    name: str
    status: str = "pending"  # pending | running | ok | failed | skipped | timeout
    startedAt: float | None = None
    endedAt: float | None = None
    ms: int | None = None
    detail: str | None = None
    error: str | None = None
    timeoutMs: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "ms": self.ms,
            "detail": self.detail,
            "error": self.error,
            "timeoutMs": self.timeoutMs,
            "startedAt": self.startedAt,
            "endedAt": self.endedAt,
        }


class Boot:
    """Runs post-listen work in the background, with a phase timeline."""

    def __init__(self) -> None:
        self.started_at = time.time()
        #: Milliseconds from process start to the first /api/health 200.
        self.first_health_ms: int | None = None
        #: Milliseconds from process start to the port being open (listen).
        self.listen_ms: int | None = None
        self.phases: dict[str, Phase] = {
            name: Phase(name) for name in PHASES
        }
        self.extra: list[Phase] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.done = False
        self.fatal: dict[str, Any] | None = None

    # -- phases ---------------------------------------------------------
    def phase(self, name: str, timeout_ms: int | None = None) -> Phase:
        with self._lock:
            ph = self.phases.get(name)
            if ph is None:
                ph = Phase(name)
                self.extra.append(ph)
            ph.timeoutMs = timeout_ms
            return ph

    def run_phase(
        self,
        name: str,
        fn: Callable[[], Any],
        *,
        timeout_ms: int | None = None,
        required: bool = False,
    ) -> Phase:
        """Run one phase, bounded by ``timeout_ms``.

        The phase is executed on a helper thread so a hung call (DNS, a stuck
        socket, a slow disk walk) cannot hold the boot thread past its budget.
        A timeout marks the phase ``timeout`` and moves on; the helper thread is
        left to die on its own daemon time.
        """
        ph = self.phase(name, timeout_ms)
        ph.status = "running"
        ph.startedAt = time.time()
        t0 = time.perf_counter()
        box: dict[str, Any] = {}

        def _target() -> None:
            try:
                box["value"] = fn()
            except BaseException as exc:  # noqa: BLE001 - reported, not raised
                box["error"] = exc

        th = threading.Thread(target=_target, name=f"boot-{name}", daemon=True)
        th.start()
        th.join(None if timeout_ms is None else timeout_ms / 1000.0)
        ph.endedAt = time.time()
        ph.ms = int((time.perf_counter() - t0) * 1000)
        if th.is_alive():
            ph.status = "timeout"
            ph.error = f"still running after {timeout_ms} ms — moved on"
            logger.warning(f"[boot] phase {name} timed out after {timeout_ms} ms")
            if required:
                raise TimeoutError(ph.error)
            return ph
        exc = box.get("error")
        if exc is not None:
            ph.status = "failed"
            ph.error = f"{type(exc).__name__}: {exc}"
            logger.warning(f"[boot] phase {name} failed: {ph.error}")
            if required:
                raise exc  # type: ignore[misc]
            return ph
        value = box.get("value")
        if isinstance(value, dict) and value.get("skipped"):
            ph.status = "skipped"
            ph.detail = value.get("detail")
        else:
            ph.status = "ok"
            if isinstance(value, dict) and value.get("detail"):
                ph.detail = str(value["detail"])
        logger.info(f"[boot] phase {name}: {ph.status} in {ph.ms} ms")
        return ph

    # -- background runner ----------------------------------------------
    def start(self, job: Callable[["Boot"], None], delay_ms: int = 0) -> None:
        """Start the phase list on a daemon thread and return immediately."""

        def _run() -> None:
            if delay_ms:
                time.sleep(delay_ms / 1000.0)
            try:
                job(self)
            except BaseException as exc:  # noqa: BLE001 - never kill the API
                logger.exception(f"[boot] background boot crashed: {exc}")
                self.fatal = failure_payload(
                    "boot",
                    f"{type(exc).__name__}: {exc}",
                    "The API is serving, but post-listen setup did not finish. "
                    "Check /api/status and data/logs/api.log.",
                )
                record_fatal(self.fatal)
            finally:
                self.done = True

        self._thread = threading.Thread(target=_run, name="boot-phases", daemon=True)
        self._thread.start()

    def mark_health(self) -> None:
        if self.first_health_ms is None:
            self.first_health_ms = int((time.time() - self.started_at) * 1000)

    def summary(self) -> dict[str, Any]:
        phases = [p.as_dict() for p in self.phases.values()] + [
            p.as_dict() for p in self.extra
        ]
        failed = [p["name"] for p in phases if p["status"] in ("failed", "timeout")]
        return {
            "startedAt": int(self.started_at * 1000),
            "listenMs": self.listen_ms,
            "firstHealthMs": self.first_health_ms,
            "done": self.done,
            "phases": phases,
            "failedPhases": failed,
            "fatal": self.fatal,
        }


# --------------------------------------------------------------------------- #
# loud failure + single instance
# --------------------------------------------------------------------------- #

_LAST_FAILURE: dict[str, Any] | None = None


def failure_payload(
    where: str,
    cause: str,
    fix: str,
    *,
    code: str = "boot_failed",
) -> dict[str, Any]:
    return {
        "where": where,
        "cause": cause,
        "fix": fix,
        "code": code,
        "at": int(time.time() * 1000),
    }


def record_fatal(payload: dict[str, Any], data_dir: Path | None = None) -> None:
    """Persist a boot failure so the preview can show it instead of spinning."""
    global _LAST_FAILURE
    _LAST_FAILURE = payload
    logger.error(
        f"[boot] FATAL ({payload.get('where')}): {payload.get('cause')} "
        f"→ FIX: {payload.get('fix')}"
    )
    if data_dir is None:
        return
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        tmp = data_dir / "boot_failure.json.tmp"
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        os.replace(tmp, data_dir / "boot_failure.json")
    except Exception as exc:  # noqa: BLE001 - reporting must never fail boot
        logger.warning(f"[boot] could not write boot_failure.json: {exc}")


def last_failure(data_dir: Path) -> dict[str, Any] | None:
    if _LAST_FAILURE:
        return _LAST_FAILURE
    try:
        return json.loads((data_dir / "boot_failure.json").read_text("utf-8"))
    except Exception:
        return None


def clear_failure(data_dir: Path) -> None:
    global _LAST_FAILURE
    _LAST_FAILURE = None
    try:
        (data_dir / "boot_failure.json").unlink()
    except FileNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"[boot] could not clear boot_failure.json: {exc}")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


@dataclass
class LockInfo:
    path: Path
    pid: int
    created_at: float = field(default_factory=time.time)


class InstanceLock:
    """A pid lock so two servers/workers never share the port or job files.

    ``acquire()`` is atomic (``O_EXCL``). A lock whose pid is gone is stale and
    is reclaimed on boot; a lock held by a *different live* pid is refused with
    the exact reason, which is what makes "port already served by a second
    uvicorn" a one-line error instead of two processes corrupting one store.
    """

    def __init__(self, path: Path, name: str, pid: int | None = None) -> None:
        self.path = path
        self.name = name
        self.pid = pid if pid is not None else os.getpid()
        self.acquired = False
        self.stale_reclaimed: int | None = None
        self.holder: int | None = None

    def _read(self) -> dict[str, Any] | None:
        try:
            return json.loads(self.path.read_text("utf-8"))
        except FileNotFoundError:
            return None
        except Exception:
            return None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                cur = self._read()
                holder = int((cur or {}).get("pid") or 0)
                if holder and holder != self.pid and _pid_alive(holder):
                    self.holder = holder
                    logger.error(
                        f"[boot] {self.name} lock is held by a live pid {holder} "
                        f"({self.path}) — refusing to start a second instance"
                    )
                    return False
                # stale: same pid (re-exec) or dead pid
                self.stale_reclaimed = holder or None
                logger.warning(
                    f"[boot] reclaiming stale {self.name} lock at {self.path} "
                    f"(pid {holder or 'unknown'} is not running)"
                )
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(
                    {"pid": self.pid, "name": self.name, "at": time.time()},
                    fh,
                )
            self.acquired = True
            return True
        return False

    def release(self) -> None:
        if not self.acquired:
            return
        cur = self._read()
        if int((cur or {}).get("pid") or 0) == self.pid:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
        self.acquired = False


def pid_alive(pid: int) -> bool:
    return _pid_alive(pid)
