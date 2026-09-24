"""Single-job runner process.

Each job runs in its OWN process (spawned by InProcessWorker). This is what
makes the stall watchdog real: a hung whisper/ffmpeg call inside the API
process cannot be killed (Python cannot interrupt a thread), but an entire
child process — including any ffmpeg descendants — can be SIGKILLed, releasing
its memory, file handles and locks. A retry then spawns a genuinely fresh
worker instead of re-attaching to the hung one.

Usage:  python -m app.runner <job_id>
"""

from __future__ import annotations

import sys


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python -m app.runner <job_id>", file=sys.stderr)
        return 2
    job_id = argv[1]
    from app.config import load_config
    from app.store import LocalStore
    from app.task import Pipeline

    settings = load_config(quiet=True)
    store = LocalStore(settings.app.paths)
    Pipeline(settings, store).run(job_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
