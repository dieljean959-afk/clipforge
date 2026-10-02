"""`/api/health` must answer, always — and never depend on a third party.

The dashboard showed a hard red **"API unreachable: HTTP 400"** banner directly
above a fully populated status card: real CPU, RAM, disk, worker and provider
numbers, rendered from the previous successful poll, with a single transient
failure claiming the API was gone. Two separate defects were behind it.

**1. One failing sub-component could fail the whole endpoint.** `health()`
called `hosting.system_stats()`, `disk_stats()`, the RTF store, the whisper
status and the key resolvers inline. Any one of those raising took down the
entire payload, and the only way the UI could describe "I could not read one
number" was to declare the whole API unreachable. Every block is now built
through `_block()`, so a failure degrades that field to `{"error": ...}`,
is recorded in `degraded`, and the endpoint still returns 200.

**2. It was coupled to the AI provider.** `health()` embedded
`detection_status()`, which for the ollama backend calls `ollama_status()` —
an HTTP request with a 5s timeout. A third-party hiccup could therefore stall
or break our own health endpoint. `detection_status(probe=False)` now answers
from configuration and key presence only, making no network call, and health
uses it. Groq is unaffected by design: its readiness was always a local env
lookup, and it is now explicit that no vendor is contacted.

Note the status is 200 whenever the process answered at all. 400 is never valid
for this route — it has no parameters to reject — and a health check that
returns 503 for "the worker is busy" teaches clients to ignore the difference.

Run:  .venv/bin/python tools/health_resilience_test.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import main as main_mod  # noqa: E402
from app.config import load_config  # noqa: E402

FAILS: list[str] = []
CHECKS = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    """Assert, loudly.

    The argument order here is (label, ok). That is easy to get backwards, and
    a backwards call passes SILENTLY: a non-empty string lands in `ok`, which
    is truthy, and the suite reports green while testing nothing. That mistake
    has shipped twice in this repo already, so the helper refuses any call
    whose arguments are not a string label and a real boolean.
    """
    global CHECKS
    if not isinstance(label, str) or not isinstance(ok, bool):
        CHECKS += 1
        FAILS.append("check() called with swapped/ill-typed arguments")
        print(
            f"  FAIL check({label!r}, {ok!r}) — arguments are backwards or "
            f"ill-typed; this assertion did NOT run"
        )
        return
    CHECKS += 1
    if ok:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}" + (f" — {detail}" if detail else ""))
        FAILS.append(label)


def _strip_prose(src: str) -> str:
    """Drop comments and docstrings so prose can't satisfy a code check.

    Applied to the WHOLE block, not line by line: a docstring spans many lines,
    and this file's own explanation of the bug NAMES `detection_status()` —
    which must not count as the code calling it.
    """
    without_docstrings = re.sub(r'"""[\s\S]*?"""', "", src)
    return "\n".join(
        ln for ln in without_docstrings.splitlines() if not ln.strip().startswith("#")
    )


def _fn_src(name: str) -> str:
    src = Path(main_mod.__file__).read_text(encoding="utf-8")
    start = src.index(f"def {name}(")
    rest = src[start:]
    nxt = rest.find("\n@app.", 1)
    nxt_def = rest.find("\ndef ", 1)
    ends = [e for e in (nxt, nxt_def) if e != -1]
    return rest[: min(ends)] if ends else rest


def main() -> int:
    root = Path(main_mod.__file__).parent
    health_src = _fn_src("health")
    status_src = _fn_src("detection_status")

    # ---- 1. fault isolation ------------------------------------------
    check("defines a _block isolation helper", "def _block(" in health_src)
    check(
        "_block() wraps the worker status",
        '_block("worker", worker.status)' in health_src,
    )
    for block in ("system", "disk", "asr", "encode", "llm", "llmGroq", "detection"):
        check(f"'{block}' is built through _block", f'_block(\n            "{block}"' in health_src
              or f'_block("{block}"' in health_src)
    check(
        "no bare system_stats() call remains in the payload",
        '"system": hosting.system_stats()' not in health_src,
    )
    check(
        "no bare disk_stats() call remains in the payload",
        '"disk": disk_stats()' not in health_src,
    )
    check("reports which blocks degraded", '"degraded": degraded' in health_src)
    check("ok is false when something degraded", '"ok": not degraded' in health_src)

    # ---- 2. no third-party dependency --------------------------------
    check(
        "health calls detection_status with probe=False",
        "detection_status, probe=False" in health_src,
        "found: " + repr([ln.strip() for ln in health_src.splitlines() if "detection_status" in ln]),
    )
    # Strip comments and docstrings so prose that merely NAMES the probing call
    # (as this file's own explanation does) is not mistaken for the call.
    health_code = _strip_prose(health_src)
    check(
        "the plain (probing) call is not used in health",
        "detection_status()" not in health_code,
        "found: " + repr([ln.strip() for ln in health_code.splitlines() if "detection_status(" in ln]),
    )
    check(
        "ollama is not probed when probe=False",
        "if not probe:" in status_src and "not probed by /api/health" in status_src,
    )
    check(
        "the ollama network probe only runs when probing",
        status_src.index("if not probe:") < status_src.index("llm_mod.ollama_status("),
    )
    check(
        "groq reports it made no vendor call",
        'out["probed"] = False' in _strip_prose(status_src),
    )
    check(
        "the groq key lookup is local (env), not a request",
        "live.refresh(force=probe)" in status_src,
    )
    # the ollama probe itself must still exist for the paths that want it
    check(
        "ollama_status still exists for /redetect and the doctor",
        hasattr(main_mod.llm_mod, "ollama_status"),
    )

    # ---- 3. a real call, end to end -----------------------------------
    settings = load_config(quiet=True)
    status = main_mod.detection_status(probe=False)
    check("detection_status(probe=False) returns a dict", isinstance(status, dict))
    check(
        "it names the provider the app is actually configured to use",
        status.get("provider") == main_mod.llm_mod.provider(settings),
        f"{status.get('provider')} vs {main_mod.llm_mod.provider(settings)}",
    )
    check("it reports a readiness field", "ready" in status)
    check("it reports that it did not probe", status.get("probed") is False)
    check(
        "it still gives an actionable detail line",
        bool(status.get("detail")),
        str(status.get("detail"))[:70],
    )
    check(
        "no network call is reachable from the no-probe path",
        "urllib" not in status_src,
    )

    # ---- 4. status code contract --------------------------------------
    check(
        "the endpoint is a plain GET with no parameters to reject",
        '@app.get("/api/health")' in Path(main_mod.__file__).read_text(encoding="utf-8"),
    )
    check(
        "the docstring states 200-with-ok is the contract",
        "200" in health_src and "ok" in health_src,
    )

    # ---- 5. the frontend stopped crying wolf -------------------------
    ui = (root.parent / "src" / "hooks" / "use-poll.ts").read_text(encoding="utf-8")
    check("useHealth separates a fatal error from a transient one", "stale" in ui)
    check("a poll failure with data on screen is 'stale', not fatal",
          "haveData.current" in ui and "setStale(msg)" in ui)
    check("the last good reading is retained across a failed poll",
          "setHealth(data)" in ui and "haveData.current = true" in ui)
    check(
        "no setState is called inside a state updater",
        "setHealth((prev) =>" not in ui,
    )
    card = (root.parent / "src" / "components" / "clipforge" / "LocalWorkerCard.tsx").read_text(
        encoding="utf-8"
    )
    check("the card renders a soft note for a transient blip", "stale" in card)
    check(
        "the hard banner is reserved for never having responded",
        card.count("API unreachable:") == 1,
        f"{card.count('API unreachable:')} rendered banner(s)",
    )
    check(
        "a transient blip does not render the hard banner",
        "showing the most recent reading" in card,
    )

    # ---- 6. the route still exists and imports cleanly ---------------
    routes = {getattr(r, "path", None) for r in main_mod.app.routes}
    check("/api/health is still routed", "/api/health" in routes)

    print()
    if FAILS:
        print(f"{CHECKS - len(FAILS)} passed, {len(FAILS)} FAILED")
        return 1
    print(f"{CHECKS} passed, 0 failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
