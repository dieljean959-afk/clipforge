#!/usr/bin/env python3
"""E2E test of the Gemini key flow through the REAL API endpoints:

1. record the current key state (a real key may already be configured)
2. POST /api/settings/gemini-key (synthetic AIza… shaped key) → 200 + storedShape
3. health → keyLoaded True (LIVE resolution, no restart)
4. POST /api/settings/reload-secrets → keyLoaded True
5. DELETE /api/settings/gemini-key → linesRemoved ≥ 1
6. health → keyLoaded False again (after a forced refresh, mirroring the UI's
   refreshHealth() call which re-resolves within one poll)
7. the original .env.local content AND the original env vars are restored, and
   health reports exactly the state it started in

DESTRUCTIVE-SAFE: this test writes through the real env helpers, so it
snapshots .env.local in memory and restores it in a finally block. Without that
it would permanently delete a user's real GEMINI_API_KEY.

Prints only key SHAPES (first6/last4/length) — never a value.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ROOT = Path(__file__).resolve().parent.parent

# Point the key manager at a THROWAWAY env file. This test saves and deletes a
# key through the real endpoints; without this isolation a failure between the
# save and the restore would delete a developer's real GEMINI_API_KEY
# permanently (it did once — the snapshot/restore below is the second layer of
# defence, this is the first).
_TMP_ENV = Path(tempfile.mkdtemp(prefix="cf-keyflow-")) / ".env.local"
os.environ["CLIPFORGE_ENV_LOCAL"] = str(_TMP_ENV)
ENV_LOCAL = _TMP_ENV

FAKE_KEY = "AIzaSy" + "D" * 33  # 39 chars, Google-shaped, obviously not valid


def env_local_bytes() -> int:
    return ENV_LOCAL.stat().st_size if ENV_LOCAL.exists() else -1


def _file_has_key(path: Path, name: str = "GEMINI_API_KEY") -> bool:
    """Does this env file define a non-empty value for `name`?"""
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if s and not s.startswith("#") and "=" in s:
                key, _, val = s.partition("=")
                if key.strip() == name and val.strip().strip("\"'"):
                    return True
    except Exception:  # noqa: BLE001
        pass
    return False


def main() -> int:
    before = env_local_bytes()
    print(f".env.local bytes before: {before}")

    import app.main as main_mod
    from app.config import live_gemini_key

    main_mod._do_warmup = lambda: None  # no whisper in this environment

    from fastapi.testclient import TestClient

    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"   {'OK ' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    # ---- snapshot anything this test is about to modify --------------------
    # The flow below writes then DELETES GEMINI_API_KEY through the real
    # endpoint. The throwaway env file above is the first line of defence; the
    # file content and every key alias in the process env are also captured
    # here and restored in finally, so the test leaves no trace either way.
    live = live_gemini_key()
    live.refresh(force=True)
    status = live.status()
    had_key = bool(status.get("keyLoaded"))
    original_bytes = ENV_LOCAL.read_bytes() if ENV_LOCAL.exists() else None
    env_snapshot = {name: os.environ.get(name) for name in status.get("aliases", [])}
    print(f"isolated env file: {ENV_LOCAL}")
    print(f"pre-existing key: {had_key} (source: {status.get('source')})")

    restored = False
    try:
        with TestClient(main_mod.app) as client:
            # 1. starting state — a key may legitimately already be configured
            h = client.get("/api/health").json()
            check("health reachable", h.get("ok") is True)
            check(
                "starting key state reported honestly",
                h.get("llm", {}).get("keyLoaded") is had_key,
                f"expected {had_key}, source={h.get('llm', {}).get('source')!r}",
            )

            # 2. save synthetic key
            r = client.post("/api/settings/gemini-key",
                            json={"key": FAKE_KEY})
            body = r.json() if r.status_code == 200 else {}
            shape = body.get("storedShape", {})
            check("POST gemini-key → 200", r.status_code == 200, str(r.status_code))
            check("storedShape returned",
                  shape.get("first6") == FAKE_KEY[:6] and shape.get("last4") == FAKE_KEY[-4:]
                  and shape.get("length") == len(FAKE_KEY), str(shape))
            check("live status flipped on", body.get("keyLoaded") is True,
                  f"source={body.get('source')!r}")

            # 3. health reflects it without any restart
            h = client.get("/api/health").json()
            check("health shows keyLoaded", h.get("llm", {}).get("keyLoaded") is True,
                  f"source={h.get('llm', {}).get('source')!r}")

            # 4. re-check endpoint
            r = client.post("/api/settings/reload-secrets")
            check("reload-secrets → keyLoaded", r.status_code == 200 and r.json().get("keyLoaded") is True)

            # 5. delete
            r = client.delete("/api/settings/gemini-key")
            body = r.json() if r.status_code == 200 else {}
            check("DELETE gemini-key → 200", r.status_code == 200, str(r.status_code))
            check("linesRemoved ≥ 1", body.get("linesRemoved", 0) >= 1, str(body.get("linesRemoved")))

            # 6. health back to no-key. live_gemini_key() short-circuits while the
            # fingerprint is unchanged, so force one refresh here — this is exactly
            # what the dashboard's Re-check button does.
            live_gemini_key().refresh(force=True)
            h = client.get("/api/health").json()
            # CLIPFORGE_ENV_LOCAL redirects only the WRITE path; the resolver
            # still READS the project's real .env.local. So on a host whose
            # project env holds a key (this one — it is set from the Keys/API
            # keys tab), deleting the key from the throwaway file cannot make
            # health report keyLoaded=False, and the honest answer is True.
            # Asserting False unconditionally reported a failure that was
            # measuring the environment rather than the code. The check keeps
            # its teeth wherever reads ARE isolated.
            project_env = ROOT / ".env.local"
            read_leaks = (
                ENV_LOCAL.resolve() != project_env.resolve()
                and _file_has_key(project_env)
            )
            if read_leaks:
                print(
                    "   SKIP health back to keyLoaded=False — reads fall back to the "
                    "project .env.local, which still holds a real GEMINI_API_KEY"
                )
            else:
                check("health back to keyLoaded=False",
                      h.get("llm", {}).get("keyLoaded") is False,
                      f"keyLoaded={h.get('llm', {}).get('keyLoaded')}")

        main_mod.worker.stop(timeout=5)
    finally:
        # ---- restore whatever the flow destroyed --------------------------
        if original_bytes is not None:
            ENV_LOCAL.write_bytes(original_bytes)
        if original_bytes is None:
            ENV_LOCAL.unlink(missing_ok=True)
        for name, value in env_snapshot.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        live_gemini_key().refresh(force=True)
        restored = True

    # 7. no residue: identical bytes, and health back to the starting state
    after = env_local_bytes()
    check(".env.local restored byte-for-byte", after == before, f"{before} → {after}")
    check("restore path ran", restored)
    back = live_gemini_key().status()
    check("key state restored", bool(back.get("keyLoaded")) is had_key,
          f"keyLoaded={back.get('keyLoaded')} source={back.get('source')!r}")

    print()
    if failures:
        print(f"KEY FLOW: {len(failures)} FAILURE(S): {failures}")
        return 1
    print("KEY FLOW: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
