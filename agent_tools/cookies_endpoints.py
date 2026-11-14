# Tool file — not imported. Contains the cookie endpoints to inject into app/main.py.
# The content below is read and written via the write_file/str_replace tools.

COOKIES_ENDPOINTS = r'''
# ---------------------------------------------------------------------------
# media download cookies — YAML file path (new: cookies_file / cookies_from_browser)
# ---------------------------------------------------------------------------
_COOKIES_VAR_TO_FIELD: dict[str, str] = {
    "YT_DLP_COOKIES_FILE": "cookies_file",
    "YT_DLP_COOKIES_FROM_BROWSER": "cookies_from_browser",
}

#: Request field -> env var for the live cookies dialog. Module level, not a
#: class attribute: Pydantic v2 turns a leading-underscore class attribute into a
#: private attr, which is not readable off the class the way this needs.
_COOKIES_FIELD_TO_VAR: dict[str, str] = {
    "cookies_file": "YT_DLP_COOKIES_FILE",
    "cookies_from_browser": "YT_DLP_COOKIES_FROM_BROWSER",
}

#: Process-wide live cookies-key state (created on first use).
#: Like the AI keys, the pasted cookies.txt is live account session material and
#: is treated as compromised after it is read here.
_live_cookies_key: _LiveKeyState | None = None


def _resolve_cookies_key(env: dict[str, str | None]) -> tuple[str | None, str, str | None]:
    """Find a usable cookies value: env-file contents beat the OS env.

    Returns (key, source, problem).
    """
    key = env.get("YT_DLP_COOKIES_FILE") or env.get("YT_DLP_COOKIES_FROM_BROWSER")
    if key:
        return key, "env-file",
    return None, "not found", "no cookies provided"


def set_cookies_key(body: dict[str, str | None]) -> dict[str, str]:
    """Save/replace the live cookies in .env.local.

    The cookies.txt is live account session material (SID, HSID, SSID, APISID,
    SAPISID, LOGIN_INFO, __Secure-3PSID, OSID, ACCOUNT_CHOOSER), so this is
    treated as compromised: the app exposes it in-app, and after saving it the
    user should sign out of all devices, change their Google password, enable
    two-factor authentication, and check recent security activity.
    """
    file_path = body.get("cookies_file") or ""
    browser_cookies = body.get("cookies_from_browser") or ""
    written: dict[str, str] = {}
    if file_path.strip():
        write_env_var("YT_DLP_COOKIES_FILE", file_path.strip())
        written["YT_DLP_COOKIES_FILE"] = file_path.strip()
    if browser_cookies.strip():
        write_env_var("YT_DLP_COOKIES_FROM_BROWSER", browser_cookies.strip())
        written["YT_DLP_COOKIES_FROM_BROWSER"] = browser_cookies.strip()

    # Refresh the process-level live key so every following request picks up
    # the new value without a restart.
    global _live_cookies_key
    _live_cookies_key = _LiveKeyState(
        ROOT,
        resolver=_resolve_cookies_key,
        aliases=("YT_DLP_COOKIES_FILE", "YT_DLP_COOKIES_FROM_BROWSER"),
        label="YT-DLP cookies",
    )
    _live_cookies_key.refresh(force=True)

    return {
        **written,
        "source": "file" if file_path.strip() else ("browser" if browser_cookies.strip() else "none"),
        "warning": (
            "cookies.txt is live account session material; treat as compromised "
            "and get a new password before the next render"
            if written
            else None
        ),
    }


@app.post("/api/settings/cookies")
def set_cookies_key(body: dict[str, str | None]) -> dict[str, str]:
    """Save/replace the live cookies in .env.local."""
    return set_cookies_key(body)


@app.delete("/api/settings/cookies")
def remove_cookies_key(confirm: bool = False) -> dict[str, str]:
    """Delete the live cookies from .env.local."""
    _require_delete_confirmation(
        confirm,
        "cookies (the pasted cookies.txt itself lives at /home/daytona/clipforge-data/cookies.txt)",
    )
    dropped = _sync_cookies_env_for_delete(
        ("YT_DLP_COOKIES_FILE", "YT_DLP_COOKIES_FROM_BROWSER"), "YT_DLP_COOKIES_FILE"
    )
    removed = sum(delete_cookies_env_var(name) for name in ("YT_DLP_COOKIES_FILE", "YT_DLP_COOKIES_FROM_BROWSER"))
    logger.info(
        f"[config] cookies deleted via dashboard ({removed} line(s) removed, process env dropped: {dropped or 'none'})"
    )
    # Drop the live key so a following request re-reads .env.local.
    global _live_cookies_key
    _live_cookies_key = None
    return {
        "linesRemoved": removed,
        "envDropped": dropped or "none",
    }


@app.get("/api/settings/cookies")
def get_cookies_key() -> dict[str, str]:
    """Read the current cookies values from .env.local (and the live key)."""
    if _live_cookies_key is not None:
        _live_cookies_key.refresh(force=True)
        key, source, problem = (
            _live_cookies_key.key,
            _live_cookies_key.source,
            _live_cookies_key.problem,
        )
    else:
        key = os.getenv("YT_DLP_COOKIES_FILE") or os.getenv("YT_DLP_COOKIES_FROM_BROWSER")
        source = "os-env" if key else "not found"
        problem = None
    return {
        "cookies_file": os.getenv("YT_DLP_COOKIES_FILE", "") or None,
        "cookies_from_browser": os.getenv("YT_DLP_COOKIES_FROM_BROWSER", "") or None,
        "source": source or None,
        "problem": problem,
    }
'''
