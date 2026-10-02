#!/usr/bin/env python3
"""Env-file manager: set / delete / list variables in .env.local.

The platform's Keys/API keys UI currently has no Edit or Delete for existing
variables (it errors with "Variable already exists. Use edit instead" while
offering no edit control). This tool provides the missing operations directly
against the project env file the backend reads.

Usage:
  .venv/bin/python tools/envmanage.py set NAME VALUE      # add or replace
  .venv/bin/python tools/envmanage.py delete NAME [NAME…] # remove lines
  .venv/bin/python tools/envmanage.py list                # names + shape only

`list` NEVER prints values — only presence, length, and first/last 4 chars
shape info so pastes can be verified without exposing secrets.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ENV_LOCAL = Path(__file__).resolve().parent.parent / ".env.local"

# Vars kept even if the user asks to delete them (platform-managed).
PROTECTED = {
    "DOTENV_PUBLIC_KEY_LOCAL", "DOTENV_PRIVATE_KEY_LOCAL",
    "VITE_CONVEX_URL", "CONVEX_DEPLOYMENT", "VITE_CONVEX_SITE_URL",
    "VITE_VLY_APP_ID", "VITE_VLY_MONITORING_URL",
}


def read_lines() -> list[str]:
    if not ENV_LOCAL.exists():
        return []
    return ENV_LOCAL.read_text(encoding="utf-8", errors="replace").splitlines()


def write_lines(lines: list[str]) -> None:
    tmp = ENV_LOCAL.with_suffix(".local.tmp")
    tmp.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
    tmp.replace(ENV_LOCAL)


def shape(value: str) -> str:
    if not value:
        return "(empty)"
    if value.startswith("encrypted:"):
        return f"(encrypted blob, {len(value)} chars)"
    return f"{len(value)} chars, starts {value[:6]!r}, ends {value[-4:]!r}"


def parse_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    name, _, value = stripped.partition("=")
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return name.strip(), value


def cmd_set(name: str, value: str) -> int:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        print(f"invalid variable name: {name!r}")
        return 1
    lines = read_lines()
    replaced = False
    out: list[str] = []
    for line in lines:
        parsed = parse_line(line)
        if parsed and parsed[0] == name:
            if not replaced:
                out.append(f"{name}={value}")
                replaced = True
            # drop duplicate definitions of the same name
            continue
        out.append(line)
    if not replaced:
        out.append(f"{name}={value}")
    write_lines(out)
    print(f"set {name}: {shape(value)}")
    return 0


def cmd_delete(names: list[str]) -> int:
    names = [n for n in names if n not in PROTECTED]
    refused = [n for n in names if n in PROTECTED]
    if refused:
        print(f"protected (skipped): {', '.join(refused)}")
    if not names:
        print("nothing to delete")
        return 1
    lines = read_lines()
    out: list[str] = []
    removed: dict[str, int] = {}
    for line in lines:
        parsed = parse_line(line)
        if parsed and parsed[0] in names:
            removed[parsed[0]] = removed.get(parsed[0], 0) + 1
            continue
        out.append(line)
    write_lines(out)
    for n in names:
        print(f"deleted {n}: {removed.get(n, 0)} line(s) removed")
    return 0


def cmd_list() -> int:
    lines = read_lines()
    seen: dict[str, str] = {}
    for line in lines:
        parsed = parse_line(line)
        if parsed:
            seen[parsed[0]] = parsed[1]
    for name, value in sorted(seen.items()):
        print(f"{name}: {shape(value)}")
    print(f"\n{len(seen)} variable(s) in {ENV_LOCAL.name}")
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    cmd = sys.argv[1]
    if cmd == "set" and len(sys.argv) == 4:
        return cmd_set(sys.argv[2], sys.argv[3])
    if cmd == "delete" and len(sys.argv) >= 3:
        return cmd_delete(sys.argv[2:])
    if cmd == "list" and len(sys.argv) == 2:
        return cmd_list()
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
