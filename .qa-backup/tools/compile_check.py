#!/usr/bin/env python3
"""Compile-check every backend module without importing heavy deps.

Also runs an AST-level undefined-name scan per module: any name that is
*referenced* but never *defined/imported* in the file (and is not a builtin)
would raise NameError at runtime mid-pipeline — exactly the `_probe_duration`
failure class — so it fails the check here, at compile time.
"""
from __future__ import annotations

import ast
import builtins
import py_compile
import sys
from pathlib import Path

_BUILTINS = set(dir(builtins))
# Names injected by the runtime that AST cannot see.
_ALLOWED = {"__name__", "__file__", "__doc__", "__package__", "__annotations__"}


def _collect_bound(tree: ast.Module) -> set[str]:
    """Names the module defines, imports, or assigns — conservatively, at any
    scope (a name defined in any function is not a module-level NameError when
    referenced from elsewhere in that same function)."""
    bound: set[str] = set(_BUILTINS) | _ALLOWED
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.alias):
            bound.add(node.asname or node.name.split(".")[0])
        elif isinstance(node, (ast.ExceptHandler,)):
            if node.name:
                bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
        elif isinstance(node, ast.Attribute):
            pass  # attribute access is resolved at runtime, not a NameError
    return bound


def _undefined_names(tree: ast.Module) -> list[str]:
    bound = _collect_bound(tree)
    bad: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in bound:
                bad.add(node.id)
    return sorted(bad)


failed = 0
for p in sorted(Path("app").glob("*.py")):
    try:
        py_compile.compile(str(p), doraise=True)
    except Exception as exc:
        failed += 1
        print(f"FAIL {p}: {exc}")
        continue
    try:
        tree = ast.parse(p.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        failed += 1
        print(f"FAIL {p}: syntax error: {exc}")
        continue
    undefined = _undefined_names(tree)
    if undefined:
        failed += 1
        print(f"FAIL {p}: referenced but never defined/imported: {', '.join(undefined)}")
        continue
    print(f"OK   {p}")

# ---------------------------------------------------------------------------
# IMPORT the app, not just compile it.
#
# Compiling proves the file parses. It says nothing about whether the names
# resolve at runtime, which is exactly the gap that let a broken import ship:
#
#     from fastapi.background import BackgroundTask   # ImportError
#
# compiled fine, passed the undefined-name AST scan, passed every other suite —
# and made the whole backend fail to start, so the site served 500 on every
# request. Only actually importing `app.main` and building the FastAPI object
# catches that class of mistake.
# ---------------------------------------------------------------------------
try:
    import contextlib
    import importlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        main_mod = importlib.import_module("app.main")
    app_obj = getattr(main_mod, "app", None)
    if app_obj is None:
        failed += 1
        print("FAIL app.main imported but exposes no `app` object")
    else:
        routes = len(getattr(app_obj, "routes", []))
        if routes < 10:
            failed += 1
            print(f"FAIL app.main built an app with only {routes} routes")
        else:
            print(f"OK   app.main imports and builds the ASGI app ({routes} routes)")
except Exception as exc:  # noqa: BLE001 - this is the whole point of the check
    failed += 1
    print(f"FAIL app.main could not be imported: {type(exc).__name__}: {exc}")

if failed:
    print(f"\n{failed} module(s) failed")
sys.exit(1 if failed else 0)
