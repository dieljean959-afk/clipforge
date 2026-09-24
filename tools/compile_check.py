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

if failed:
    print(f"\n{failed} module(s) failed")
sys.exit(1 if failed else 0)
