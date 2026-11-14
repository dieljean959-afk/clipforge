#!/usr/bin/env python3
from __future__ import annotations
import ast
from pathlib import Path

W = Path('app/worker.py')
lines = W.read_text(encoding='utf-8').splitlines()

start_idx = None
end_idx = None
for idx, line in enumerate(lines):
    if start_idx is None and line.startswith('# Consecutive resumes that produced NO'):
        start_idx = idx
    if start_idx is not None and line.startswith('# An attempt that lived at least'):
        end_idx = idx
        break

if start_idx is None or end_idx is None:
    print('region not found', start_idx, end_idx)
    raise SystemExit(1)

replacement = [
    '# Consecutive resumes that produced NO additional finished units. Restarting',
    '# only helps while each attempt gets further than the last; once it does not,',
    '# the loop is not converging and the job must stop and say so instead of',
    '# burning the host forever.',
    '#',
    '# Set to 0 (default) to never hard-fail a job purely because the host kept',
    '# restarting — every interrupted render-stage job is re-queued and resumes',
    '# from the next unfinished clip. Increase only if you want a safety cap for',
    '# pathological restart loops on a specific deployment.',
    'STALLED_RESUME_LIMIT = int(os.getenv("CLIPFORGE_STALLED_RESUME_LIMIT", "0"))',
    '',
]

out = lines[:start_idx] + replacement + lines[end_idx:]
W.write_text('\n'.join(out) + '\n', encoding='utf-8')
ast.parse(W.read_text(encoding='utf-8'))
print('worker.py parsed ok')
