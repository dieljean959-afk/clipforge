#!/usr/bin/env python3
from __future__ import annotations
import ast
from pathlib import Path

p = Path('app/worker.py')
lines = p.read_text(encoding='utf-8').splitlines()

out: list[str] = []
i = 0
while i < len(lines):
    line = lines[i]
    if line.startswith('    def _note_interrupted(self, job:'):
        out.append('    def _note_interrupted(self, job: dict[str, Any] | None, job_id: str, rc: int,')
        out.append('                          stage: str | None, tail: str) -> bool:')
        i += 1
        continue
    out.append(line)
    i += 1

p.write_text('\n'.join(out) + '\n', encoding='utf-8')
ast.parse(p.read_text(encoding='utf-8'))
print('worker.py parses')
