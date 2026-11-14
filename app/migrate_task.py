#!/usr/bin/env python3
"""Patch app/task.py so _render_one_clip has a parseable body, then verify imports."""
from __future__ import annotations
import ast
from pathlib import Path

TASK = Path('app/task.py')

BODY = [
    '        last: Exception | None = None',
    '        for attempt in (1, 2, 3):',
    '            try:',
    '                with self._heartbeat(job_id, f"render clip {number}"):',
    '                    meta = video.render_clip(',
    '                        src, out_path,',
    '                        start=start, end=end,',
    '                        settings=self.settings,',
    '                        ass_path=ass_path,',
    '                        caption_cues=cues,',
    '                        crop_windows=crop_windows,',
    '                        silence_cuts=silence_cuts,',
    '                        src_fps=src_fps,',
    '                        safe=attempt >= 2,',
    '                        _progress_callback=_progress_callback,',
    '                    )',
    '                if not isinstance(meta, dict):',
    '                    meta = {}',
    '                video.verify_output(out_path, expected_sec=end - start)',
    '                if attempt >= 2:',
    '                    self._log(',
    '                        job_id,',
    '                        f"clip {number} of {total}: attempt {attempt - 1} failed "',
    '                        f"({last}); attempt {attempt} with safer settings produced "',
    '                        f"a usable file",',
    '                    )',
    '                return meta',
    '            except Exception as exc:  # noqa: BLE001',
    '                last = exc',
    '                try:',
    '                    out_path.unlink(missing_ok=True)',
    '                except OSError:',
    '                    pass',
    '                if attempt < 3:',
    '                    self._log(',
    '                        job_id,',
    '                        f"clip {number} of {total} failed on attempt {attempt}/3 "',
    '                        f"({type(exc).__name__}: {exc}) — retrying attempt "',
    '                        f"{attempt + 1}/3 with safer settings",',
    '                    )',
    '                    self._progress(',
    '                        job_id, "rendering",',
    '                        (number - 1) / max(1, total),',
    '                        f"Clip {number} failed, retrying {attempt + 1}/3 with safer settings",',
    '                        unit_label="clips", units_done=number - 1,',
    '                        units_total=total,',
    '                    )',
    '        raise video.RenderError(str(last) if last else "render failed after 3 attempts")',
]


def patch() -> None:
    lines = TASK.read_text(encoding='utf-8').splitlines()
    out: list[str] = []
    in_func = False
    func_close_seen = False
    inserted = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if not in_func and line.startswith('    def _render_one_clip('):
            in_func = True
            out.append(line)
            i += 1
            continue
        if in_func and not func_close_seen:
            out.append(line)
            if line.strip() == '"""':
                func_close_seen = True
                out.extend(BODY)
                inserted = True
                i += 1
                continue
            i += 1
            continue
        out.append(line)
        i += 1

    if not inserted:
        raise SystemExit('BODY INSERTION NOT DONE')

    TASK.write_text('\n'.join(out) + '\n', encoding='utf-8')
    ast.parse(TASK.read_text(encoding='utf-8'))


if __name__ == '__main__':
    patch()
    print('task.py parsed ok')
