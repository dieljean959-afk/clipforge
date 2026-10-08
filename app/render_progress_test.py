#!/usr/bin/env python3
import re
from pathlib import Path

root = Path('app/video.py')
src = root.read_text(encoding='utf-8')

prog = re.compile(
    r"^frame=(\d+)\r?$\n"
    r".*?time=([0-9.]+)\r?$\n"
    r".*?fps=([0-9.]+)",
    re.MULTILINE,
)

sample = "frame=123\r\n" "some noise\r\n" "time=4.567\r\n" "fps=29.97\r\n"
m = prog.match(sample)
assert m, "progress regex did not match sample"
assert int(m.group(1)) == 123
assert float(m.group(2)) == 4.567
assert float(m.group(3)) == 29.97
print('ffmpeg progress regex matches')
