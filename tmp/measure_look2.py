import json
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/home/daytona/codebase")
CLIPS = Path("/tmp/lock_clips")


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)


def frame(p, t, out):
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.3f}", "-i", str(p), "-frames:v", "1", str(out)],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr[-500:]
    return Image.open(out).convert("RGB")


def mask(im):
    a = np.asarray(im)
    r, g, b = [a[:, :, i].astype(int) for i in range(3)]
    spread = a.max(2).astype(int) - a.min(2).astype(int)
    white = (r > 225) * (g > 225) * (b > 225) * (spread < 25)
    yellow = (r > 175) * (g > 175) * (b < 100) * ((r - b) > 100) * ((g - b) > 100)
    green = (g > 155) * (r < 100) * (b < 100)
    red = (r > 160) * (g < 90) * (b < 90)
    return (white + yellow + green + red) > 0


def groups_from_mask(m, x0=80, x1=1000, y0=1100, y1=1500, minpix=12):
    cols = m[y0:y1, x0:x1].sum(axis=1)
    rows = [y0 + i for i, n in enumerate(cols) if n >= minpix]
    gs = []
    for y in rows:
        if gs and y - gs[-1][1] <= 3:
            gs[-1] = (gs[-1][0], y)
        else:
            gs.append((y, y))
    return gs


def bar(im):
    a = np.asarray(im)
    white = (a[:, :, 0] > 245) * (a[:, :, 1] > 245) * (a[:, :, 2] > 245)
    edge = white[:, [0, 2, -3, -1]].all(axis=1)
    ys = np.where(edge[:500])[0]
    return [int(ys.min()), int(ys.max())] if len(ys) else None


def ass(p):
    text = Path(p).read_text()
    style = re.search(r"Style: ([^\n]+)", text).group(1)

    def sec(x):
        h, m, s = x.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    ev = []
    for line in text.splitlines():
        if line.startswith("Dialogue:"):
            q = line.split(",", 9)
            ev.append((sec(q[1]), sec(q[2]), q[9]))
    return style, ev


out = []
for p in [CLIPS / "clip_000.mp4", CLIPS / "clip_001.mp4", CLIPS / "clip_003.mp4"]:
    style, ev = ass(p.with_suffix(".ass"))
    cands = []
    for s, e, t in ev:
        act = [x for x in ev if x[0] <= s + 0.02 < x[1]]
        if len(act) >= 2:
            cands.append(round(s + 0.3, 2))
    t = next((x for x in sorted(set(cands)) if 1.0 < x < 8), sorted(set(cands))[0] if cands else 1.0)
    im = frame(p, t, CLIPS / f"{p.stem}_m2.png")
    m = mask(im)
    gs = groups_from_mask(m)
    gs = sorted(gs, key=lambda g: g[1] - g[0], reverse=True)[:2]
    gs = sorted(gs)
    y0 = gs[0][0] if gs else 0
    y1 = gs[-1][1] if gs else 0
    cols = np.where(m[y0:y1 + 1].sum(axis=0) >= 3)[0]
    width = int(cols.max() - cols.min() + 1) if len(cols) else 0
    a = np.asarray(im.convert("L"), dtype=np.float32)
    strips = np.concatenate([a[:, 0:150], a[:, 930:1080]], axis=1)
    sh = np.abs(np.diff(strips, axis=1)).mean(axis=1)
    top = max(range(500, 750), key=lambda y: abs(float(sh[y]) - float(sh[y - 1])))
    bot = max(range(1150, 1400), key=lambda y: abs(float(sh[y]) - float(sh[y - 1])))
    d = json.loads(run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(p)]).stdout)
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    au = next(s for s in d["streams"] if s["codec_type"] == "audio")
    lr = run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(p), "-af", "ebur128=peak=true", "-f", "null", "-"]).stderr
    integ = re.findall(r"I:\s*(-?[0-9.]+) LUFS", lr)
    peak = re.findall(r"Peak:\s*(-?[0-9.]+) dBFS", lr)
    out.append({
        "clip": p.name,
        "caption_time": t,
        "video": {k: v.get(k) for k in ("codec_name", "profile", "pix_fmt", "width", "height", "avg_frame_rate", "r_frame_rate")},
        "audio": {k: au.get(k) for k in ("codec_name", "sample_rate", "channels", "bit_rate")},
        "duration": d["format"].get("duration"),
        "bar": bar(im),
        "foreground_top": top,
        "foreground_bottom": bot,
        "caption_groups": gs,
        "caption_width": width,
        "loudness_lufs": integ[-1] if integ else None,
        "peak_dbfs": peak[-1] if peak else None,
        "ass_style": style,
    })
print(json.dumps(out, indent=2))
