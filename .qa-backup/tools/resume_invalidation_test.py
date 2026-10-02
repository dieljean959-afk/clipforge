"""Resume must not ship a stale render.

The pipeline resumes by reusing clip docs BY INDEX. That is only sound while
the new candidate still covers the same window. A re-detection is allowed to
return a different cut at the same index -- clip 3 is simply no longer the
same three seconds -- and if the old encode survives, the job reports success
while serving the previous video under the new metadata.

This is not hypothetical: the Groq re-run of job c57d7782c548 produced fresh
cuts, `create_clips` kept `renderState="done"`, all ten renders were skipped,
and the batch recheck found frame counts up to 2x off (clip 5 wanted 20.35s
and found 40.87s). Ten videos, ten wrong cuts, zero errors reported.

Run:  .venv/bin/python tools/resume_invalidation_test.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Paths  # noqa: E402
from app.store import LocalStore  # noqa: E402

FAILS: list[str] = []
CHECKS = 0


def check(ok: bool, label: str) -> None:
    global CHECKS
    CHECKS += 1
    if ok:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}")
        FAILS.append(label)


def candidate(start: float, end: float, score: float = 90.0) -> dict:
    return {"start": start, "end": end, "score": score, "title": f"clip {start}"}


def rendered_doc(**over) -> dict:
    """The render-complete half of a clip doc.

    Deliberately carries no _id: callers merge it INTO the doc create_clips
    handed back, so there is exactly one clip document and index-based reuse
    is actually being exercised.
    """
    doc = {
        "videoPath": "/outputs/clip_01.mp4",
        "thumbPath": "/outputs/clip_01.jpg",
        "mediaBytes": 1234,
        "mediaWidth": 1080,
        "mediaHeight": 1920,
        "mediaDurationSec": 20.0,
        "mediaError": None,
        "renderState": "done",
        "renderMs": 4000,
        "captionCues": [{"start": 0.0, "end": 1.0, "text": "hi"}],
        "silenceCuts": [[1.0, 2.0]],
        "cropBox": {"x": 1, "y": 2, "w": 3, "h": 4},
        "captionPreset": "lower_third",
    }
    doc.update(over)
    return doc


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = LocalStore(Paths(
            data_dir=root / "data",
            outputs_dir=root / "outputs",
            downloads_dir=root / "downloads",
            static_dir=root / "static",
        ))
        job_id = "jobtest000001"
        store.write_job({"_id": job_id, "source": "x", "title": "t", "stage": "rendering"})

        # ---- 1. identical cuts: the render must survive ----------------
        docs = store.create_clips(job_id, [candidate(142.1, 165.57)], "groq")
        store.write_clip({**docs[0], **rendered_doc()})

        same = store.create_clips(job_id, [candidate(142.1, 165.57)], "groq")
        check(len(store.list_clips(job_id)) == 1, "resume reuses one clip doc, not a new one")
        check(same[0].get("renderState") == "done", "unchanged cut keeps renderState=done")
        check(same[0].get("videoPath") == "/outputs/clip_01.mp4", "unchanged cut keeps videoPath")
        check(same[0].get("mediaDurationSec") == 20.0, "unchanged cut keeps measured duration")
        check(same[0].get("cropBox") == {"x": 1, "y": 2, "w": 3, "h": 4}, "unchanged cut keeps cropBox")
        check(same[0].get("captionCues") != [], "unchanged cut keeps caption cues")
        check(
            same[0].get("_id") == docs[0].get("_id"),
            "unchanged cut keeps the clip id (stable download links)",
        )

        # ---- 2. sub-frame float noise must NOT invalidate --------------
        noise = store.create_clips(job_id, [candidate(142.100001, 165.570001)], "groq")
        check(noise[0].get("renderState") == "done", "float noise (<20ms) still resumes")

        # ---- 3. the cut MOVED: the render must be invalidated ----------
        moved = store.create_clips(job_id, [candidate(440.23, 486.22)], "groq")
        check(moved[0].get("renderState") == "pending", "moved start invalidates the render")
        for field in ("videoPath", "thumbPath", "mediaBytes", "mediaWidth",
                      "mediaHeight", "mediaDurationSec", "renderMs", "cropBox",
                      "mediaError"):
            check(moved[0].get(field) is None, f"moved cut clears {field}")
        check(moved[0].get("captionCues") == [], "moved cut clears caption cues")
        check(moved[0].get("silenceCuts") == [], "moved cut clears silence cuts")
        check(
            abs(moved[0]["start"] - 440.23) < 1e-6 and abs(moved[0]["end"] - 486.22) < 1e-6,
            "moved cut stores the NEW window",
        )

        # ---- 4. only the tail moved: still invalid ---------------------
        store.write_clip({**moved[0], **rendered_doc()})
        tail = store.create_clips(job_id, [candidate(440.23, 500.0)], "groq")
        check(tail[0].get("renderState") == "pending", "moved end invalidates the render")

        # ---- 5. a failed render is retried -----------------------------
        store.write_clip({**tail[0], **rendered_doc(renderState="failed", mediaError="boom")})
        again = store.create_clips(job_id, [candidate(440.23, 500.0)], "groq")
        check(again[0].get("renderState") == "pending", "failed render stays retryable")
        check(again[0].get("mediaError") is None, "retry clears the old media error")

        # ---- 6. a stuck 'running' render is retried --------------------
        store.write_clip({**again[0], **rendered_doc(renderState="running")})
        stuck = store.create_clips(job_id, [candidate(440.23, 500.0)], "groq")
        check(stuck[0].get("renderState") == "pending", "interrupted 'running' render is retried")

        # ---- 7. fewer candidates than before: extras are dropped -------
        store.write_clip({**stuck[0], **rendered_doc()})
        shrunk = store.create_clips(job_id, [candidate(1.0, 5.0)], "groq")
        check(len(shrunk) == 1, "candidate list is authoritative for clip count")
        kept = [c for c in store.list_clips(job_id)]
        check(all(c.get("order") == 0 for c in kept), "no orphan clip docs keep a stale order")

        # ---- 8. persistence: the invalidation survives a reload --------
        store.write_clip({**shrunk[0], **rendered_doc()})
        store.create_clips(job_id, [candidate(999.0, 1010.0)], "groq")
        reloaded = store.list_clips(job_id)[0]
        check(reloaded.get("renderState") == "pending", "invalidation is persisted to disk")
        check(reloaded.get("videoPath") is None, "persisted doc has no stale videoPath")

        # ---- 9. the guard itself ---------------------------------------
        check(store._cut_moved({"start": 1.0, "end": 2.0}, {"start": 1.0, "end": 2.0}) is False,
              "_cut_moved: equal window")
        check(store._cut_moved({"start": 1.0, "end": 2.0}, {"start": 1.4, "end": 2.0}) is True,
              "_cut_moved: moved start")
        check(store._cut_moved({"start": 1.0, "end": 2.0}, {"start": 1.0, "end": 2.4}) is True,
              "_cut_moved: moved end")
        check(store._cut_moved({}, {"start": 1.0, "end": 2.0}) is True,
              "_cut_moved: missing prior cut is treated as moved")
        check(store._cut_moved({"start": "x", "end": 2.0}, {"start": 1.0, "end": 2.0}) is True,
              "_cut_moved: unparseable prior cut is treated as moved")

    print()
    if FAILS:
        print(f"{CHECKS - len(FAILS)} passed, {len(FAILS)} FAILED")
        return 1
    print(f"{CHECKS} passed, 0 failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
