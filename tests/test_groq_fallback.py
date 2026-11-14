from app import groq_asr


def test_word_only_reply_becomes_timed_segments():
    # Current Groq replies to timestamp_granularities[]=word with `words` and
    # no `segments`; treating that as empty silently lost a whole transcript.
    segs = groq_asr.segments_from_words([
        {"word": "hola", "start": 1.0, "end": 1.4},
        {"word": "mundo", "start": 1.4, "end": 1.9},
        {"word": "", "start": 2.0, "end": 2.1},
    ])
    assert [s["text"] for s in segs] == ["hola", "mundo"]
    assert segs[1]["start"] == 1.4


def test_chunk_plan_uses_two_second_overlap():
    assert groq_asr.OVERLAP_SEC == 2.0
    windows = groq_asr.plan_chunks(25.0, chunk_sec=20.0, overlap_sec=2.0)
    assert windows[0] == (0.0, 20.0)
    assert windows[1] == (18.0, 25.0)
    assert windows[-1][1] == 25.0
