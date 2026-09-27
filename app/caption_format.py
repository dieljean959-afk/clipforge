"""Post-caption formatting rules: one line of copy + 5-7 earned hashtags.

The model is asked for the right shape, but it is not trusted to *keep* it --
models happily return nine tags, two copies of the same tag, or a caption with
an embedded newline. Every rule below is therefore enforced here, locally, so
the shape holds even when a reply is sloppy. That also makes the rules
testable without a network round trip.
"""

from __future__ import annotations

import re

# Discovery tags earn reach; content tags earn the right audience. The mix is
# 2-3 discovery to 2-4 content. Ordered by how broadly each is understood.
DISCOVERY_TAGS: tuple[str, ...] = ("#fyp", "#foryou", "#viral")

MIN_TAGS = 5
MAX_TAGS = 7
MAX_DISCOVERY = 3
MIN_DISCOVERY = 2
# The spec is a MIX, not just a count: 2-3 discovery tags AND 2-4 content tags.
# normalize_hashtags can always reach 5 by adding more discovery tags, so the
# shortfall that actually matters is the CONTENT one.
MIN_CONTENT = 2
MAX_CONTENT = 4
MAX_CAPTION_CHARS = 150

_TAG_RE = re.compile(r"[^0-9a-z]+")


def clean_tag(raw: object) -> str:
    """Normalise one tag to a bare lowercase ``#word`` form.

    Strips the leading ``#`` if present, any punctuation, emoji and whitespace,
    so ``" #AI & ML "`` becomes ``"#aiml"`` and a non-tag degrades to ``""``.
    """
    text = str(raw or "").strip().lstrip("#").lower()
    text = _TAG_RE.sub("", text)
    return f"#{text}" if text else ""


def normalize_hashtags(
    tags: object,
    *,
    min_total: int = MIN_TAGS,
    max_total: int = MAX_TAGS,
) -> list[str]:
    """Return deduped, ordered tags honouring the discovery/content mix.

    Relevance over volume: an empty slot is never filled with a generic
    content tag. If the model produced too few, the shortfall is topped up from
    ``DISCOVERY_TAGS`` only, because those are the tags a short clip is allowed
    to borrow. A model that returns 3 solid content tags plus #fyp gets
    #foryou, #viral and #shorts rather than three invented filler tags.
    """
    if not isinstance(tags, (list, tuple)):
        return list(DISCOVERY_TAGS[:3])

    seen: set[str] = set()
    content: list[str] = []
    discovery: list[str] = []
    for raw in tags:
        tag = clean_tag(raw)
        if not tag or tag in seen:
            continue  # dedupe case-insensitively (clean_tag already lowercases)
        seen.add(tag)
        (discovery if tag in DISCOVERY_TAGS else content).append(tag)

    # A 4th discovery tag adds no reach that the first three did not cover.
    if len(discovery) > MAX_DISCOVERY:
        content = discovery[MAX_DISCOVERY:] + content
        discovery = discovery[:MAX_DISCOVERY]
    # 5 good content tags is one too many for the spec (2-4 content), and the
    # spare slot belongs to a discovery tag anyway.
    content = content[:MAX_CONTENT]

    for tag in DISCOVERY_TAGS:
        if len(discovery) + len(content) >= min_total and len(discovery) >= MIN_DISCOVERY:
            break
        if tag in seen:
            continue
        seen.add(tag)
        discovery.append(tag)

    return (discovery + content)[:max_total]


def shorten_caption(raw: object, *, limit: int = MAX_CAPTION_CHARS) -> str:
    """Collapse to a single line and trim on a word boundary.

    A caption that wraps to three lines in the platform composer is not "one
    line", so newlines and runs of whitespace become a single space *before* the
    length cut, otherwise the cut can leave a fragment mid-word.
    """
    text = re.sub(r"\s+", " ", str(raw or "")).strip()
    if len(text) <= limit:
        return text
    clipped = text[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:-–—")
    return clipped


def build_post_caption(caption: object, tags: object) -> str:
    """The exact text that gets copied to the clipboard or posted.

    One line of copy, a blank line, then the hashtag block. This is shared by
    "Copy caption" and every "Post to..." action so a user never has to re-type
    what they already copied.
    """
    line = shorten_caption(caption)
    block = " ".join(normalize_hashtags(tags))
    return f"{line}\n\n{block}".strip()


# --------------------------------------------------------------------------- #
# Content tags mined from the clip's own transcript
# --------------------------------------------------------------------------- #

_STOPWORDS = frozenset(
    """
    a about actually all also am an and any are as at back be because been
    before being but by can cant come could did do does doing dont down each
    even every for from get getting go goes going gonna good got great had
    has have having he her here hers him his how i id if im in into is it its
    ive just know knew like literally lot me mean more most much must my never
    new no not now ok okay on one only or other our out over own really right
    said same say saying she should so some still stuff such sure take tell
    than that the their them then there these they thing things think this
    those though to too up us use used using very was way we well were what
    when where which who why will with would you your youre
    """.split()
)

_WORD_RE = re.compile(r"[a-z0-9']+")

# A mined tag has to clear this bar or it is filler and is dropped. Anything
# rarer than a hapax in a ~50-word clip is as likely to be a transcription
# artefact ("um", "gonna") as a topic.
_MIN_COUNT = 2


def suggest_content_tags(
    transcript: object,
    *,
    limit: int = 4,
    min_count: int = _MIN_COUNT,
) -> list[str]:
    """Rank topic words from the clip's own transcript as candidate hashtags.

    The model supplies the tags it is confident about; this only covers the
    remainder, and only with words the clip actually says. A word must appear
    at least ``min_count`` times to qualify, which is what keeps one-off noise
    out -- "no filler" is enforced by the bar, not by hoping.
    """
    words = _WORD_RE.findall(str(transcript or "").lower())
    if not words:
        return []

    counts: dict[str, int] = {}
    for word in words:
        if len(word) < 4 or word in _STOPWORDS:
            continue
        counts[word] = counts.get(word, 0) + 1

    ranked = sorted(
        ((w, n) for w, n in counts.items() if n >= min_count),
        key=lambda item: (-item[1], -len(item[0]), item[0]),
    )
    return [f"#{w}" for w, _ in ranked[:limit]]


def finalize_clip_caption(
    caption: object,
    tags: object,
    transcript: object = "",
) -> tuple[str, list[str]]:
    """The single entry point: a 1-line caption and a 5-7 tag block.

    Model tags are kept in the order given (it chose them for a reason) and the
    transcript fills only the CONTENT shortfall, so the block is 2-3 discovery
    tags followed by as many genuinely on-topic content tags as exist. If a clip
    is too thin to fill the mix without padding, it returns fewer rather than
    inventing.
    """
    line = shorten_caption(caption)
    ordered = normalize_hashtags(tags)
    content = [t for t in ordered if t not in DISCOVERY_TAGS]
    if len(content) < MIN_CONTENT:
        present = set(ordered)
        for tag in suggest_content_tags(transcript, limit=MAX_CONTENT):
            if len(content) >= MAX_CONTENT or len(ordered) >= MAX_TAGS:
                break
            if tag in present:
                continue
            present.add(tag)
            ordered.append(tag)
            content.append(tag)
    return line, ordered[:MAX_TAGS]
