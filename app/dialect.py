"""Spanish dialect vocabulary for Caribbean and Andean speech.

Two consumers, one source of truth:

  * the Whisper `prompt` parameter, which biases the acoustic decoder toward
    the spellings a Spanish-dominant ASR model gets wrong, and
  * the cleanup model's system prompt, which teaches it the same vocabulary
    plus what each word MEANS.

The single most important constraint in this file: **Groq caps the Whisper
`prompt` at 224 tokens.** Send more and the request is rejected outright, so
`whisper_prompt()` must always fit. It builds the list dialect-FIRST and
appends custom glossary terms only while they fit, because a custom brand name
the user explicitly typed is worth more than the twentieth built-in slang
word, and a rejected request costs the whole chunk.
"""

from __future__ import annotations

#: The dialects the UI offers. "caribbean_mix" is the honest default for a
#: creator whose audience is not one country — Dominican and Puerto Rican
#: speech share most of this vocabulary anyway.
DIALECTS: dict[str, str] = {
    "auto": "Neutral Latin American Spanish",
    "dominican": "Dominican Spanish (Santo Domingo)",
    "puerto_rican": "Puerto Rican Spanish",
    "colombian": "Colombian Spanish",
    "caribbean_mix": "Caribbean mix — Dominican and Puerto Rican with Spanglish",
}

DEFAULT_DIALECT = "caribbean_mix"

# --------------------------------------------------------------------------- #
# Built-in vocabulary, per dialect.
#
# Words are written the way they are SPOKEN, because the Whisper prompt is a
# spelling hint: "ta'" and "tamo'" teach the decoder what an aspirated 'd'
# looks like, which is the single biggest source of Caribbean ASR errors.
# --------------------------------------------------------------------------- #

DOMINICAN = (
    "pa'", "pa' alla", "tu ta'", "tamo'", "klk", "que lo que", "dime a ver",
    "tigre", "vaina", "chinas", "jevi", "motoconcho", "guagua", "colmado",
    "dique", "mi hermano", "muchacho", "compe", "yolo", "zafacón",
)

PUERTO_RICAN = (
    "wepa", "boricua", "nene", "nena", "janguear", "chavos", "guagua",
    "pana", "brutal", "ay bendito", "acho", "coqui", "bregar", "choretón",
    "lambío", "tíguere", "mapo", "chinchorro",
)

COLOMBIAN = (
    "parce", "parcero", "parche", "chimba", "bacano", "que nota", "mono",
    "vaina", "listo", "de una", "pilas", "tinto", "rumbear", "paila",
    "gonorrea", "guaro", "berraco", "man", "chevere", "con todo",
)

SPANGLISH = (
    "parquear", "troca", "lonche", "bro", "boss", "parking", "cornflake",
    "zafacón", "friki", "chulo", "rango", "guachimán",
)

BY_DIALECT: dict[str, tuple[str, ...]] = {
    "dominican": DOMINICAN,
    "puerto_rican": PUERTO_RICAN,
    "colombian": COLOMBIAN,
}

#: Shared by every dialect — the Spanglish and generic Caribbean core.
COMMON = ("klk", "vaina", "guagua", "pana", "parce", "bro", "chimba", "tigre")

#: Groq rejects a `prompt` longer than this. Kept as a constant so the test
#: suite can assert the builder against the real documented ceiling instead
#: of a number someone typed twice.
GROQ_PROMPT_TOKEN_LIMIT = 224

#: Words per "token" for a rough estimate. Deliberately pessimistic: under-
#: estimating tokens keeps us inside the limit; over-estimating would silently
#: drop custom glossary terms the user explicitly asked for.
_CHARS_PER_TOKEN = 3.5


def dialect_label(dialect: str) -> str:
    """Human label for a dialect id; unknown ids fall back to the mix."""
    return DIALECTS.get(dialect, DIALECTS[DEFAULT_DIALECT])


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def split_terms(raw: str) -> list[str]:
    """Split a user-typed glossary into terms.

    Separators (comma, semicolon, newline) are the PRIMARY split, and each
    piece is kept WHOLE — because the point of this field is names and brands,
    and "Don Tato" or "Casa Bonita" must survive as units. Splitting those on
    spaces leaves the cleanup model hunting for two unrelated words.

    A glossary with NO separator at all is ambiguous, and resolved by word
    count: up to three words is treated as ONE term (someone typing "Casa
    Bonita" deserves one entry, not two), and four or more is treated as a
    bare space-separated list, because pasting from a phone keyboard routinely
    produces "tigre pana parce guacha" and one enormous term is useless.

    Order is preserved and duplicates dropped: a repeated term wastes prompt
    budget that a real term could have used.
    """
    if not raw or not raw.strip():
        return []
    words = raw.split()
    has_sep = any(ch in raw for ch in ",;\n")
    if has_sep:
        chunks = raw.replace(";", ",").replace("\n", ",").split(",")
    elif len(words) <= 3:
        chunks = [" ".join(words)]
    else:
        chunks = words
    out: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        term = " ".join(chunk.split()).strip(" .\"'")
        if not term:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(term)
    return out


def vocabulary(dialect: str, glossary: str = "") -> list[str]:
    """Built-in terms for `dialect`, then the user's own, in that order.

    Built-ins come first on purpose: they are the ones a generic Spanish ASR
    model reliably mangles, and they are what makes the dialect switch do
    anything at all. Custom terms follow so a brand or a name the user cares
    about is still heard when the budget allows.
    """
    base = list(COMMON)
    for term in BY_DIALECT.get(dialect, ()):
        if term.lower() not in {t.lower() for t in base}:
            base.append(term)
    if dialect in {"caribbean_mix", "auto"}:
        for source in (DOMINICAN, PUERTO_RICAN, SPANGLISH):
            for term in source:
                if term.lower() not in {t.lower() for t in base}:
                    base.append(term)
    for term in split_terms(glossary):
        if term.lower() not in {t.lower() for t in base}:
            base.append(term)
    return base


def whisper_prompt(dialect: str, glossary: str = "", limit: int = GROQ_PROMPT_TOKEN_LIMIT) -> str:
    """A `prompt` string guaranteed to fit Groq's ceiling.

    Over-long prompts are REJECTED (HTTP 400), not truncated server-side, so
    one generous user glossary would otherwise cost every chunk of their job.

    Each term is added by MEASURING THE STRING THAT WOULD RESULT, not by
    summing per-term estimates. An earlier version did the arithmetic up front
    and forgot that the " Vocabulario: " marker and the joining separators
    count too — it emitted a 233-token prompt against a 224-token limit, which
    is exactly the failure this function exists to prevent.

    Terms are added in priority order (built-ins first) and the walk STOPS at
    the first term that does not fit, rather than continuing to look for a
    shorter one: a later custom term is worth less than an earlier built-in,
    and searching would also make the kept set depend on term lengths.
    """
    head = f"Transcripción de {dialect_label(dialect)}."
    marker = " Vocabulario: "
    kept: list[str] = []
    for term in vocabulary(dialect, glossary):
        candidate = head + marker + ", ".join([*kept, term])
        if estimate_tokens(candidate) > limit:
            break
        kept.append(term)
    if not kept:
        # Never send an empty prompt: a bare instruction with no vocabulary is
        # worse than useless, it just spends the parameter.
        return head
    return head + marker + ", ".join(kept)


def cleanup_hint(dialect: str, glossary: str = "") -> str:
    """The dialect paragraph appended to the cleanup model's system prompt.

    Unlike the Whisper prompt this has no hard token ceiling — DeepSeek and
    the other text models take long system prompts fine — so it carries the
    FULL vocabulary plus the user's own terms.
    """
    terms = vocabulary(dialect, glossary)
    lines = [
        "",
        f"DIALECT FOR THIS JOB: {dialect_label(dialect)}.",
        "",
        "Vocabulary the transcriber is likely to get wrong. Treat these as",
        "CORRECT SPELLINGS of what was probably said — do not replace them:",
        "".join(f"- {t}" for t in terms),
    ]
    custom = split_terms(glossary)
    if custom:
        lines += [
            "",
            "Names, brands and terms the user supplied. Spell them EXACTLY as",
            "written here whenever the speech is anywhere near them:",
            "".join(f"- {t}" for t in custom),
        ]
    return "\n".join(lines)
