"""Public entry point for the GreenLeaf chat.

As of Stage 3 v2 the single-shot router was replaced by an agentic tool-calling
loop (see app/chat/agent.py). This module is the public entry point:

    from app.chat.answer import answer   # used by the Streamlit UI

It calls the agent loop and then applies one deterministic post-processing step —
strip_meta_opener() — to the final answer text only, because the prompt-only rule
forbidding process-narration openers ("Now I have…", "Let me…") kept slipping.
Sources and trace are returned untouched.

answer(question, max_iterations=6) -> {answer, sources, trace}.
"""
import re

from app.chat import agent

__all__ = ["answer", "strip_meta_opener", "strip_suggested_questions"]

# Leading phrases that signal process narration rather than the answer itself.
_META_OPENERS = (
    "now i have",
    "i now have",
    "let me",
    "let's",
    "here's",
    "here is",
    "based on my analysis",
    "i notice",
    "looking at",
    "to answer",
)

# A "sentence" for opener-detection ends at the first period, newline, or colon.
_SENTENCE_END = re.compile(r"[.\n:]")
_MIN_KEEP = 20  # never strip down to an empty/too-short answer (safety)


def _split_first_sentence(text):
    """Return (first_sentence_incl_delimiter, rest)."""
    m = _SENTENCE_END.search(text)
    if not m:
        return text, ""
    return text[: m.end()], text[m.end():]


def strip_meta_opener(text):
    """Remove leading process-narration sentence(s) so the answer starts with the
    finding. Strips ONE leading sentence per pass (never body content), repeating
    only to clear stacked openers like 'Now I have… Let me…'. If a removal would
    leave fewer than 20 chars of substance, it is not made (the original stands)."""
    if not text:
        return text
    result = text
    while True:
        stripped = result.lstrip()
        first, rest = _split_first_sentence(stripped)
        if not any(first.strip().lower().startswith(p) for p in _META_OPENERS):
            break
        candidate = rest.lstrip()
        if len(candidate) < _MIN_KEEP:
            break  # safety: don't nuke a short answer that is mostly the opener
        result = candidate
    return result


# Phrases that signal a dodge / "ask instead" suggestion at the END of an answer.
_DODGE_PREFIXES = (
    "you could ask",
    "you would need to ask",
    "you would need",
    "a good follow-up question",
    "to answer your question, i would need",
    "consider asking",
    "you could rephrase",
    "to determine",
    "or provide",
)
# Headers that introduce a suggestion block; cut the header line and everything after.
_DODGE_HEADERS = (
    "what you can ask instead",
    "what you could ask",
    "questions you could ask",
    "you might also ask",
)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*)$")


def strip_suggested_questions(text):
    """Remove a trailing dodge / 'ask instead' block from the answer TAIL only.

    Strips: a suggestion header (and everything after it), trailing lines that start
    with a dodge phrase, and a trailing bulleted/numbered list of questions ('...?')
    or 'provide X' requests. Never touches the substantive body; if stripping would
    leave <20 chars, the original is returned unchanged."""
    if not text:
        return text
    original = text

    # 1) Cut a suggestion header and everything after it.
    low = text.lower()
    for header in _DODGE_HEADERS:
        idx = low.find(header)
        if idx != -1:
            line_start = text.rfind("\n", 0, idx)
            text = text[:line_start] if line_start != -1 else text[:idx]
            break

    # 2) Drop trailing blank lines, dodge-prefixed lines, and question/provide bullets.
    def _strippable(line):
        s = line.strip()
        if not s:
            return True
        low_s = s.lower()
        if any(low_s.startswith(p) for p in _DODGE_PREFIXES):
            return True
        m = _BULLET.match(s)
        if m:
            body = m.group(1).strip().lower()
            if body.endswith("?") or body.startswith("provide") or body.startswith("or provide"):
                return True
        return False

    lines = text.split("\n")
    while lines and _strippable(lines[-1]):
        lines.pop()

    # 3) Drop trailing dodge SENTENCES from the last remaining line.
    if lines:
        sentences = re.split(r"(?<=[.!?])\s+", lines[-1])
        while len(sentences) > 1 and any(
            sentences[-1].strip().lower().startswith(p) for p in _DODGE_PREFIXES
        ):
            sentences.pop()
        lines[-1] = " ".join(sentences)

    result = "\n".join(lines).rstrip()
    if len(result) < 20:           # safety: never nuke the answer
        return original
    return result


def answer(question, max_iterations=6):
    """Run the agent loop, then strip a meta-narration opener and any trailing
    suggested-question dodge from the answer text (sources/trace untouched)."""
    result = agent.answer(question, max_iterations=max_iterations)
    if isinstance(result, dict) and isinstance(result.get("answer"), str):
        cleaned = strip_meta_opener(result["answer"])
        cleaned = strip_suggested_questions(cleaned)
        result["answer"] = cleaned
    return result
