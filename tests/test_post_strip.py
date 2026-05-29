"""Tests for the deterministic meta-opener post-strip in app/chat/answer.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.chat.answer import strip_meta_opener, strip_suggested_questions


def test_strips_stacked_meta_openers():
    text = "Now I have a clear picture. Let me answer. High Light wins on profit ($839 vs $379)."
    assert strip_meta_opener(text) == "High Light wins on profit ($839 vs $379)."


def test_clean_answer_unchanged():
    text = "High Light outperforms Shade on profit ($839 vs $379) and ROI (59% vs 31%)."
    assert strip_meta_opener(text) == text


def test_short_meta_only_answer_unchanged():
    # Removing the opener would leave nothing substantive -> keep original (safety).
    assert strip_meta_opener("Let me answer.") == "Let me answer."


def test_case_insensitive():
    text = "NOW I HAVE the data. Precision delivered $47,034 across all 120 plots."
    assert strip_meta_opener(text) == "Precision delivered $47,034 across all 120 plots."


def test_colon_delimited_opener():
    text = "Here's the comparison: High Light beats Shade on every financial metric."
    assert strip_meta_opener(text) == "High Light beats Shade on every financial metric."


def test_only_one_sentence_when_body_is_not_meta():
    # Strips just the single leading opener, never the substantive sentence after it.
    text = "Let me answer. Strawberry is the most profitable crop at $679 per plot."
    assert strip_meta_opener(text) == "Strawberry is the most profitable crop at $679 per plot."


def test_empty_and_none_safe():
    assert strip_meta_opener("") == ""
    assert strip_meta_opener(None) is None


# --- strip_suggested_questions (P2) ---

def test_strips_trailing_dodge_sentence():
    text = ("High Light is the most profitable treatment at $839 per plot. "
            "You could ask about per-crop differences for more detail.")
    out = strip_suggested_questions(text)
    assert out == "High Light is the most profitable treatment at $839 per plot."
    assert "you could ask" not in out.lower()


def test_strips_trailing_question_bullets():
    text = (
        "Strawberry is the most profitable crop at $679 per plot.\n\n"
        "What you can ask instead:\n"
        "- Which treatment works best for strawberry?\n"
        "- How does profit vary by region?\n"
    )
    out = strip_suggested_questions(text)
    assert out == "Strawberry is the most profitable crop at $679 per plot."
    assert "?" not in out
    assert "what you can ask" not in out.lower()


def test_strips_bare_question_bullets_without_header():
    text = (
        "Average VPD is 1.40 kPa, within the healthy range.\n"
        "- Would you like a per-crop breakdown?\n"
        "- Should I compare to the optimal band?"
    )
    out = strip_suggested_questions(text)
    assert out == "Average VPD is 1.40 kPa, within the healthy range."


def test_clean_answer_with_legit_question_unchanged():
    # A body question that is NOT a trailing dodge bullet/sentence stays put.
    text = "Is tomato yield above benchmark? No — 9.4 vs 60 kg/m2, so it trails substantially."
    assert strip_suggested_questions(text) == text


def test_suggested_strip_safety_short():
    # Stripping would leave too little -> keep original.
    assert strip_suggested_questions("You could ask about profit.") == "You could ask about profit."


def test_suggested_strip_empty_safe():
    assert strip_suggested_questions("") == ""
    assert strip_suggested_questions(None) is None
