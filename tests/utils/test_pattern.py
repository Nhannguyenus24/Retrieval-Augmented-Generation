# tests/test_utils/test_pattern.py
import pytest
import re

# adjust the import path according to your project structure
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.pattern import (
    HEADING_PATTERNS,
    WHITESPACE_ONLY_PATTERN,
    BULLET_PATTERN,
    SPLIT_PATTERN,
)

def _matches_any(patterns, s: str) -> bool:
    return any(p.match(s) for p in patterns)

# -----------------------------
# HEADING_PATTERNS
# -----------------------------

@pytest.mark.parametrize(
    "text",
    [
        # 1) Numbered/enumerated heading
        "1 Introduction",
        "1.2 Overview",
        "10.3.7 Deep Dive",
        "2) Methods",
        "A) Appendix",
        "B. Background",  # [A-Z]+[\.)] allows "B." before title
        "IV) Roman style",
        # 2) General heading (Capitalized, no terminal punctuation)
        "Introduction to RAG",
        "FastAPI Best Practices",
        "Chunking Heuristics And Guidelines",
        # 3) Chapter/Section with number/Roman + optional title
        "Chapter 1 Introduction",
        "Section 10: Results",
        "section IV – Methods",   # IGNORECASE + dash
        "Chapter 3.2 Advanced Topics",
        "Chap 3",
    ],
)
def test_heading_patterns_positive(text):
    assert _matches_any(HEADING_PATTERNS, text), f"Should match: {text!r}"

@pytest.mark.parametrize(
    "text",
    [
        # missing title after number/symbol
        "1.",
        "2)  ",
        "A)     ",
        # ending with punctuation => not a general heading #2
        "Heading.",
        "Heading!",
        "Heading?",
        "Heading…",
        # not capitalized first letter (not matching rule #2)
        "not capitalized heading",
        # Chapter/Section without number
        "Chapter",
        "Section",
        # Chapter with spelled-out number (pattern currently not supported)
        "Chapter One",
        # unusual leading character
        "# Heading",
    ],
)
def test_heading_patterns_negative(text):
    assert not _matches_any(HEADING_PATTERNS, text), f"Should NOT match: {text!r}"

# -----------------------------
# WHITESPACE_ONLY_PATTERN
# -----------------------------

@pytest.mark.parametrize("text", ["", "   ", "\t", "\n", "\r\n", "\t \n  "])
def test_whitespace_only_positive(text):
    assert bool(WHITESPACE_ONLY_PATTERN.match(text)), f"Should be whitespace-only: {text!r}"

@pytest.mark.parametrize("text", [" a ", ".", " - ", "0", " \n x"])
def test_whitespace_only_negative(text):
    assert not bool(WHITESPACE_ONLY_PATTERN.match(text)), f"Should NOT be whitespace-only: {text!r}"

# -----------------------------
# BULLET_PATTERN
# -----------------------------

@pytest.mark.parametrize(
    "text",
    [
        "- item",
        "* item",
        "• item",
        "   -   item",
        "1. item",
        "2) item",
        "\t10.   item",
    ],
)
def test_bullet_pattern_positive(text):
    m = BULLET_PATTERN.match(text)
    assert m is not None, f"Should detect bullet: {text!r}"

@pytest.mark.parametrize(
    "text",
    [
        "-- not a bullet (double dash without space)",
        "** no space after star",
        "•no-space",
        "1.item",    # missing space after dot
        "2)item",    # missing space after parenthesis
        "+ item",    # not in the allowed symbol set
        "A. item",   # bullet pattern does not allow letter + '.' (this is more like a heading)
        " item",     # only space, no bullet
    ],
)
def test_bullet_pattern_negative(text):
    assert BULLET_PATTERN.match(text) is None, f"Should NOT detect bullet: {text!r}"

# -----------------------------
# SPLIT_PATTERN (sentence split)
# -----------------------------

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Hello world! How are you? I'm fine… Thanks.",
            ["Hello world!", "How are you?", "I'm fine…", "Thanks."],
        ),
        (
            "One. Two. Three.",
            ["One.", "Two.", "Three."],
        ),
        (
            "No split here",
            ["No split here"],
        ),
        (
            "Edge…case… with ellipsis… Done.",
            ["Edge…case…", "with ellipsis…", "Done."],
        ),
    ],
)
def test_split_pattern_basic(text, expected):
    parts = SPLIT_PATTERN.split(text)
    assert parts == expected

def test_split_pattern_preserves_punctuation():
    text = "Is it ok? Yes! Great."
    parts = SPLIT_PATTERN.split(text)
    # make sure punctuation stays at the end of the previous sentence
    assert parts == ["Is it ok?", "Yes!", "Great."]

# -----------------------------
# Sanity checks (types/flags)
# -----------------------------

def test_patterns_are_compiled_regex():
    assert all(hasattr(p, "pattern") and isinstance(p, re.Pattern) for p in HEADING_PATTERNS)
    assert isinstance(WHITESPACE_ONLY_PATTERN, re.Pattern)
    assert isinstance(BULLET_PATTERN, re.Pattern)
    assert isinstance(SPLIT_PATTERN, re.Pattern)
