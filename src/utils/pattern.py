import re

HEADING_PATTERNS = [
    # 1. Numbered or enumerated heading (numeric or lettered lists with optional dot/parenthesis)
    re.compile(r'^\s*(?:\d+(?:\.\d+)*[\.)]?|[A-Z]+[\.)])\s+[A-Z][^.]*$'),
    # 2. General heading: starts with capital, no ending punctuation
    re.compile(r'^\s*[A-Z][^.!?]*[^.!?\s]$'),
    # 3. "Chapter/Section" heading with number (digits or Roman numeral) and optional title
    re.compile(
        r'^\s*(?:Chapter|Section)\s+(?:\d+|[IVXLCDM]+)'
        r'(?:(?:[\.\-:]\s*|\s+)[^.!?]+[^.!?\s])?$',
        re.IGNORECASE
    )
]

WHITESPACE_ONLY_PATTERN = re.compile(r"^\s*$")
BULLET_PATTERN = re.compile(r"^\s*([\-•\*]|\d+[\.\)])\s+")
SPLIT_PATTERN = re.compile(r"(?<=[\.\?\!…])\s+")


__all__ = ["HEADING_PATTERNS", "WHITESPACE_ONLY_PATTERN", "BULLET_PATTERN", "SPLIT_PATTERN"]