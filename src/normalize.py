"""
normalize.py
Urdu Unicode text normalizer for CS 417 Assignment 1.
Handles character unification, removes non-printing characters,
standardizes digits, and cleans up whitespace.
"""

import re
import unicodedata

# Character replacement mapping based on assignment requirements
CHAR_REPLACEMENTS = {
    # Yeh variants -> Urdu Yeh (U+06CC)
    '\u064A': '\u06CC',  # Arabic Yeh (ي) -> Urdu Yeh (ی)
    '\u0649': '\u06CC',  # Alef Maksura (ى) -> Urdu Yeh (ی)
    '\u06D2': '\u06D2',  # Bari Yeh (ے) - keep distinct
    
    # Kaf variants -> Urdu Kaf (U+06A9)
    '\u0643': '\u06A9',  # Arabic Kaf (ك) -> Urdu Kaf (ک)
    
    # Heh variants: map Arabic heh to Urdu gol heh, keep do-chashmi heh separate
    '\u0647': '\u06C1',  # Arabic Heh (ه) -> Urdu Gol Heh (ہ)
    # Note: Do-chashmi Heh (ھ, U+06BE) is kept as is because it represents aspiration

    # Tatweel / Kashida -> remove
    '\u0640': '',

    # Non-printing and directional marks -> remove
    '\uFEFF': '',        # BOM
    '\u200B': '',        # Zero width space
    '\u200D': '',        # ZWJ
    '\u200E': '',        # LTR mark
    '\u200F': '',        # RTL mark
    '\u202A': '',        # LRE
    '\u202B': '',        # RLE
    '\u202C': '',        # PDF
    '\u202D': '',        # LRO
    '\u202E': '',        # RLO
    '\u061C': '',        # Arabic letter mark
    # Note: ZWNJ (\u200C) is intentionally preserved for Urdu compound words
    
    # Digits: map Arabic and Urdu digits to standard 0-9
    '\u0660': '0', '\u0661': '1', '\u0662': '2', '\u0663': '3', '\u0664': '4',
    '\u0665': '5', '\u0666': '6', '\u0667': '7', '\u0668': '8', '\u0669': '9',
    '\u06F0': '0', '\u06F1': '1', '\u06F2': '2', '\u06F3': '3', '\u06F4': '4',
    '\u06F5': '5', '\u06F6': '6', '\u06F7': '7', '\u06F8': '8', '\u06F9': '9',
}

TRANSLATION_TABLE = str.maketrans(CHAR_REPLACEMENTS)

# Whitespace cleanup patterns
RE_SPACES = re.compile(r'[\t\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000 ]+')
RE_NEWLINES = re.compile(r'\r\n|\r')
RE_MULTI_NEWLINES = re.compile(r'\n{3,}')
RE_LINE_TRAILING_WS = re.compile(r' +(?=\n)')
RE_LINE_LEADING_WS = re.compile(r'(?<=\n) +')

def normalize(text: str) -> str:
    """
    Normalizes a string of Urdu text:
    - Unicode NFC normalization
    - Fixes common keyboard mismatches (Arabic yeh, kaf, heh)
    - Strips tatweel and invisible formatting characters (keeps ZWNJ)
    - Standardizes numerals to ASCII 0-9
    - Cleans up spaces and extra blank lines
    """
    if not text:
        return ""

    # 1. Unicode NFC
    text = unicodedata.normalize('NFC', text)

    # 2. Character replacements
    text = text.translate(TRANSLATION_TABLE)

    # 3. Normalize newlines to standard \n
    text = RE_NEWLINES.sub('\n', text)

    # 4. Collapse consecutive horizontal spaces into one space
    text = RE_SPACES.sub(' ', text)

    # 5. Remove spaces at line boundaries
    text = RE_LINE_TRAILING_WS.sub('', text)
    text = RE_LINE_LEADING_WS.sub('', text)

    # 6. Keep at most two newlines for paragraph breaks
    text = RE_MULTI_NEWLINES.sub('\n\n', text)

    return text.strip()


if __name__ == "__main__":
    # Test cases to verify normalization behavior
    test_cases = [
        ("علي و موسى", "علی و موسی"),
        ("كتاب", "کتاب"),
        ("مكه اور گھر", "مکہ اور گھر"),
        ("تـــــعــــلـــیـــم", "تعلیم"),
        ("بے\u200cشک \uFEFFاور ساتھ\u200D", "بے\u200cشک اور ساتھ"),
        ("سال ۱۹۹۸ اور سنہ ۱۲۳۴ اور 1234", "سال 1998 اور سنہ 1234 اور 1234"),
        ("یہ    ایک   ٹیسٹ    ہے۔\n\n\n\nدوسرا پیراگراف۔", "یہ ایک ٹیسٹ ہے۔\n\nدوسرا پیراگراف۔"),
    ]

    print("Running normalization tests...")
    all_ok = True
    for idx, (raw, expected) in enumerate(test_cases, 1):
        result = normalize(raw)
        if result == expected:
            print(f"  Test {idx}: OK")
        else:
            print(f"  Test {idx}: FAILED")
            print(f"    Got:      {repr(result)}")
            print(f"    Expected: {repr(expected)}")
            all_ok = False

    if all_ok:
        print("All tests passed.")
    else:
        print("Some tests failed.")
