from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TypedDict
import json

import tiktoken

from bpe import SPECIAL_TEXT, UrduBPETokenizer


class TokenMetrics(TypedDict):
    words: int
    counts: dict[str, int]
    ratios: dict[str, float]


HAND_INPUTS = [
    ("News", "حکومت نے آج نئے منصوبے کا اعلان کیا، جس سے شہریوں کو سہولت ملے گی۔"),
    ("Poetry", "بارش آئی، صحن میں خوشبو پھیل گئی\nخاموش رات میں ایک نئی امید کھل گئی"),
    ("Urdu with English", "آج team نے project update کیا اور نئی deadline جمعہ کو رکھی۔"),
    ("Roman Urdu", "Aaj mausam bohat khubsurat hai, chalo Islamabad chalte hain."),
    ("Numbers and dates", "قیمت ۱۲۵۰ روپے ہے؛ تاریخ 11-10-2026 اور وقت 9:45 بجے ہے۔"),
    ("Diacritics", "اَللّٰهُ أَكْبَرُ، اُردُو کے اعراب اور حرکات"),
    ("Personal names", "عائشہ، عبدالرحمٰن، ژوبیر اور مریم آج لاہور پہنچے۔"),
]

BAD_CASE_REASONS = {
    "News": "The sentence's specific vocabulary may be rare in the training corpus.",
    "Poetry": "Poetic wording and inflections may be less frequent than prose.",
    "Urdu with English": "English spans use a different byte distribution and may be sparse in mixed contexts.",
    "Roman Urdu": "Romanized spellings vary and do not share the Urdu-script merges learned by this tokenizer.",
    "Numbers and dates": "Digits and punctuation form combinations that are not reusable word-like units.",
    "Diacritics": "Combining marks add UTF-8 bytes and are relatively sparse, so words may remain split.",
    "Personal names": "Proper names are often rare or have spelling variants, limiting name-specific merges.",
}


def markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "\\n").replace("\r", "\\r")


def render_token(token_bytes: bytes) -> str:
    try:
        return json.dumps(token_bytes.decode("utf-8"), ensure_ascii=False)
    except UnicodeDecodeError:
        return f"invalid UTF-8: {token_bytes!r}"


def iter_documents(path: Path) -> Iterator[str]:
    buffer: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as source:
        for line in source:
            parts = line.split(SPECIAL_TEXT)
            for part_number, part in enumerate(parts):
                buffer.append(part)
                if part_number < len(parts) - 1:
                    yield "".join(buffer)
                    buffer.clear()
    if buffer:
        yield "".join(buffer)


def measure_text(
    text: str,
    tokenizer: UrduBPETokenizer,
    encodings: Mapping[str, tiktoken.Encoding],
) -> TokenMetrics:
    word_count = len(text.split())
    token_counts = {"Urdu BPE": len(tokenizer.encode(text))}
    token_counts.update(
        {name: len(encoding.encode_ordinary(text)) for name, encoding in encodings.items()}
    )
    return {
        "words": word_count,
        "counts": token_counts,
        "ratios": {
            name: count / max(word_count, 1)
            for name, count in token_counts.items()
        },
    }


def measure_test_file(
    path: Path,
    tokenizer: UrduBPETokenizer,
    encodings: Mapping[str, tiktoken.Encoding],
) -> TokenMetrics:
    word_count = 0
    token_counts = {"Urdu BPE": 0, **{name: 0 for name in encodings}}
    for document in iter_documents(path):
        metrics = measure_text(document, tokenizer, encodings)
        word_count += metrics["words"]
        for name, count in metrics["counts"].items():
            token_counts[name] += count
    return {
        "words": word_count,
        "counts": token_counts,
        "ratios": {
            name: count / max(word_count, 1)
            for name, count in token_counts.items()
        },
    }
