import argparse
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TypedDict

import regex
import tiktoken

from bpe import FIRST_MERGE_ID, SPECIAL_TEXT, UrduBPETokenizer


class TokenMetrics(TypedDict):
    words: int
    counts: dict[str, int]
    ratios: dict[str, float]


class MergeAnalysis(TypedDict):
    rows: list[tuple[int, str, int, str, str]]
    two_byte_letters: list[tuple[int, str]]
    invalid_early: list[int]
    longest: list[tuple[int, int, str]]
    first_whole_word: tuple[int, str] | None


HAND_INPUTS = [
    (
        "News",
        "حکومت نے آج نئے منصوبے کا اعلان کیا، جس سے شہریوں کو سہولت ملے گی۔",
    ),
    (
        "Poetry",
        "بارش آئی، صحن میں خوشبو پھیل گئی\nخاموش رات میں ایک نئی امید کھل گئی",
    ),
    (
        "Urdu with English",
        "آج team نے project update کیا اور نئی deadline جمعہ کو رکھی۔",
    ),
    (
        "Roman Urdu",
        "Aaj mausam bohat khubsurat hai, chalo Islamabad chalte hain.",
    ),
    (
        "Numbers and dates",
        "قیمت ۱۲۵۰ روپے ہے؛ تاریخ 11-10-2026 اور وقت 9:45 بجے ہے۔",
    ),
    (
        "Diacritics",
        "اَللّٰهُ أَكْبَرُ، اُردُو کے اعراب اور حرکات",
    ),
    (
        "Personal names",
        "عائشہ، عبدالرحمٰن، ژوبیر اور مریم آج لاہور پہنچے۔",
    ),
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


def first_whole_word(
    train_path: Path,
    tokenizer: UrduBPETokenizer,
) -> tuple[int, str] | None:
    lexical_word = regex.compile(r" ?[\p{L}\p{M}\u200c]+")
    candidates: dict[bytes, int] = {}
    for merge_number, pair in enumerate(tokenizer.merges, start=1):
        token_bytes = (
            tokenizer.vocabulary[pair[0]] + tokenizer.vocabulary[pair[1]]
        )
        try:
            word = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if lexical_word.fullmatch(word) and len(regex.findall(r"\p{L}", word)) >= 2:
            candidates.setdefault(token_bytes, merge_number)

    if not candidates:
        return None

    lowest_candidate = min(candidates.values())
    best_candidate: tuple[int, str] | None = None
    with train_path.open("r", encoding="utf-8", newline="") as source:
        for line in source:
            for text_part in line.split(SPECIAL_TEXT):
                for match in tokenizer.word_pattern.finditer(text_part):
                    word = match.group()
                    merge_number = candidates.get(word.encode("utf-8"))
                    if merge_number is not None and (
                        best_candidate is None or merge_number < best_candidate[0]
                    ):
                        best_candidate = (merge_number, word)
                        if merge_number == lowest_candidate:
                            return best_candidate
    return best_candidate


def merge_analysis(tokenizer: UrduBPETokenizer, train_path: Path) -> MergeAnalysis:
    merge_rows: list[tuple[int, str, int, str, str]] = []
    selected: set[int] = set(range(1, min(50, len(tokenizer.merges)) + 1))
    selected.update(range(5_000, len(tokenizer.merges) + 1, 5_000))
    if tokenizer.merges and len(tokenizer.merges) not in selected:
        selected.add(len(tokenizer.merges))

    two_byte_letters: list[tuple[int, str]] = []
    invalid_early: list[int] = []
    for merge_number, pair in enumerate(tokenizer.merges, start=1):
        token_bytes = (
            tokenizer.vocabulary[pair[0]] + tokenizer.vocabulary[pair[1]]
        )
        valid_utf8 = True
        try:
            decoded_text = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            decoded_text = None
            valid_utf8 = False

        if merge_number <= 50:
            if not valid_utf8:
                invalid_early.append(merge_number)
            elif (
                decoded_text is not None
                and len(token_bytes) == 2
                and regex.fullmatch(r"\p{L}", decoded_text)
            ):
                two_byte_letters.append((merge_number, decoded_text))

        if merge_number in selected:
            merge_rows.append(
                (
                    merge_number,
                    f"{pair[0]} + {pair[1]}",
                    len(token_bytes),
                    "yes" if valid_utf8 else "no",
                    render_token(token_bytes),
                )
            )

    longest: list[tuple[int, int, str]] = []
    for token_id, token_bytes in tokenizer.vocabulary.items():
        if token_id >= FIRST_MERGE_ID:
            longest.append((token_id, len(token_bytes), render_token(token_bytes)))
    longest.sort(key=lambda item: (-item[1], item[0]))

    first_word = first_whole_word(train_path, tokenizer)
    return {
        "rows": merge_rows,
        "two_byte_letters": two_byte_letters,
        "invalid_early": invalid_early,
        "longest": longest[:20],
        "first_whole_word": first_word,
    }


def create_report(
    tokenizer_path: Path,
    tokenizer: UrduBPETokenizer,
    merge_data: MergeAnalysis,
    test_metrics: TokenMetrics,
    hand_metrics: list[tuple[str, str, TokenMetrics]],
) -> str:
    final_model = len(tokenizer.merges) == 49_743
    status = "Final 50,000-token tokenizer" if final_model else "PRELIMINARY trial tokenizer"
    lines = [
        "# Phase 3: Tokenizer Analysis",
        "",
        f"**Status:** {status} ({len(tokenizer.merges) + FIRST_MERGE_ID:,} vocabulary entries).",
        f"**Tokenizer:** `{tokenizer_path}`. "
        + (
            "This report uses the required final vocabulary size."
            if final_model
            else "Regenerate with the final 50,000-token model before submission."
        ),
        "",
        "## 1. Merge Analysis",
        "",
        "| Merge | Input IDs | Bytes | Valid UTF-8 alone | Decoded token |",
        "|---:|---|---:|:---:|---|",
    ]
    for merge_number, pair, byte_length, valid_utf8, decoded in merge_data["rows"]:
        lines.append(
            f"| {merge_number:,} | {pair} | {byte_length} | {valid_utf8} | {markdown_cell(decoded)} |"
        )

    letter_rows = ", ".join(
        f"{merge_number}: {json.dumps(letter, ensure_ascii=False)}"
        for merge_number, letter in merge_data["two_byte_letters"]
    ) or "None in the first 50 merges."
    invalid_rows = ", ".join(map(str, merge_data["invalid_early"])) or "None in the first 50 merges."
    first_word = merge_data["first_whole_word"]
    whole_word_text = (
        f"Merge {first_word[0]:,} first forms the multi-letter corpus word {json.dumps(first_word[1], ensure_ascii=False)}."
        if first_word
        else "No whole lexical word was found in the available training text."
    )
    lines.extend(
        [
            "",
            f"Two-byte Urdu letters rebuilt in the first 50: {letter_rows}",
            f"Early merges invalid as standalone UTF-8 (first 50): {invalid_rows}",
            whole_word_text,
            "",
            "### 20 Longest Learned Tokens",
            "",
            "| Token ID | UTF-8 bytes | Token |",
            "|---:|---:|---|",
        ]
    )
    for token_id, byte_length, decoded in merge_data["longest"]:
        lines.append(f"| {token_id} | {byte_length} | {markdown_cell(decoded)} |")

    lines.extend(
        [
            "",
            "## 2. Tokens per Word",
            "",
            "Word counts use whitespace-delimited words. `<|endoftext|>` separators are excluded from the test-corpus comparison.",
            "",
            "| Input | Words | Urdu BPE tokens/word | GPT-2 tokens/word | o200k_base tokens/word |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    all_metrics = [("test.txt", test_metrics)] + [
        (name, metrics) for name, _, metrics in hand_metrics
    ]
    for name, metrics in all_metrics:
        ratios = metrics["ratios"]
        lines.append(
            f"| {markdown_cell(name)} | {metrics['words']:,} | "
            f"{ratios['Urdu BPE']:.3f} | {ratios['GPT-2']:.3f} | {ratios['o200k_base']:.3f} |"
        )
    lines.extend(
        [
            "",
            (
                f"On test.txt, this tokenizer used {test_metrics['ratios']['Urdu BPE']:.3f} tokens per word, "
                f"compared with {test_metrics['ratios']['GPT-2']:.3f} for GPT-2 and "
                f"{test_metrics['ratios']['o200k_base']:.3f} for o200k_base. "
                "Roman Urdu, rare spellings, and mixed-script inputs remain useful stress cases."
            ),
            "",
            "## 3. Inputs Handled Poorly",
            "",
            "The following are the three hand-picked inputs with the highest Urdu-BPE token/word ratio:",
            "",
            "| Input type | Urdu BPE tokens/word | Example | Interpretation |",
            "|---|---:|---|---|",
        ]
    )
    worst = sorted(
        hand_metrics,
        key=lambda item: item[2]["ratios"]["Urdu BPE"],
        reverse=True,
    )[:3]
    for name, text, metrics in worst:
        lines.append(
            f"| {name} | {metrics['ratios']['Urdu BPE']:.3f} | "
            f"{markdown_cell(json.dumps(text, ensure_ascii=False))} | "
            f"{BAD_CASE_REASONS[name]} |"
        )
    lines.extend(
        [
            "",
            "These are small diagnostic examples, not a representative evaluation set. Replace them with the final tokenizer results before submitting.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    project_folder = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Analyze a trained Urdu BPE tokenizer.")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "tokenizer.json",
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=project_folder / "data" / "processed" / "train.txt",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=project_folder / "data" / "processed" / "test.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_folder / "reports" / "phase3_analysis.md",
    )
    args = parser.parse_args()

    for path, label in (
        (args.tokenizer, "tokenizer"),
        (args.train, "training corpus"),
        (args.test, "test corpus"),
    ):
        if not path.is_file():
            parser.error(f"{label} file does not exist: {path}")

    tokenizer = UrduBPETokenizer.load(args.tokenizer)
    merge_data = merge_analysis(tokenizer, args.train)
    encodings = {
        "GPT-2": tiktoken.get_encoding("gpt2"),
        "o200k_base": tiktoken.get_encoding("o200k_base"),
    }
    test_metrics = measure_test_file(args.test, tokenizer, encodings)
    hand_metrics = [
        (name, text, measure_text(text, tokenizer, encodings))
        for name, text in HAND_INPUTS
    ]
    report = create_report(
        args.tokenizer,
        tokenizer,
        merge_data,
        test_metrics,
        hand_metrics,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Vocabulary size: {len(tokenizer.vocabulary):,}")
    print(f"Test words: {test_metrics['words']:,}")
    print(f"Analysis report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
