from pathlib import Path
from typing import TypedDict
import json

import regex

from bpe import FIRST_MERGE_ID, SPECIAL_TEXT, UrduBPETokenizer
from analysis_metrics import render_token


class MergeAnalysis(TypedDict):
    rows: list[tuple[int, str, int, str, str]]
    two_byte_letters: list[tuple[int, str]]
    invalid_early: list[int]
    longest: list[tuple[int, int, str]]
    first_whole_word: tuple[int, str] | None


def first_whole_word(
    train_path: Path,
    tokenizer: UrduBPETokenizer,
) -> tuple[int, str] | None:
    lexical_word = regex.compile(r" ?[\p{L}\p{M}\u200c]+")
    candidates: dict[bytes, int] = {}
    for merge_number, pair in enumerate(tokenizer.merges, start=1):
        token_bytes = tokenizer.vocabulary[pair[0]] + tokenizer.vocabulary[pair[1]]
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
    selected: set[int] = set(range(1, min(50, len(tokenizer.merges)) + 1))
    selected.update(range(5_000, len(tokenizer.merges) + 1, 5_000))
    if tokenizer.merges:
        selected.add(len(tokenizer.merges))

    rows = []
    two_byte_letters = []
    invalid_early = []
    for merge_number, pair in enumerate(tokenizer.merges, start=1):
        token_bytes = tokenizer.vocabulary[pair[0]] + tokenizer.vocabulary[pair[1]]
        try:
            decoded_text = token_bytes.decode("utf-8")
            valid_utf8 = True
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
            rows.append(
                (
                    merge_number,
                    f"{pair[0]} + {pair[1]}",
                    len(token_bytes),
                    "yes" if valid_utf8 else "no",
                    render_token(token_bytes),
                )
            )

    longest = [
        (token_id, len(token_bytes), render_token(token_bytes))
        for token_id, token_bytes in tokenizer.vocabulary.items()
        if token_id >= FIRST_MERGE_ID
    ]
    longest.sort(key=lambda item: (-item[1], item[0]))

    return {
        "rows": rows,
        "two_byte_letters": two_byte_letters,
        "invalid_early": invalid_early,
        "longest": longest[:20],
        "first_whole_word": first_whole_word(train_path, tokenizer),
    }
