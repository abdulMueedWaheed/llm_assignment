import json
from collections import Counter
from pathlib import Path
from typing import Iterable, TypeVar

import regex

from bpe_tokens import (
    CACHE_LIMIT,
    FIRST_MERGE_ID,
    SPECIAL_ID,
    SPECIAL_TEXT,
    URDU_WORD_PATTERN,
    merge_pair_left_to_right,
    open_text_file,
)
from bpe_training import TrainingMixin


class CorpusMixin:
    def __init__(self) -> None:
        self.word_pattern = regex.compile(URDU_WORD_PATTERN)
        self.merges: list[tuple[int, int]] = []
        self.vocabulary = self.make_byte_vocabulary()
        self.inverse_vocabulary = {
            value: key for key, value in self.vocabulary.items()
        }
        self.merge_rank: dict[tuple[int, int], int] = {}
        self.word_cache: dict[bytes, list[int]] = {}
        self.training_stats: dict[str, int] = {}

    @staticmethod
    def make_byte_vocabulary() -> dict[int, bytes]:
        vocabulary = {index: bytes([index]) for index in range(256)}
        vocabulary[SPECIAL_ID] = SPECIAL_TEXT.encode("utf-8")
        return vocabulary

    def read_words(
        self,
        path: str | Path,
        max_bytes: int | None = None,
    ) -> Iterable[str]:
        bytes_read = 0
        with open_text_file(path) as input_file:
            for line in input_file:
                bytes_read += len(line.encode("utf-8", errors="surrogatepass"))
                for text_part in line.split(SPECIAL_TEXT):
                    for match in self.word_pattern.finditer(text_part):
                        yield match.group()
                if max_bytes is not None and bytes_read >= max_bytes:
                    break

    def count_words(
        self,
        path: str | Path,
        max_bytes: int | None = None,
    ) -> Counter[str]:
        return Counter(self.read_words(path, max_bytes))

    def rebuild_vocabulary(self) -> None:
        self.vocabulary = self.make_byte_vocabulary()
        self.inverse_vocabulary = {}
        for token_id, token_bytes in self.vocabulary.items():
            self.inverse_vocabulary.setdefault(token_bytes, token_id)

        self.merge_rank = {}
        for merge_number, pair in enumerate(self.merges):
            new_id = FIRST_MERGE_ID + merge_number
            left_id, right_id = pair
            if SPECIAL_ID in pair:
                raise ValueError("the special token cannot be merged")
            if left_id not in self.vocabulary or right_id not in self.vocabulary:
                raise ValueError(f"merge {merge_number + 1} uses an unknown token")

            merged_bytes = self.vocabulary[left_id] + self.vocabulary[right_id]
            self.vocabulary[new_id] = merged_bytes
            self.inverse_vocabulary.setdefault(merged_bytes, new_id)
            self.merge_rank[pair] = new_id

        self.word_cache.clear()


class EncodingMixin:
    def encode_word(self, word_bytes: bytes) -> list[int]:
        token_ids = list(word_bytes)

        while len(token_ids) > 1:
            available_pairs = [
                pair for pair in zip(token_ids, token_ids[1:])
                if pair in self.merge_rank
            ]
            if not available_pairs:
                break
            best_pair = min(available_pairs, key=self.merge_rank.get)
            token_ids = merge_pair_left_to_right(
                token_ids,
                best_pair,
                self.merge_rank[best_pair],
            )

        return token_ids

    def encode_text_part(self, text: str) -> list[int]:
        token_ids = []
        for match in self.word_pattern.finditer(text):
            word_bytes = match.group().encode("utf-8", errors="surrogatepass")
            cached_ids = self.word_cache.get(word_bytes)

            if cached_ids is None:
                cached_ids = self.encode_word(word_bytes)
                if len(self.word_cache) < CACHE_LIMIT:
                    self.word_cache[word_bytes] = cached_ids

            token_ids.extend(cached_ids)

        return token_ids

    def encode(self, text: str) -> list[int]:
        token_ids = []
        text_parts = text.split(SPECIAL_TEXT)
        for part_number, text_part in enumerate(text_parts):
            if part_number:
                token_ids.append(SPECIAL_ID)
            if text_part:
                token_ids.extend(self.encode_text_part(text_part))
        return token_ids

    def decode(self, token_ids: Iterable[int]) -> str:
        token_bytes = b"".join(self.vocabulary[token_id] for token_id in token_ids)
        try:
            return token_bytes.decode("utf-8", errors="surrogatepass")
        except UnicodeDecodeError:
            return token_bytes.decode("utf-8", errors="replace")


SerializationType = TypeVar("SerializationType", bound="SerializationMixin")


class SerializationMixin:
    def save(self, path: str | Path) -> None:
        data = {
            "version": 2,
            "vocab_size": FIRST_MERGE_ID + len(self.merges),
            "num_merges": len(self.merges),
            "pattern": self.word_pattern.pattern,
            "special_tokens": {SPECIAL_TEXT: SPECIAL_ID},
            "train_stats": self.training_stats,
            "merges": [[left, right] for left, right in self.merges],
        }
        with open(path, "w", encoding="utf-8", newline="\n") as output_file:
            json.dump(data, output_file, ensure_ascii=True)

    @classmethod
    def load(
        cls: type[SerializationType],
        path: str | Path,
    ) -> SerializationType:
        with open(path, "r", encoding="utf-8") as input_file:
            data = json.load(input_file)

        if data.get("special_tokens") != {SPECIAL_TEXT: SPECIAL_ID}:
            raise ValueError("tokenizer has an invalid special-token mapping")

        tokenizer = cls()
        tokenizer.word_pattern = regex.compile(data["pattern"])
        tokenizer.merges = [tuple(pair) for pair in data["merges"]]

        expected_size = FIRST_MERGE_ID + len(tokenizer.merges)
        if data.get("vocab_size") != expected_size:
            raise ValueError("tokenizer vocabulary size does not match its merges")
        if data.get("num_merges", len(tokenizer.merges)) != len(tokenizer.merges):
            raise ValueError("tokenizer merge count does not match its merge list")

        tokenizer.training_stats = data.get("train_stats", {})
        tokenizer.rebuild_vocabulary()
        return tokenizer


class BPETokenizer(
    CorpusMixin,
    TrainingMixin,
    EncodingMixin,
    SerializationMixin,
):
    pass
