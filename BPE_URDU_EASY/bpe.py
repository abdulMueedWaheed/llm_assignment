import argparse
from collections import Counter
from collections import defaultdict
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import time
from typing import Callable, Iterable, Mapping

import regex

from output_utils import Tee, show_title, show_value


SPECIAL_TEXT = "<|endoftext|>"
SPECIAL_ID = 256
FIRST_MERGE_ID = 257
CACHE_LIMIT = 5_000_000

URDU_WORD_PATTERN = (
    r"""'s|'t|'re|'ve|'m|'ll|'d"""
    r"""| ?[\p{L}\p{M}\u200c]+"""
    r"""| ?\p{N}+"""
    r"""| ?[^\s\p{L}\p{M}\p{N}\u200c]+"""
    r"""|\s+(?!\S)"""
    r"""|\s+"""
)


def open_text_file(path: str | Path):
    return open(path, "r", encoding="utf-8", newline="")


def merge_pair_left_to_right(
    token_ids: list[int],
    pair: tuple[int, int],
    new_id: int,
) -> list[int]:
    left_id, right_id = pair
    merged_ids = []
    position = 0

    while position < len(token_ids):
        is_pair = (
            position + 1 < len(token_ids)
            and token_ids[position] == left_id
            and token_ids[position + 1] == right_id
        )
        if is_pair:
            merged_ids.append(new_id)
            position += 2
        else:
            merged_ids.append(token_ids[position])
            position += 1

    return merged_ids


class UrduBPETokenizer:
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

                # split special markers out so they cannot be merged
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

    def train_reference(
        self,
        word_occurrences: Iterable[str],
        vocabulary_size: int = 50_000,
        log_every: int = 1_000,
        log: Callable[[str], object] = print,
    ) -> list[tuple[int, int]]:
        # the reference version stores every word occurrence and recounts
        # every pair over the whole corpus after each merge
        self.check_training_settings(vocabulary_size, log_every)
        running_words = [
            list(word.encode("utf-8", errors="surrogatepass"))
            for word in word_occurrences
        ]
        self.training_stats = {
            "distinct_total": 0,
            "distinct_used": 0,
            "occurrences_total": len(running_words),
            "occurrences_used": len(running_words),
            "min_freq": 1,
        }

        byte_vocabulary = self.make_byte_vocabulary()
        learned_merges = []
        start_time = time.perf_counter()
        merge_count = vocabulary_size - FIRST_MERGE_ID

        for merge_number in range(merge_count):
            pair_counts: Counter[tuple[int, int]] = Counter()

            # this full recount is what makes the reference version simple
            for word in running_words:
                pair_counts.update(zip(word, word[1:]))

            if not pair_counts:
                break

            best_pair = max(
                pair_counts,
                key=lambda pair: (
                    pair_counts[pair],
                    byte_vocabulary[pair[0]],
                    byte_vocabulary[pair[1]],
                    pair,
                ),
            )
            new_id = FIRST_MERGE_ID + merge_number
            running_words = [
                merge_pair_left_to_right(word, best_pair, new_id)
                for word in running_words
            ]
            byte_vocabulary[new_id] = (
                byte_vocabulary[best_pair[0]] + byte_vocabulary[best_pair[1]]
            )
            learned_merges.append(best_pair)

            if log_every and (merge_number + 1) % log_every == 0:
                elapsed = time.perf_counter() - start_time
                log(
                    f"[reference] {merge_number + 1:,}/{merge_count:,} merges"
                    f" | best pair count: {pair_counts[best_pair]:,}"
                    f" | elapsed: {elapsed:.0f}s"
                )

        self.merges = learned_merges
        self.rebuild_vocabulary()
        return self.merges

    def train_fast(
        self,
        word_frequencies: Mapping[str, int],
        vocabulary_size: int = 50_000,
        min_frequency: int = 1,
        log_every: int = 1_000,
        log: Callable[[str], object] = print,
        checkpoint: Callable[[list[tuple[int, int]]], object] | None = None,
        verify_counts: bool = False,
    ) -> list[tuple[int, int]]:
        # the fast version keeps each different word once, but weights its
        # pair counts by how often that word appeared in the input
        self.check_training_settings(vocabulary_size, log_every)
        if min_frequency < 1:
            raise ValueError("minimum frequency must be at least 1")
        if any(
            not isinstance(count, int) or count < 1
            for count in word_frequencies.values()
        ):
            raise ValueError("word frequencies must be positive integers")

        total_occurrences = sum(word_frequencies.values())
        used_words = [
            (word, count)
            for word, count in word_frequencies.items()
            if count >= min_frequency
        ]
        used_occurrences = sum(count for _, count in used_words)
        self.training_stats = {
            "distinct_total": len(word_frequencies),
            "distinct_used": len(used_words),
            "occurrences_total": total_occurrences,
            "occurrences_used": used_occurrences,
            "min_freq": min_frequency,
        }
        log(
            f"Words kept: {len(used_words):,}/{len(word_frequencies):,} distinct"
            f" ({100 * used_occurrences / max(total_occurrences, 1):.2f}%"
            " of occurrences)"
        )

        running_words = [
            list(word.encode("utf-8", errors="surrogatepass"))
            for word, _ in used_words
        ]
        frequencies = [count for _, count in used_words]
        del used_words

        learned_merges = self.learn_fast_merges(
            running_words,
            frequencies,
            vocabulary_size - FIRST_MERGE_ID,
            log_every,
            log,
            checkpoint,
            verify_counts,
        )
        self.merges = learned_merges
        self.rebuild_vocabulary()
        return self.merges

    @staticmethod
    def check_training_settings(vocabulary_size: int, log_every: int) -> None:
        if vocabulary_size <= FIRST_MERGE_ID or vocabulary_size > 65_536:
            raise ValueError("vocabulary size must be between 258 and 65,536")
        if log_every < 0:
            raise ValueError("log interval cannot be negative")

    def learn_fast_merges(
        self,
        running_words: list[list[int]],
        frequencies: list[int],
        merge_count: int,
        log_every: int,
        log: Callable[[str], object],
        checkpoint: Callable[[list[tuple[int, int]]], object] | None,
        verify_counts: bool,
    ) -> list[tuple[int, int]]:
        # pair_counts stores total pair counts; pair_to_words says which
        # distinct words contain each pair. no heap is used in this version.
        pair_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
        pair_to_words: defaultdict[tuple[int, int], set[int]] = defaultdict(set)

        for word_id, word in enumerate(running_words):
            for pair in zip(word, word[1:]):
                pair_counts[pair] += frequencies[word_id]
                pair_to_words[pair].add(word_id)

        byte_vocabulary = self.make_byte_vocabulary()
        learned_merges = []
        start_time = time.perf_counter()

        for merge_number in range(merge_count):
            if not pair_counts:
                break

            # choose the highest count; byte values break ties consistently
            best_pair = max(
                pair_counts,
                key=lambda pair: (
                    pair_counts[pair],
                    byte_vocabulary[pair[0]],
                    byte_vocabulary[pair[1]],
                    pair,
                ),
            )
            best_count = pair_counts[best_pair]
            new_id = FIRST_MERGE_ID + merge_number
            byte_vocabulary[new_id] = (
                byte_vocabulary[best_pair[0]] + byte_vocabulary[best_pair[1]]
            )
            learned_merges.append(best_pair)
            changed_pairs: set[tuple[int, int]] = set()
            affected_word_ids = list(pair_to_words[best_pair])

            # only visit words that contained the pair being merged
            for word_id in affected_word_ids:
                old_word = running_words[word_id]
                new_word = merge_pair_left_to_right(old_word, best_pair, new_id)

                old_counts = Counter(zip(old_word, old_word[1:]))
                new_counts = Counter(zip(new_word, new_word[1:]))
                all_changed_pairs = set(old_counts) | set(new_counts)

                for pair in all_changed_pairs:
                    difference = new_counts[pair] - old_counts[pair]
                    if difference == 0:
                        continue

                    pair_counts[pair] += difference * frequencies[word_id]
                    changed_pairs.add(pair)

                    if new_counts[pair]:
                        pair_to_words[pair].add(word_id)
                    else:
                        pair_to_words[pair].discard(word_id)

                running_words[word_id] = new_word

            for pair in changed_pairs:
                if pair_counts[pair] <= 0:
                    pair_counts.pop(pair, None)
                    pair_to_words.pop(pair, None)

            if verify_counts:
                self.check_pair_indexes(
                    running_words,
                    frequencies,
                    pair_counts,
                    pair_to_words,
                )

            if log_every and (merge_number + 1) % log_every == 0:
                elapsed = time.perf_counter() - start_time
                remaining = elapsed * (merge_count - merge_number - 1)
                remaining /= merge_number + 1
                log(
                    f"[fast] {merge_number + 1:,}/{merge_count:,} merges"
                    f" | best pair count: {best_count:,}"
                    f" | pairs: {len(pair_counts):,}"
                    f" | elapsed: {elapsed:.0f}s"
                    f" | estimated remaining: {remaining / 3600:.2f}h"
                )

            if checkpoint is not None:
                checkpoint(learned_merges)

        return learned_merges

    @staticmethod
    def check_pair_indexes(
        words: list[list[int]],
        frequencies: list[int],
        pair_counts: Mapping[tuple[int, int], int],
        pair_to_words: Mapping[tuple[int, int], set[int]],
    ) -> None:
        fresh_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
        fresh_word_index: defaultdict[tuple[int, int], set[int]] = defaultdict(set)

        for word_id, word in enumerate(words):
            for pair in zip(word, word[1:]):
                fresh_counts[pair] += frequencies[word_id]
                fresh_word_index[pair].add(word_id)

        assert dict(pair_counts) == dict(fresh_counts)
        assert dict(pair_to_words) == dict(fresh_word_index)

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
            word_bytes = match.group().encode(
                "utf-8",
                errors="surrogatepass",
            )
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

    def save(self, path: str | Path) -> None:
        data = {
            "version": 2,
            "vocab_size": FIRST_MERGE_ID + len(self.merges),
            "num_merges": len(self.merges),
            "pattern": URDU_WORD_PATTERN,
            "special_tokens": {SPECIAL_TEXT: SPECIAL_ID},
            "train_stats": self.training_stats,
            "merges": [[left, right] for left, right in self.merges],
        }
        with open(path, "w", encoding="utf-8", newline="\n") as output_file:
            json.dump(data, output_file, ensure_ascii=True)

    @classmethod
    def load(cls, path: str | Path) -> "UrduBPETokenizer":
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


def run_builtin_test(report_path: Path) -> None:
    sample = (
        "اردو زبان کا نمونہ ہے۔ یہ ایک سادہ ٹیسٹ ہے۔ "
        "BPE متن کو ٹوکنز میں تقسیم کرتا ہے۔\n"
    ) * 20
    tokenizer = UrduBPETokenizer()
    words = list(tokenizer.word_pattern.finditer(sample))
    word_pieces = [match.group() for match in words]
    vocabulary_size = FIRST_MERGE_ID + 100

    reference = UrduBPETokenizer()
    reference.train_reference(word_pieces, vocabulary_size, log_every=0)
    fast = UrduBPETokenizer()
    fast.train_fast(
        Counter(word_pieces),
        vocabulary_size,
        log_every=0,
        log=lambda _: None,
    )

    if reference.merges != fast.merges:
        raise AssertionError("reference and fast merge lists differ")
    if fast.decode(fast.encode(sample)) != sample:
        raise AssertionError("text did not survive the encode/decode check")

    show_title("BPE BUILT-IN TEST")
    show_value("Result", "Passed")
    show_value("Merges compared", len(fast.merges))
    show_value("Reference matches fast", "Yes")
    show_value("Encode/decode", "Exact")
    show_value("Report file", report_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the built-in BPE test.")
    default_report = Path(__file__).resolve().parent / "outputs" / "bpe_test.txt"
    parser.add_argument("--report", default=str(default_report))
    args = parser.parse_args()

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8", newline="\n") as report:
        with redirect_stdout(Tee(sys.stdout, report)):
            run_builtin_test(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
