"""Byte-level BPE tokenizer for Urdu, implemented without tokenizer libraries."""

import argparse
from contextlib import redirect_stdout
import heapq
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Mapping

import regex

from output_utils import Tee

SPECIAL = "<|endoftext|>"
SPECIAL_ID = 256
FIRST_MERGE_ID = 257
CACHE_MAX = 5_000_000

PATTERN = (
    r"""'s|'t|'re|'ve|'m|'ll|'d"""
    r"""| ?[\p{L}\p{M}\u200c]+"""
    r"""| ?\p{N}+"""
    r"""| ?[^\s\p{L}\p{M}\p{N}\u200c]+"""
    r"""|\s+(?!\S)"""
    r"""|\s+"""
)


def open_text(path: str | Path):
    """Open UTF-8 text without translating stored line endings."""
    return open(path, "r", encoding="utf-8", newline="")


def merge_ids(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace non-overlapping occurrences of a pair from left to right."""
    left, right = pair
    merged = []
    index = 0
    while index < len(ids):
        if index + 1 < len(ids) and ids[index] == left and ids[index + 1] == right:
            merged.append(new_id)
            index += 2
        else:
            merged.append(ids[index])
            index += 1
    return merged


class _HeapEntry:
    __slots__ = ("count", "byte_key", "pair")

    def __init__(self, count: int, byte_key: tuple[bytes, bytes], pair: tuple[int, int]):
        self.count = count
        self.byte_key = byte_key
        self.pair = pair

    def __lt__(self, other: "_HeapEntry") -> bool:
        if self.count != other.count:
            return self.count > other.count
        if self.byte_key != other.byte_key:
            return self.byte_key > other.byte_key
        return self.pair > other.pair


class BPETokenizer:
    def __init__(self) -> None:
        self.pat = regex.compile(PATTERN)
        self.merges: list[tuple[int, int]] = []
        self.vocab = self._base_vocab()
        self.inverse_vocab = {token: token_id for token_id, token in self.vocab.items()}
        self.special_tokens = {SPECIAL: SPECIAL_ID}
        self.rank: dict[tuple[int, int], int] = {}
        self.cache: dict[bytes, list[int]] = {}
        self.train_stats: dict[str, int] = {}

    @staticmethod
    def _base_vocab() -> dict[int, bytes]:
        vocab = {token_id: bytes([token_id]) for token_id in range(256)}
        vocab[SPECIAL_ID] = SPECIAL.encode("utf-8")
        return vocab

    def _rebuild_from_merges(self) -> None:
        self.vocab = self._base_vocab()
        self.inverse_vocab = {}
        for token_id, token in self.vocab.items():
            self.inverse_vocab.setdefault(token, token_id)
        self.rank = {}
        for index, pair in enumerate(self.merges):
            new_id = FIRST_MERGE_ID + index
            left, right = pair
            if SPECIAL_ID in pair:
                raise ValueError("the special token must never be merged")
            if left not in self.vocab or right not in self.vocab:
                raise ValueError(f"merge {index + 1} uses a token ID that does not exist yet")
            self.vocab[new_id] = self.vocab[left] + self.vocab[right]
            self.inverse_vocab.setdefault(self.vocab[new_id], new_id)
            self.rank[pair] = new_id
        self.special_tokens = {SPECIAL: SPECIAL_ID}
        self.cache.clear()

    def iter_pieces(self, path: str | Path, max_bytes: int | None = None) -> Iterable[str]:
        """Yield pre-tokens line by line; special-token text is excluded."""
        bytes_read = 0
        with open_text(path) as source:
            for line in source:
                bytes_read += len(line.encode("utf-8", errors="surrogatepass"))
                for chunk in line.split(SPECIAL):
                    for match in self.pat.finditer(chunk):
                        yield match.group()
                if max_bytes is not None and bytes_read >= max_bytes:
                    break

    def count_words(self, path: str | Path, max_bytes: int | None = None) -> Counter[str]:
        """Count distinct pre-tokens for the optimized trainer."""
        return Counter(self.iter_pieces(path, max_bytes))

    def train_reference(
        self,
        pieces: Iterable[str],
        vocab_size: int = 50_000,
        log_every: int = 1_000,
        log: Callable[[str], object] = print,
    ) -> list[tuple[int, int]]:
        """Train the unoptimized reference model from every pre-token occurrence."""
        if vocab_size <= FIRST_MERGE_ID or vocab_size > 65_536:
            raise ValueError("vocab_size must be between 258 and 65,536")
        if log_every < 0:
            raise ValueError("log_every cannot be negative")

        words = [
            list(piece.encode("utf-8", errors="surrogatepass"))
            for piece in pieces
        ]
        self.train_stats = {
            "distinct_total": 0,
            "distinct_used": 0,
            "occurrences_total": len(words),
            "occurrences_used": len(words),
            "min_freq": 1,
        }
        self.merges = self._train_reference(
            words, vocab_size - FIRST_MERGE_ID, log_every, log
        )
        self._rebuild_from_merges()
        return self.merges

    def train_from_counts(
        self,
        word_counts: Mapping[str, int],
        vocab_size: int = 50_000,
        min_freq: int = 1,
        log_every: int = 1_000,
        log: Callable[[str], object] = print,
        checkpoint: Callable[[list[tuple[int, int]]], object] | None = None,
        verify: bool = False,
        heap_slack: int = 100_000,
    ) -> list[tuple[int, int]]:
        """Train the optimized model from distinct pre-token frequencies."""
        if vocab_size <= FIRST_MERGE_ID:
            raise ValueError(f"vocab_size must be greater than {FIRST_MERGE_ID}")
        if min_freq < 1:
            raise ValueError("min_freq must be at least 1")
        if heap_slack < 0:
            raise ValueError("heap_slack cannot be negative")
        if vocab_size > 65_536:
            raise ValueError("vocab_size must not exceed 65,536 for uint16 output")
        if any(not isinstance(count, int) or count < 1 for count in word_counts.values()):
            raise ValueError("word frequencies must be positive integers")

        total_occurrences = sum(word_counts.values())
        kept = [(word, count) for word, count in word_counts.items() if count >= min_freq]
        kept_occurrences = sum(count for _, count in kept)
        self.train_stats = {
            "distinct_total": len(word_counts),
            "distinct_used": len(kept),
            "occurrences_total": total_occurrences,
            "occurrences_used": kept_occurrences,
            "min_freq": min_freq,
        }
        log(
            f"words used: {len(kept):,} of {len(word_counts):,} distinct "
            f"({100.0 * kept_occurrences / max(total_occurrences, 1):.3f}% "
            f"of running pre-tokens, min_freq={min_freq})"
        )

        words = [
            list(word.encode("utf-8", errors="surrogatepass"))
            for word, _ in kept
        ]
        frequencies = [frequency for _, frequency in kept]
        del kept
        merges = self._train_fast(
            words,
            frequencies,
            vocab_size - FIRST_MERGE_ID,
            log_every,
            log,
            checkpoint,
            verify,
            heap_slack,
        )
        self.merges = merges
        self._rebuild_from_merges()
        return merges

    def _train_reference(
        self,
        words: list[list[int]],
        num_merges: int,
        log_every: int,
        log: Callable[[str], object],
    ) -> list[tuple[int, int]]:
        """Naive baseline: recount every pair in every running word per merge."""
        vocab = self._base_vocab()
        merges = []
        started = time.perf_counter()

        for index in range(num_merges):
            counts: Counter[tuple[int, int]] = Counter()
            for word in words:
                counts.update(zip(word, word[1:]))
            if not counts:
                break

            best = max(
                counts,
                key=lambda pair: (
                    counts[pair],
                    (vocab[pair[0]], vocab[pair[1]]),
                    pair,
                ),
            )
            new_id = FIRST_MERGE_ID + index
            words = [merge_ids(word, best, new_id) for word in words]
            vocab[new_id] = vocab[best[0]] + vocab[best[1]]
            merges.append(best)

            if log_every and (index + 1) % log_every == 0:
                log(
                    f"[ref] merge {index + 1}/{num_merges} {vocab[new_id]!r} "
                    f"count={counts[best]} t={time.perf_counter() - started:.0f}s"
                )
        return merges

    def _train_fast(
        self,
        words: list[list[int]],
        frequencies: list[int],
        num_merges: int,
        log_every: int,
        log: Callable[[str], object],
        checkpoint: Callable[[list[tuple[int, int]]], object] | None,
        verify: bool,
        heap_slack: int,
    ) -> list[tuple[int, int]]:
        """Incremental weighted trainer with a pair index and lazy heap."""
        vocab = self._base_vocab()
        merges: list[tuple[int, int]] = []
        started = time.perf_counter()
        pair_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
        pair_words: defaultdict[tuple[int, int], set[int]] = defaultdict(set)

        for word_id, word in enumerate(words):
            frequency = frequencies[word_id]
            for pair in zip(word, word[1:]):
                pair_counts[pair] += frequency
                pair_words[pair].add(word_id)

        def make_entry(pair: tuple[int, int], count: int) -> _HeapEntry:
            return _HeapEntry(count, (vocab[pair[0]], vocab[pair[1]]), pair)

        heap = [make_entry(pair, count) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

        for index in range(num_merges):
            best = None
            best_count = 0
            while heap:
                entry = heapq.heappop(heap)
                actual = pair_counts.get(entry.pair, 0)
                if actual == entry.count and actual > 0:
                    best, best_count = entry.pair, actual
                    break
                if 0 < actual < entry.count:
                    heapq.heappush(heap, make_entry(entry.pair, actual))
            if best is None:
                break

            new_id = FIRST_MERGE_ID + index
            vocab[new_id] = vocab[best[0]] + vocab[best[1]]
            merges.append(best)
            changed: set[tuple[int, int]] = set()
            previous_counts: dict[tuple[int, int], int] = {}

            for word_id in list(pair_words.get(best, ())):
                word = words[word_id]
                merged_word = merge_ids(word, best, new_id)
                if len(merged_word) == len(word):
                    continue

                delta: dict[tuple[int, int], int] = {}
                for pair in zip(word, word[1:]):
                    delta[pair] = delta.get(pair, 0) - 1
                new_pairs = list(zip(merged_word, merged_word[1:]))
                for pair in new_pairs:
                    delta[pair] = delta.get(pair, 0) + 1
                new_pair_set = set(new_pairs)
                frequency = frequencies[word_id]

                for pair, difference in delta.items():
                    if difference == 0:
                        continue
                    previous_counts.setdefault(pair, pair_counts.get(pair, 0))
                    pair_counts[pair] += difference * frequency
                    changed.add(pair)
                    if pair == best:
                        continue
                    if difference > 0:
                        pair_words[pair].add(word_id)
                    elif pair not in new_pair_set:
                        pair_words[pair].discard(word_id)
                words[word_id] = merged_word

            pair_words.pop(best, None)
            for pair in changed:
                count = pair_counts.get(pair, 0)
                if count <= 0:
                    pair_counts.pop(pair, None)
                    pair_words.pop(pair, None)
                elif count > previous_counts[pair]:
                    heapq.heappush(heap, make_entry(pair, count))

            if len(heap) > 3 * len(pair_counts) + heap_slack:
                heap = [make_entry(pair, count) for pair, count in pair_counts.items()]
                heapq.heapify(heap)

            if verify:
                self._verify_state(words, frequencies, pair_counts, pair_words)

            if log_every and (index + 1) % log_every == 0:
                elapsed = time.perf_counter() - started
                remaining = elapsed * (num_merges - index - 1) / (index + 1)
                log(
                    f"[fast] merge {index + 1}/{num_merges} {vocab[new_id]!r} "
                    f"count={best_count} pairs={len(pair_counts):,} "
                    f"heap={len(heap):,} t={elapsed:.0f}s "
                    f"ETA={remaining / 3600:.2f}h"
                )
            if checkpoint is not None:
                checkpoint(merges)
        return merges

    @staticmethod
    def _verify_state(
        words: list[list[int]],
        frequencies: list[int],
        pair_counts: Mapping[tuple[int, int], int],
        pair_words: Mapping[tuple[int, int], set[int]],
    ) -> None:
        """Rebuild indexes and assert that the incremental state is exact."""
        counts: defaultdict[tuple[int, int], int] = defaultdict(int)
        locations: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
        for word_id, word in enumerate(words):
            for pair in zip(word, word[1:]):
                counts[pair] += frequencies[word_id]
                locations[pair].add(word_id)
        assert {pair: count for pair, count in pair_counts.items() if count} == dict(counts)
        assert {pair: ids for pair, ids in pair_words.items() if ids} == dict(locations)

    def _bpe_word(self, word_bytes: bytes) -> list[int]:
        """Apply learned merges in rank order to one pre-token."""
        ids = list(word_bytes)
        while len(ids) >= 2:
            best_pair = None
            best_rank = None
            for pair in zip(ids, ids[1:]):
                rank = self.rank.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair, best_rank = pair, rank
            if best_pair is None or best_rank is None:
                break
            ids = merge_ids(ids, best_pair, best_rank)
        return ids

    def _encode_chunk(self, text: str) -> list[int]:
        """Encode text that does not contain the reserved special-token string."""
        ids = []
        for match in self.pat.finditer(text):
            word_bytes = match.group().encode("utf-8", errors="surrogatepass")
            tokens = self.cache.get(word_bytes)
            if tokens is None:
                tokens = self._bpe_word(word_bytes)
                if len(self.cache) < CACHE_MAX:
                    self.cache[word_bytes] = tokens
            ids.extend(tokens)
        return ids

    def encode(self, text: str) -> list[int]:
        """Encode text; each literal special-token string maps to ID 256."""
        if not text:
            return []
        ids = []
        for index, chunk in enumerate(text.split(SPECIAL)):
            if index:
                ids.append(SPECIAL_ID)
            if chunk:
                ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        """Decode token IDs, preserving valid UTF-8 and surrogate-pass text."""
        data = b"".join(self.vocab[token_id] for token_id in ids)
        try:
            return data.decode("utf-8", errors="surrogatepass")
        except UnicodeDecodeError:
            return data.decode("utf-8", errors="replace")

    def save(self, path: str | Path, merges: list[tuple[int, int]] | None = None) -> None:
        """Save learned merges and metadata as JSON."""
        saved_merges = self.merges if merges is None else merges
        payload = {
            "version": 2,
            "vocab_size": FIRST_MERGE_ID + len(saved_merges),
            "num_merges": len(saved_merges),
            "pattern": PATTERN,
            "special_tokens": {SPECIAL: SPECIAL_ID},
            "train_stats": self.train_stats,
            "merges": [[left, right] for left, right in saved_merges],
        }
        with open(path, "w", encoding="utf-8", newline="\n") as output:
            json.dump(payload, output, ensure_ascii=True)

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as source:
            payload = json.load(source)
        if payload.get("special_tokens") != {SPECIAL: SPECIAL_ID}:
            raise ValueError("tokenizer file has an invalid special-token mapping")
        tokenizer = cls()
        tokenizer.pat = regex.compile(payload["pattern"])
        tokenizer.merges = [tuple(pair) for pair in payload["merges"]]
        expected_size = FIRST_MERGE_ID + len(tokenizer.merges)
        if payload.get("vocab_size") != expected_size:
            raise ValueError("tokenizer file is inconsistent (vocab_size vs merges)")
        if payload.get("num_merges", len(tokenizer.merges)) != len(tokenizer.merges):
            raise ValueError("tokenizer file is inconsistent (num_merges vs merges)")
        tokenizer.train_stats = payload.get("train_stats", {})
        tokenizer._rebuild_from_merges()
        return tokenizer

    def token_text(self, token_id: int) -> str:
        return self.vocab[token_id].decode("utf-8", errors="replace")

    def token_is_valid_utf8(self, token_id: int) -> bool:
        try:
            self.vocab[token_id].decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False

    def token_char_len(self, token_id: int) -> int | None:
        try:
            return len(self.vocab[token_id].decode("utf-8"))
        except UnicodeDecodeError:
            return None


def _run_builtin_test(report_path: Path) -> None:
    sample = (
        "اردو زبان کا نمونہ ہے۔ یہ ایک سادہ ٹیسٹ ہے۔ "
        "BPE متن کو ٹوکنز میں تقسیم کرتا ہے۔\n"
    ) * 20
    tokenizer = BPETokenizer()
    pieces = [
        match.group()
        for match in tokenizer.pat.finditer(sample)
    ]
    counts = Counter(pieces)
    vocab_size = FIRST_MERGE_ID + 100

    reference = BPETokenizer()
    reference.train_reference(pieces, vocab_size=vocab_size, log_every=0)
    fast = BPETokenizer()
    fast.train_from_counts(
        counts,
        vocab_size=vocab_size,
        log_every=0,
        log=lambda _: None,
    )

    if reference.merges != fast.merges:
        raise AssertionError("reference and fast merges differ")
    if fast.decode(fast.encode(sample)) != sample:
        raise AssertionError("sample failed the encode/decode round trip")

    print("BPE built-in test passed")
    print(f"sample characters : {len(sample):,}")
    print(f"pre-token count   : {len(pieces):,} ({len(counts):,} distinct)")
    print(f"merges compared   : {len(fast.merges):,}")
    print("reference == fast : yes")
    print("round trip        : exact")
    print(f"report saved to   : {report_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the built-in BPE test.")
    parser.add_argument(
        "--report",
        default=str(
            Path(__file__).resolve().parent / "outputs" / "bpe_py_test_results.txt"
        ),
        help="test report path",
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8", newline="\n") as report:
        with redirect_stdout(Tee(sys.stdout, report)):
            _run_builtin_test(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
