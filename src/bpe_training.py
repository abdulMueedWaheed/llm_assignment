from collections import Counter, defaultdict
import heapq
import time
from typing import Callable, Iterable, Mapping

from bpe_tokens import FIRST_MERGE_ID, PairHeapEntry, merge_pair_left_to_right


class TrainingMixin:
    def train_reference(
        self,
        word_occurrences: Iterable[str],
        vocabulary_size: int = 50_000,
        log_every: int = 1_000,
        log: Callable[[str], object] = print,
    ) -> list[tuple[int, int]]:
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
        pair_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
        pair_to_words: defaultdict[tuple[int, int], set[int]] = defaultdict(set)

        for word_id, word in enumerate(running_words):
            for pair in zip(word, word[1:]):
                pair_counts[pair] += frequencies[word_id]
                pair_to_words[pair].add(word_id)

        byte_vocabulary = self.make_byte_vocabulary()
        learned_merges = []
        start_time = time.perf_counter()

        def make_entry(pair: tuple[int, int], count: int) -> PairHeapEntry:
            return PairHeapEntry(
                count,
                (byte_vocabulary[pair[0]], byte_vocabulary[pair[1]]),
                pair,
            )

        heap = [make_entry(pair, count) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

        for merge_number in range(merge_count):
            best_pair = None
            best_count = 0
            while heap:
                entry = heapq.heappop(heap)
                actual_count = pair_counts.get(entry.pair, 0)
                if actual_count == entry.count and actual_count > 0:
                    best_pair = entry.pair
                    best_count = actual_count
                    break
                if 0 < actual_count < entry.count:
                    heapq.heappush(heap, make_entry(entry.pair, actual_count))

            if best_pair is None:
                break

            new_id = FIRST_MERGE_ID + merge_number
            byte_vocabulary[new_id] = (
                byte_vocabulary[best_pair[0]] + byte_vocabulary[best_pair[1]]
            )
            learned_merges.append(best_pair)
            changed_pairs: set[tuple[int, int]] = set()
            previous_counts: dict[tuple[int, int], int] = {}
            affected_word_ids = list(pair_to_words[best_pair])

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

                    previous_counts.setdefault(pair, pair_counts.get(pair, 0))
                    pair_counts[pair] += difference * frequencies[word_id]
                    changed_pairs.add(pair)

                    if new_counts[pair]:
                        pair_to_words[pair].add(word_id)
                    else:
                        pair_to_words[pair].discard(word_id)

                running_words[word_id] = new_word

            for pair in changed_pairs:
                count = pair_counts[pair]
                if count <= 0:
                    pair_counts.pop(pair, None)
                    pair_to_words.pop(pair, None)
                elif count > previous_counts[pair]:
                    heapq.heappush(heap, make_entry(pair, count))

            if len(heap) > 3 * len(pair_counts) + 100_000:
                heap = [make_entry(pair, count) for pair, count in pair_counts.items()]
                heapq.heapify(heap)

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
                    f" | heap: {len(heap):,}"
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
