"""Tests for the Urdu byte-level BPE tokenizer."""

import argparse
import os
import random
import sys
import tempfile
import time
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

import regex

from output_utils import Tee, print_field, print_header, print_section
from bpe import (
    BPETokenizer,
    FIRST_MERGE_ID,
    PATTERN,
    SPECIAL,
    SPECIAL_ID,
)


def quiet_log(_: str) -> None:
    pass


def test_sanity() -> None:
    counts = {"aaabdaaabac": 1}
    reference = BPETokenizer()
    reference.train_reference(
        ["aaabdaaabac"], vocab_size=FIRST_MERGE_ID + 3, log_every=0, log=quiet_log
    )
    fast = BPETokenizer()
    fast.train_from_counts(
        counts, vocab_size=FIRST_MERGE_ID + 3, log_every=0, log=quiet_log
    )
    for tokenizer in (reference, fast):
        assert tokenizer.merges[0] == (ord("a"), ord("a"))
        assert [tokenizer.vocab[token_id] for token_id in range(257, 260)] == [
            b"aa",
            b"aaa",
            b"aaab",
        ]
        assert tokenizer.special_tokens == {SPECIAL: SPECIAL_ID}
        assert tokenizer.inverse_vocab[b"aa"] == 257
    print("Sanity example passed.")


def test_pattern() -> None:
    pattern = regex.compile(PATTERN)
    assert pattern.findall("اُردُو زبان") == ["اُردُو", " زبان"]
    zwnj_word = "ب\u200cا"
    assert pattern.findall(zwnj_word) == [zwnj_word]
    assert pattern.findall("hello  world") == ["hello", " ", " world"]
    samples = [
        "",
        "hello  world\n\n",
        "کتاب۔ 123 ١٢٣ ۱۲۳ 😀",
        "a\tb  ",
        "\r\n\r\n",
        "\u00a0x",
    ]
    for sample in samples:
        assert "".join(pattern.findall(sample)) == sample, repr(sample)
    print("Urdu and mixed-text pre-tokenization passed.")


def _samples() -> list[str]:
    return [
        "",
        " ",
        "\n",
        "\r\n",
        "hello world",
        "Hello, World! 123",
        "کتاب",
        "اُردُو زبان",
        "یہ ایک ٹیسٹ ہے۔",
        "urdu اور english mixed",
        "😀🔥 emoji",
        "line1\nline2\n\nline4",
        "  leading and trailing  ",
        "a\u200cb",
        "\ud800 lone surrogate",
        "\ud83d\ude00 split pair",
        "tabs\t\tand\r\nwindows newlines",
        "\x00\x01 control",
        "\U0010ffff",
        SPECIAL,
        "x" + SPECIAL + "y",
        SPECIAL + SPECIAL,
        "<|endoftext|",
        "|>" + SPECIAL,
        SPECIAL + "\n",
        "a" * 5_000,
    ]


def test_lossless(tokenizer: BPETokenizer, label: str) -> None:
    for sample in _samples():
        assert tokenizer.decode(tokenizer.encode(sample)) == sample, repr(sample)
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""
    assert tokenizer.encode(SPECIAL) == [SPECIAL_ID]
    assert tokenizer.encode("x" + SPECIAL + "y").count(SPECIAL_ID) == 1
    assert tokenizer.encode(SPECIAL + SPECIAL) == [SPECIAL_ID, SPECIAL_ID]
    assert all(SPECIAL_ID not in pair for pair in tokenizer.merges)

    rng = random.Random(1)
    alphabet = (
        list("abc xyz\n\t\r.,!")
        + list("اُردوکتابیہ۔،ںےٹڈڑ")
        + ["\u200c", "\u064e", "😀", "\ud800", "١", "۱", "<", "|", ">"]
        + [SPECIAL]
    )
    for _ in range(3_000):
        sample = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        assert tokenizer.decode(tokenizer.encode(sample)) == sample, repr(sample)
    print(f"Lossless encode/decode passed: {label}.")


def _random_counts(rng: random.Random) -> Counter[str]:
    alphabet = (
        [chr(codepoint) for codepoint in range(0x627, 0x640)]
        + ["a", "b", "\u064e"]
    )
    counts: Counter[str] = Counter()
    for _ in range(rng.randint(5, 60)):
        word = "".join(
            rng.choice(alphabet[: rng.randint(2, len(alphabet))])
            for _ in range(rng.randint(1, 9))
        )
        counts[word] = rng.randint(1, 30)
    return counts


def test_random_equivalence(cases: int = 150) -> None:
    rng = random.Random(7)
    merges_checked = 0
    for case_number in range(cases):
        counts = _random_counts(rng)
        requested_size = FIRST_MERGE_ID + 5_000
        pieces = [
            word
            for word, frequency in counts.items()
            for _ in range(frequency)
        ]
        reference = BPETokenizer()
        reference.train_reference(
            pieces, requested_size, log_every=0, log=quiet_log
        )
        fast = BPETokenizer()
        fast.train_from_counts(
            counts,
            requested_size,
            log_every=0,
            log=quiet_log,
            verify=True,
            heap_slack=0,
        )
        assert reference.merges == fast.merges, (
            f"random corpus {case_number}: merge mismatch"
        )
        merges_checked += len(reference.merges)
    print(f"Random equivalence passed: {cases} corpora, {merges_checked:,} merges.")


def test_save_load() -> None:
    counts = Counter({"کتاب": 10, "کتابیں": 4, "نایاب": 1})
    tokenizer = BPETokenizer()
    tokenizer.train_from_counts(
        counts,
        vocab_size=FIRST_MERGE_ID + 20,
        min_freq=2,
        log_every=0,
        log=quiet_log,
    )
    assert tokenizer.train_stats["distinct_used"] == 2
    assert tokenizer.train_stats["distinct_total"] == 3
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "tokenizer.json")
        tokenizer.save(path)
        loaded = BPETokenizer.load(path)
    assert loaded.merges == tokenizer.merges
    assert loaded.vocab == tokenizer.vocab
    for sample in _samples():
        assert loaded.encode(sample) == tokenizer.encode(sample)
    assert tokenizer.decode(tokenizer.encode("نایاب")) == "نایاب"
    print("Tokenizer save/load and frequency filtering passed.")


def make_sample(path: str, nbytes: int = 1_000_000) -> tuple[list[str], int]:
    lines = []
    total_bytes = 0
    with open(path, "r", encoding="utf-8", newline="") as source:
        for line in source:
            lines.append(line)
            total_bytes += len(line.encode("utf-8", errors="surrogatepass"))
            if total_bytes >= nbytes:
                break
    return lines, total_bytes


def test_sample_equivalence(train_path: str, merge_count: str) -> None:
    with tempfile.TemporaryDirectory() as directory:
        sample_path = os.path.join(directory, "sample_1mb.txt")
        lines, byte_count = make_sample(train_path)
        with open(sample_path, "w", encoding="utf-8", newline="") as sample_file:
            sample_file.writelines(lines)
        pieces = list(BPETokenizer().iter_pieces(sample_path))
        counts = Counter(pieces)
    print_section("ONE-MEGABYTE SAMPLE")
    print_field("Sample size", f"{byte_count / 1_000_000:.2f} MB")
    print_field("Distinct pre-tokens", f"{len(counts):,}")
    print_field("Pre-token occurrences", f"{sum(counts.values()):,}")

    requested_merges = 49_743 if merge_count == "full" else int(merge_count)
    if requested_merges < 1:
        raise ValueError("--ref-merges must be a positive integer or 'full'")
    vocab_size = FIRST_MERGE_ID + requested_merges

    started = time.perf_counter()
    reference = BPETokenizer()
    reference.train_reference(
        pieces, vocab_size, log_every=500, log=print
    )
    reference_finished = time.perf_counter()
    fast = BPETokenizer()
    fast.train_from_counts(
        counts, vocab_size, log_every=0, log=quiet_log
    )
    fast_finished = time.perf_counter()

    assert reference.merges == fast.merges, "reference and fast merge lists differ"
    assert reference.vocab == fast.vocab
    print_section("REFERENCE / FAST COMPARISON")
    print_field("Reference time", f"{reference_finished - started:.1f}s")
    print_field("Fast time", f"{fast_finished - reference_finished:.1f}s")
    print_field("Requested merges", f"{requested_merges:,}")
    print_field("Reference merges", f"{len(reference.merges):,}")
    print_field("Fast merges", f"{len(fast.merges):,}")
    print_field("Merge lists match", f"Yes ({len(fast.merges):,} merges)")
    test_lossless(fast, "trained on the 1 MB sample")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        help="train.txt used to create the 1 MB comparison sample",
    )
    parser.add_argument(
        "--ref-merges",
        default="2000",
        help="number of 1 MB sample merges to compare, or 'full' (slow)",
    )
    parser.add_argument(
        "--report",
        default=str(
            Path(__file__).resolve().parent / "outputs" / "test_bpe_py_results.txt"
        ),
        help="text report path",
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8", newline="\n") as report:
        with redirect_stdout(Tee(sys.stdout, report)):
            print_header("BPE TOKENIZER TESTS")
            test_sanity()
            test_pattern()
            test_lossless(BPETokenizer(), "untrained tokenizer")
            test_random_equivalence()
            test_save_load()
            if args.train:
                test_sample_equivalence(args.train, args.ref_merges)
            print_section("FINAL RESULT")
            print_field("Test suite", "ALL TESTS PASSED")
            print_field("Report saved to", report_path)


if __name__ == "__main__":
    main()
