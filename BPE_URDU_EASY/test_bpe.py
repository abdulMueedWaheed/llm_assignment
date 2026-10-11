import argparse
from collections import Counter
from contextlib import redirect_stdout
import random
import sys
import tempfile
import time
from pathlib import Path

import regex

from bpe import (
    FIRST_MERGE_ID,
    SPECIAL_ID,
    SPECIAL_TEXT,
    URDU_WORD_PATTERN,
    UrduBPETokenizer,
)
from output_utils import (
    Tee,
    show_section,
    show_title,
    show_value,
)


def stop_logging(message: str) -> None:
    return None


def test_sanity_example() -> None:
    word = "aaabdaaabac"
    reference = UrduBPETokenizer()
    reference.train_reference(
        [word],
        vocabulary_size=FIRST_MERGE_ID + 3,
        log_every=0,
        log=stop_logging,
    )

    fast = UrduBPETokenizer()
    fast.train_fast(
        {word: 1},
        vocabulary_size=FIRST_MERGE_ID + 3,
        log_every=0,
        log=stop_logging,
    )

    expected_tokens = [b"aa", b"aaa", b"aaab"]
    for tokenizer in (reference, fast):
        actual_tokens = [
            tokenizer.vocabulary[token_id]
            for token_id in range(FIRST_MERGE_ID, FIRST_MERGE_ID + 3)
        ]
        assert actual_tokens == expected_tokens

    print("Sanity example passed.")


def test_pre_token_pattern() -> None:
    pattern = regex.compile(URDU_WORD_PATTERN)
    examples = [
        "",
        "hello  world\n\n",
        "کتاب۔ 123 ١٢٣ ۱۲۳ 😀",
        "a\tb  ",
        "\r\n\r\n",
        "\u00a0x",
        "اُردُو زبان",
        "ب\u200cا",
    ]

    for example in examples:
        pieces = pattern.findall(example)
        assert "".join(pieces) == example

    print("Urdu and mixed-text splitting passed.")


def get_test_texts() -> list[str]:
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
        SPECIAL_TEXT,
        "x" + SPECIAL_TEXT + "y",
        SPECIAL_TEXT + SPECIAL_TEXT,
        "<|endoftext|",
        "|>" + SPECIAL_TEXT,
        SPECIAL_TEXT + "\n",
        "a" * 5_000,
    ]


def test_text_round_trips(tokenizer: UrduBPETokenizer) -> None:
    for text in get_test_texts():
        assert tokenizer.decode(tokenizer.encode(text)) == text

    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""
    assert tokenizer.encode(SPECIAL_TEXT) == [SPECIAL_ID]
    assert tokenizer.encode(SPECIAL_TEXT + SPECIAL_TEXT) == [
        SPECIAL_ID,
        SPECIAL_ID,
    ]
    assert all(SPECIAL_ID not in pair for pair in tokenizer.merges)

    random_generator = random.Random(1)
    alphabet = (
        list("abc xyz\n\t\r.,!")
        + list("اُردوکتابیہ۔،ںےٹڈڑ")
        + ["\u200c", "\u064e", "😀", "\ud800", "١", "۱", "<", "|", ">"]
        + [SPECIAL_TEXT]
    )

    for _ in range(500):
        text = "".join(
            random_generator.choice(alphabet)
            for _ in range(random_generator.randint(0, 40))
        )
        assert tokenizer.decode(tokenizer.encode(text)) == text

    print("Text round-trip tests passed.")


def make_random_word_counts(
    random_generator: random.Random,
) -> Counter[str]:
    alphabet = [chr(value) for value in range(0x627, 0x640)]
    alphabet.extend(["a", "b", "\u064e"])
    counts: Counter[str] = Counter()

    for _ in range(random_generator.randint(5, 30)):
        word_length = random_generator.randint(1, 9)
        word = "".join(
            random_generator.choice(alphabet)
            for _ in range(word_length)
        )
        counts[word] = random_generator.randint(1, 10)

    return counts


def test_reference_and_fast_match(case_count: int = 50) -> None:
    random_generator = random.Random(7)
    total_merges = 0

    for case_number in range(case_count):
        word_counts = make_random_word_counts(random_generator)
        all_word_occurrences = [
            word
            for word, count in word_counts.items()
            for _ in range(count)
        ]
        vocabulary_size = FIRST_MERGE_ID + 100

        reference = UrduBPETokenizer()
        reference.train_reference(
            all_word_occurrences,
            vocabulary_size,
            log_every=0,
            log=stop_logging,
        )

        fast = UrduBPETokenizer()
        fast.train_fast(
            word_counts,
            vocabulary_size,
            log_every=0,
            log=stop_logging,
            verify_counts=True,
        )

        assert reference.merges == fast.merges, (
            f"reference and fast disagree in random case {case_number}"
        )
        assert reference.vocabulary == fast.vocabulary
        total_merges += len(reference.merges)

    print(
        f"Reference and fast match: {case_count} small inputs, "
        f"{total_merges:,} merges."
    )


def test_save_and_load() -> None:
    counts = Counter({"کتاب": 10, "کتابیں": 4, "نایاب": 1})
    tokenizer = UrduBPETokenizer()
    tokenizer.train_fast(
        counts,
        vocabulary_size=FIRST_MERGE_ID + 20,
        min_frequency=2,
        log_every=0,
        log=stop_logging,
    )

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "tokenizer.json"
        tokenizer.save(path)
        loaded = UrduBPETokenizer.load(path)

    assert loaded.merges == tokenizer.merges
    assert loaded.vocabulary == tokenizer.vocabulary
    for text in get_test_texts():
        assert loaded.encode(text) == tokenizer.encode(text)
    print("Save and load test passed.")


def compare_on_one_megabyte(
    train_path: str,
    requested_merges: str,
) -> None:
    words = []
    sample_bytes = 0
    tokenizer = UrduBPETokenizer()

    with open(train_path, "r", encoding="utf-8", newline="") as input_file:
        for line in input_file:
            for text_part in line.split(SPECIAL_TEXT):
                words.extend(
                    match.group()
                    for match in tokenizer.word_pattern.finditer(text_part)
                )
            sample_bytes += len(line.encode("utf-8", errors="surrogatepass"))
            if sample_bytes >= 1_000_000:
                break

    word_counts = Counter(words)
    merge_count = (
        49_743 if requested_merges == "full" else int(requested_merges)
    )
    if merge_count < 1:
        raise ValueError("requested merge count must be positive or 'full'")
    vocabulary_size = FIRST_MERGE_ID + merge_count

    show_section("ONE-MEGABYTE INPUT")
    show_value("Sample size", f"{sample_bytes / 1_000_000:.2f} MB")
    show_value("Different words", f"{len(word_counts):,}")
    show_value("Word occurrences", f"{sum(word_counts.values()):,}")

    start_time = time.perf_counter()
    reference = UrduBPETokenizer()
    reference.train_reference(
        words,
        vocabulary_size,
        log_every=500,
    )
    reference_finished = time.perf_counter()

    fast = UrduBPETokenizer()
    fast.train_fast(
        word_counts,
        vocabulary_size,
        log_every=0,
        log=stop_logging,
    )
    fast_finished = time.perf_counter()

    assert reference.merges == fast.merges
    assert reference.vocabulary == fast.vocabulary
    show_section("COMPARISON")
    show_value("Reference time", f"{reference_finished - start_time:.1f}s")
    show_value("Fast time", f"{fast_finished - reference_finished:.1f}s")
    show_value("Merges compared", f"{len(fast.merges):,}")
    show_value("Merge lists match", "Yes")
    test_text_round_trips(fast)


def run_tests(args: argparse.Namespace, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8", newline="\n") as report:
        with redirect_stdout(Tee(sys.stdout, report)):
            show_title("EASY BPE TESTS")
            test_sanity_example()
            test_pre_token_pattern()
            test_text_round_trips(UrduBPETokenizer())
            test_reference_and_fast_match()
            test_save_and_load()

            if args.train:
                compare_on_one_megabyte(args.train, args.ref_merges)

            show_section("FINAL RESULT")
            show_value("Tests", "All passed")
            show_value("Report", report_path.resolve())


def main() -> None:
    script_folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Test the beginner-friendly Urdu BPE code."
    )
    parser.add_argument("--train")
    parser.add_argument("--ref-merges", default="2000")
    parser.add_argument(
        "--report",
        default=str(script_folder / "outputs" / "test_bpe_results.txt"),
    )
    args = parser.parse_args()
    run_tests(args, Path(args.report))


if __name__ == "__main__":
    main()
