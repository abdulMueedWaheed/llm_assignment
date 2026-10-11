from collections import Counter
from contextlib import redirect_stdout
import argparse
from pathlib import Path
import sys

from output_utils import Tee, show_title, show_value
from bpe_tokenizer import BPETokenizer as _BPETokenizer
from bpe_tokens import (
    FIRST_MERGE_ID,
    SPECIAL_ID,
    SPECIAL_TEXT,
    URDU_WORD_PATTERN,
    open_text_file,
)

__all__ = [
    "BPETokenizer",
    "FIRST_MERGE_ID",
    "SPECIAL_ID",
    "SPECIAL_TEXT",
    "URDU_WORD_PATTERN",
    "UrduBPETokenizer",
    "open_text_file",
]


class BPETokenizer(_BPETokenizer):
    pass


UrduBPETokenizer = BPETokenizer


def run_builtin_test(report_path: Path) -> None:
    sample = (
        "اردو زبان کا نمونہ ہے۔ یہ ایک سادہ ٹیسٹ ہے۔ "
        "BPE متن کو ٹوکنز میں تقسیم کرتا ہے۔\n"
    ) * 20
    tokenizer = BPETokenizer()
    word_pieces = [
        match.group() for match in tokenizer.word_pattern.finditer(sample)
    ]
    vocabulary_size = FIRST_MERGE_ID + 100

    reference = BPETokenizer()
    reference.train_reference(word_pieces, vocabulary_size, log_every=0)
    fast = BPETokenizer()
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
    default_report = Path(__file__).resolve().parent.parent / "logs" / "bpe_test.log"
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
