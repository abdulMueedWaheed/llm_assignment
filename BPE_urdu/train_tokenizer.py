"""Train the Urdu byte-level BPE tokenizer."""

import argparse
import os
import platform
import sys
import time
from pathlib import Path

from bpe import BPETokenizer, FIRST_MERGE_ID
from output_utils import print_header, print_section


def peak_memory_mb() -> float | None:
    """Return peak resident memory where the standard library exposes it."""
    if sys.platform == "win32":
        return None
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_train = (
        Path(__file__).resolve().parent.parent / "data" / "processed" / "train.txt"
    )
    parser.add_argument(
        "--train",
        default=str(default_train),
        help=f"training text file (default: {default_train})",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "outputs" / "tokenizer.json"),
        help="tokenizer JSON output (default: outputs/tokenizer.json)",
    )
    parser.add_argument(
        "--log",
        default=str(
            Path(__file__).resolve().parent / "outputs" / "train_tokenizer_log.txt"
        ),
        help="progress report path (default: outputs/train_tokenizer_log.txt)",
    )
    parser.add_argument("--vocab-size", type=int, default=50_000)
    parser.add_argument("--mode", choices=("fast", "reference"), default="fast")
    parser.add_argument(
        "--min-freq",
        type=int,
        default=1,
        help="drop less frequent pre-tokens (1 preserves all training data)",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        help="read only the first N bytes, for trial runs",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5_000,
        help="write a partial tokenizer every N merges; 0 disables it",
    )
    args = parser.parse_args()
    if not Path(args.train).is_file():
        parser.error(f"training text file does not exist: {args.train}")
    if args.vocab_size <= FIRST_MERGE_ID or args.vocab_size > 65_536:
        parser.error("--vocab-size must be between 258 and 65,536")
    if args.min_freq < 1:
        parser.error("--min-freq must be at least 1")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every cannot be negative")

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    log_file = open(args.log, "w", encoding="utf-8", newline="\n") if args.log else None

    def log(message: str = "") -> None:
        print(message, flush=True)
        if log_file is not None:
            log_file.write(message + "\n")
            log_file.flush()

    tokenizer = BPETokenizer()
    partial_path = str(output_path) + ".partial"

    def checkpoint(merges: list[tuple[int, int]]) -> None:
        if args.checkpoint_every and len(merges) % args.checkpoint_every == 0:
            tokenizer.save(partial_path, merges)

    try:
        started = time.perf_counter()
        print_header("URDU BPE TOKENIZER TRAINING")
        print_section("INPUTS AND SETTINGS")
        log(f"Training data           : {Path(args.train).resolve()}")
        log(f"Tokenizer output        : {output_path.resolve()}")
        log(f"Training mode           : {args.mode}")
        log(f"Vocabulary size         : {args.vocab_size:,}")
        log(f"Minimum word frequency : {args.min_freq}")
        log("Reading and counting pre-tokens ...")
        if args.mode == "reference":
            if args.min_freq != 1:
                parser.error("--min-freq is only supported by the fast trainer")
            pieces = list(tokenizer.iter_pieces(args.train, max_bytes=args.max_bytes))
            counts = None
            occurrences = len(pieces)
            distinct = None
        else:
            counts = tokenizer.count_words(args.train, max_bytes=args.max_bytes)
            pieces = None
            occurrences = sum(counts.values())
            distinct = len(counts)
        counted_at = time.perf_counter()
        print_section("CORPUS COUNTS")
        distinct_text = (
            f"{distinct:,}" if distinct is not None else "not aggregated"
        )
        log(f"Distinct pre-tokens      : {distinct_text}")
        log(f"Pre-token occurrences   : {occurrences:,}")
        log(f"Read/count time          : {counted_at - started:.0f}s")
        if counts is not None:
            singleton_count = sum(frequency == 1 for frequency in counts.values())
            log(
                f"Singleton pre-tokens    : {singleton_count:,} "
                f"({100.0 * singleton_count / max(len(counts), 1):.1f}%)"
            )
        print_section("TRAINING")
        log(f"Running the {args.mode} BPE trainer ...")
        if args.mode == "reference":
            tokenizer.train_reference(
                pieces,
                vocab_size=args.vocab_size,
                log=log,
            )
        else:
            tokenizer.train_from_counts(
                counts,
                vocab_size=args.vocab_size,
                min_freq=args.min_freq,
                log=log,
                checkpoint=checkpoint,
            )
        finished = time.perf_counter()
        del counts, pieces

        tokenizer.save(output_path)
        expected_merges = args.vocab_size - FIRST_MERGE_ID
        complete = (
            len(tokenizer.merges) == expected_merges
            and len(tokenizer.vocab) == args.vocab_size
        )
        print_section("TRAINING REPORT")
        log(
            f"Merges learned           : {len(tokenizer.merges):,} "
            f"(expected {expected_merges:,})"
        )
        log(f"Vocabulary size          : {len(tokenizer.vocab):,}")
        stats = tokenizer.train_stats
        log(
            f"Distinct pre-tokens kept : {stats['distinct_used']:,}/"
            f"{stats['distinct_total']:,}"
        )
        retained_percent = (
            100.0
            * stats["occurrences_used"]
            / max(stats["occurrences_total"], 1)
        )
        log(f"Occurrences retained    : {retained_percent:.3f}%")
        log(f"Minimum frequency       : {args.min_freq}")
        log(f"Counting duration       : {counted_at - started:.0f}s")
        log(f"Training duration       : {finished - counted_at:.0f}s")
        log(f"Total duration          : {(finished - started) / 3600:.2f}h")
        memory = peak_memory_mb()
        memory_text = (
            f"{memory:.0f} MB"
            if memory is not None
            else "check Task Manager / Resource Monitor"
        )
        log(f"Peak memory             : {memory_text}")
        log(f"Operating system        : {platform.platform()}")
        log(f"Processor               : {platform.processor()}")
        log(f"CPU cores               : {os.cpu_count()}")
        log(f"Python version          : {platform.python_version()}")
        log(f"Tokenizer saved to      : {output_path.resolve()}")
        if os.path.exists(partial_path):
            os.remove(partial_path)
        if not complete:
            log(
                "ERROR: training stopped before reaching the requested "
                "vocabulary size. "
                "The tokenizer was saved, but the corpus or training filters did not "
                "provide enough mergeable pairs."
            )
            return 1
        return 0
    finally:
        if log_file is not None:
            log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
