import argparse
from contextlib import redirect_stdout
import os
import platform
import sys
import time
from pathlib import Path

from bpe import FIRST_MERGE_ID, UrduBPETokenizer
from output_utils import Tee, show_section, show_title, show_value


def peak_memory_mb() -> float | None:
    if sys.platform == "win32":
        return None

    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return peak / (1024 * 1024)
    return peak / 1024


def run_training(args: argparse.Namespace, log_path: Path) -> int:
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8", newline="\n") as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)):
            return train_and_save_tokenizer(args, output_path)


def train_and_save_tokenizer(
    args: argparse.Namespace,
    output_path: Path,
) -> int:
    start_time = time.perf_counter()
    tokenizer = UrduBPETokenizer()
    partial_path = Path(str(output_path) + ".partial")

    def save_checkpoint(merges: list[tuple[int, int]]) -> None:
        if args.checkpoint_every and len(merges) % args.checkpoint_every == 0:
            tokenizer.merges = list(merges)
            tokenizer.save(partial_path)

    show_title("URDU BPE TOKENIZER TRAINING")
    show_section("INPUTS AND SETTINGS")
    show_value("Training text", Path(args.train).resolve())
    show_value("Tokenizer file", output_path.resolve())
    show_value("Training version", args.mode)
    show_value("Vocabulary size", f"{args.vocab_size:,}")
    show_value("Minimum word frequency", args.min_freq)

    if args.mode == "reference":
        if args.min_freq != 1:
            raise ValueError("minimum frequency is only used by the fast version")
        word_pieces = list(
            tokenizer.read_words(args.train, max_bytes=args.max_bytes)
        )
        word_counts = None
        occurrence_count = len(word_pieces)
        distinct_count = None
    else:
        word_counts = tokenizer.count_words(
            args.train,
            max_bytes=args.max_bytes,
        )
        word_pieces = None
        occurrence_count = sum(word_counts.values())
        distinct_count = len(word_counts)

    count_finished = time.perf_counter()
    show_section("CORPUS COUNTS")
    distinct_text = (
        f"{distinct_count:,}" if distinct_count is not None else "not counted"
    )
    show_value("Distinct words", distinct_text)
    show_value("Word occurrences", f"{occurrence_count:,}")
    show_value("Counting time", f"{count_finished - start_time:.1f} seconds")

    if word_counts is not None:
        singleton_count = sum(count == 1 for count in word_counts.values())
        show_value("Words seen once", f"{singleton_count:,}")

    show_section("TRAINING")
    print("Training the tokenizer. Merge progress appears every 1,000 merges.")

    if args.mode == "reference":
        tokenizer.train_reference(
            word_pieces,
            vocabulary_size=args.vocab_size,
            log_every=1_000,
        )
    else:
        tokenizer.train_fast(
            word_counts,
            vocabulary_size=args.vocab_size,
            min_frequency=args.min_freq,
            log_every=1_000,
            checkpoint=save_checkpoint,
        )

    training_finished = time.perf_counter()
    tokenizer.save(output_path)
    expected_merges = args.vocab_size - FIRST_MERGE_ID
    training_complete = (
        len(tokenizer.merges) == expected_merges
        and len(tokenizer.vocabulary) == args.vocab_size
    )

    show_section("TRAINING RESULTS")
    show_value(
        "Merges learned",
        f"{len(tokenizer.merges):,} (expected {expected_merges:,})",
    )
    show_value("Vocabulary size", f"{len(tokenizer.vocabulary):,}")
    show_value("Tokenizer file", output_path.resolve())
    show_value("Counting time", f"{count_finished - start_time:.1f} seconds")
    show_value(
        "Training time",
        f"{training_finished - count_finished:.1f} seconds",
    )
    show_value(
        "Total time",
        f"{(training_finished - start_time) / 3600:.2f} hours",
    )
    memory = peak_memory_mb()
    memory_text = (
        f"{memory:.0f} MB"
        if memory is not None
        else "see Windows Task Manager"
    )
    show_value("Peak memory", memory_text)
    show_value("Computer", platform.platform())
    show_value("Processor", platform.processor())
    show_value("CPU cores", os.cpu_count())
    show_value("Python version", platform.python_version())

    if partial_path.exists():
        partial_path.unlink()

    if not training_complete:
        print(
            "Training stopped before the requested vocabulary size. "
            "Check the input text and merge counts."
        )
        return 1

    print("Training completed successfully.")
    return 0


def main() -> int:
    script_folder = Path(__file__).resolve().parent
    project_folder = script_folder.parent
    default_train = project_folder / "data" / "processed" / "train.txt"
    default_output = script_folder / "outputs" / "tokenizer.json"
    default_log = script_folder / "outputs" / "train_tokenizer_log.txt"

    parser = argparse.ArgumentParser(
        description="Train the beginner-friendly Urdu BPE tokenizer."
    )
    parser.add_argument("--train", default=str(default_train))
    parser.add_argument("--out", default=str(default_output))
    parser.add_argument("--log", default=str(default_log))
    parser.add_argument("--vocab-size", type=int, default=50_000)
    parser.add_argument(
        "--mode",
        choices=("reference", "fast"),
        default="fast",
    )
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--max-bytes", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    args = parser.parse_args()

    if not Path(args.train).is_file():
        parser.error(f"training file does not exist: {args.train}")
    if args.vocab_size <= FIRST_MERGE_ID or args.vocab_size > 65_536:
        parser.error("vocabulary size must be between 258 and 65,536")
    if args.min_freq < 1:
        parser.error("minimum word frequency must be at least 1")
    if args.checkpoint_every < 0:
        parser.error("checkpoint interval cannot be negative")

    return run_training(args, Path(args.log))


if __name__ == "__main__":
    raise SystemExit(main())
