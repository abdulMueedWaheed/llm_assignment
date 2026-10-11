"""Encode corpus text into a flat uint16 token array."""

import argparse
from contextlib import redirect_stdout
import os
from pathlib import Path
import sys
import time

import numpy as np
from tqdm import tqdm

from bpe import BPETokenizer, SPECIAL, SPECIAL_ID, open_text
from output_utils import (
    Tee,
    format_duration,
    local_timestamp,
    print_field,
    print_header,
    print_section,
)

TOKEN_CHUNK = 50_000_000
OUTPUT_BUFFER_TOKENS = 1_000_000


def first_document(path: str | Path, max_chars: int = 5_000_000) -> str | None:
    """Read the first document's text, bounded to avoid huge verification reads."""
    pieces = []
    chars_read = 0
    with open_text(path) as source:
        while chars_read < max_chars:
            piece = source.read(min(1_000_000, max_chars - chars_read))
            if not piece:
                break
            chars_read += len(piece)
            pieces.append(piece)
            joined = "".join(pieces)
            if SPECIAL in joined:
                return joined.split(SPECIAL, 1)[0]
    return None


def verify_bin(
    tokenizer: BPETokenizer,
    binary_path: str | Path,
    text_path: str | Path,
    expected_tokens: int,
    expected_specials: int,
) -> None:
    if os.path.getsize(binary_path) != 2 * expected_tokens:
        raise ValueError("binary file size is not two bytes per token")
    if expected_tokens:
        data = np.memmap(binary_path, dtype=np.uint16, mode="r")
        if len(data) != expected_tokens:
            raise ValueError("token count differs from the number written")
        max_id = 0
        special_count = 0
        for start in range(0, len(data), TOKEN_CHUNK):
            chunk = data[start:start + TOKEN_CHUNK]
            max_id = max(max_id, int(chunk.max()))
            special_count += int(np.count_nonzero(chunk == SPECIAL_ID))
    else:
        data = np.empty(0, dtype=np.uint16)
        max_id = 0
        special_count = 0

    if max_id >= len(tokenizer.vocab):
        raise ValueError(f"token ID {max_id} is outside the vocabulary")
    if special_count != expected_specials:
        raise ValueError(
            f"binary has {special_count} end-of-text IDs; text has {expected_specials}"
        )
    print_field("Binary validation", "Passed")
    print_field("Data type", "uint16")
    print_field("Token count", f"{expected_tokens:,}")
    print_field("Maximum token ID", max_id)
    print_field("End-of-text markers", f"{special_count:,}")

    document = first_document(text_path)
    if document is None:
        print_field("First-document check", "Skipped (no complete delimiter in range)")
        return
    document_end = None
    for start in range(0, len(data), TOKEN_CHUNK):
        chunk = data[start:start + TOKEN_CHUNK]
        delimiters = np.flatnonzero(chunk == SPECIAL_ID)
        if len(delimiters):
            document_end = start + int(delimiters[0])
            break
    if document_end is None:
        print_field(
            "First-document check",
            "Skipped (document exceeds verification limit)",
        )
        return
    decoded = tokenizer.decode(map(int, data[:document_end]))
    if decoded != document:
        raise ValueError("first document does not round-trip from the binary file")
    print_field("First-document check", f"Passed ({len(document):,} characters)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    parser.add_argument(
        "--tok",
        default=str(script_dir / "outputs" / "tokenizer.json"),
        help="trained tokenizer JSON (default: outputs/tokenizer.json)",
    )
    parser.add_argument(
        "--inp",
        help="input text file (default: encode both processed train.txt and val.txt)",
    )
    parser.add_argument(
        "--out",
        help="output uint16 binary file (default: bin_outputs/<input-name>.bin)",
    )
    parser.add_argument(
        "--log",
        help="text report path (default: outputs/encode_corpus_<input>_log.txt)",
    )
    parser.add_argument(
        "--check-every",
        type=int,
        default=2_000,
        help="round-trip every Nth line while encoding (0 disables line checks)",
    )
    args = parser.parse_args()
    if args.check_every < 0:
        parser.error("--check-every cannot be negative")

    if args.inp:
        inputs = [Path(args.inp)]
    else:
        if args.out:
            parser.error("--out requires --inp when encoding a single file")
        if args.log:
            parser.error("--log requires --inp when encoding a single file")
        inputs = [
            project_dir / "data" / "processed" / "train.txt",
            project_dir / "data" / "processed" / "val.txt",
        ]

    missing_inputs = [str(path) for path in inputs if not path.is_file()]
    if missing_inputs:
        parser.error("input text file(s) not found: " + ", ".join(missing_inputs))
    if not Path(args.tok).is_file():
        parser.error(f"tokenizer file not found: {args.tok}")

    for input_path in inputs:
        output_path = Path(args.out) if args.out else (
            script_dir / "bin_outputs" / f"{input_path.stem}.bin"
        )
        log_path = Path(args.log) if args.log else (
            script_dir
            / "outputs"
            / f"encode_corpus_{input_path.stem}_log.txt"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_args = argparse.Namespace(
            tok=args.tok,
            inp=str(input_path),
            out=str(output_path),
            check_every=args.check_every,
        )
        with open(log_path, "w", encoding="utf-8", newline="\n") as report:
            with redirect_stdout(Tee(sys.stdout, report)):
                started_at = local_timestamp()
                started_clock = time.perf_counter()
                completed = False
                print_header("URDU BPE CORPUS ENCODING")
                print_section(f"INPUT: {input_path.name}")
                print_field("Started at", started_at)
                print_field("Input text", input_path.resolve())
                print_field("Tokenizer", Path(args.tok).resolve())
                print_field("Binary output", output_path.resolve())
                print_field("Text report", log_path.resolve())
                try:
                    encode(file_args, log_path)
                    completed = True
                finally:
                    finished_at = local_timestamp()
                    elapsed = time.perf_counter() - started_clock
                    print_section("RUN SUMMARY")
                    print_field("Status", "Completed" if completed else "Failed")
                    print_field("Finished at", finished_at)
                    print_field("Elapsed time", format_duration(elapsed))
    return 0


def encode(args: argparse.Namespace, log_path: Path) -> int:
    tokenizer = BPETokenizer.load(args.tok)
    if len(tokenizer.vocab) > 65_536:
        raise ValueError("token IDs do not fit in uint16")

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(str(output_path) + ".tmp")
    token_buffer: list[int] = []
    token_total = 0
    special_total = 0
    input_size = os.path.getsize(args.inp)

    print_section("ENCODING PROGRESS")
    with open_text(args.inp) as source, open(temporary_path, "wb") as output, tqdm(
        total=input_size,
        unit="B",
        unit_scale=True,
        dynamic_ncols=True,
        bar_format=(
            "{l_bar}{bar:30} | {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}]"
        ),
    ) as progress:
        for line_number, line in enumerate(source, start=1):
            ids = tokenizer.encode(line)
            if args.check_every and line_number % args.check_every == 0:
                if tokenizer.decode(ids) != line:
                    raise ValueError(f"round-trip failed on input line {line_number}")
            special_total += line.count(SPECIAL)
            token_buffer.extend(ids)
            progress.update(len(line.encode("utf-8", errors="surrogatepass")))
            if len(token_buffer) >= OUTPUT_BUFFER_TOKENS:
                if token_buffer and max(token_buffer) >= len(tokenizer.vocab):
                    raise ValueError("encoded token ID is outside the vocabulary")
                np.asarray(token_buffer, dtype=np.uint16).tofile(output)
                token_total += len(token_buffer)
                token_buffer.clear()
        if token_buffer:
            if max(token_buffer) >= len(tokenizer.vocab):
                raise ValueError("encoded token ID is outside the vocabulary")
            np.asarray(token_buffer, dtype=np.uint16).tofile(output)
            token_total += len(token_buffer)

    verify_bin(tokenizer, temporary_path, args.inp, token_total, special_total)
    os.replace(temporary_path, output_path)
    print_section("ENCODING RESULTS")
    print_field("Input file", args.inp)
    print_field("Tokens written", f"{token_total:,}")
    print_field("Binary file", output_path)
    print_field("Text bytes per token", f"{input_size / max(token_total, 1):.3f}")
    print_field("Encoding report", log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
