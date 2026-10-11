"""Encode corpus text into a flat uint16 token array."""

import argparse
from contextlib import redirect_stdout
import os
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

from bpe import BPETokenizer, SPECIAL, SPECIAL_ID, open_text
from output_utils import Tee

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
    print(
        f"read-back OK: dtype=uint16, {expected_tokens:,} tokens, "
        f"max ID {max_id}, {special_count:,} end-of-text IDs"
    )

    document = first_document(text_path)
    if document is None:
        print("first-document check skipped: no complete special-token delimiter in range")
        return
    document_end = None
    for start in range(0, len(data), TOKEN_CHUNK):
        chunk = data[start:start + TOKEN_CHUNK]
        delimiters = np.flatnonzero(chunk == SPECIAL_ID)
        if len(delimiters):
            document_end = start + int(delimiters[0])
            break
    if document_end is None:
        print("first-document check skipped: first document exceeds verification chunk")
        return
    decoded = tokenizer.decode(map(int, data[:document_end]))
    if decoded != document:
        raise ValueError("first document does not round-trip from the binary file")
    print(f"first document ({len(document):,} characters) round-trips exactly")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tok", required=True, help="trained tokenizer JSON")
    parser.add_argument("--inp", required=True, help="input text file")
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

    output_path = Path(args.out) if args.out else (
        Path(__file__).resolve().parent
        / "bin_outputs"
        / f"{Path(args.inp).stem}.bin"
    )
    args.out = str(output_path)
    log_path = Path(args.log) if args.log else (
        Path(__file__).resolve().parent
        / "outputs"
        / f"encode_corpus_{Path(args.inp).stem}_log.txt"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8", newline="\n") as report:
        with redirect_stdout(Tee(sys.stdout, report)):
            return encode(args, log_path)


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

    with open_text(args.inp) as source, open(temporary_path, "wb") as output, tqdm(
        total=input_size, unit="B", unit_scale=True
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
    print(f"{args.inp}: {token_total:,} tokens -> {output_path}")
    print(f"bytes of text per token: {input_size / max(token_total, 1):.3f}")
    print(f"encoding report saved to: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
