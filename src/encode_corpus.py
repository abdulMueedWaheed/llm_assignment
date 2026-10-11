import argparse
from contextlib import redirect_stdout
import os
from pathlib import Path
import sys
import time

import numpy as np
from tqdm import tqdm

from bpe import SPECIAL_ID, SPECIAL_TEXT, UrduBPETokenizer, open_text_file
from output_utils import (
    Tee,
    clock_time,
    show_section,
    show_title,
    show_value,
    time_taken,
)


OUTPUT_BUFFER_SIZE = 1_000_000
READ_BACK_CHUNK_SIZE = 50_000_000


def read_first_document(path: str | Path) -> str | None:
    collected_text = []
    character_count = 0

    with open_text_file(path) as input_file:
        while character_count < 5_000_000:
            text_part = input_file.read(1_000_000)
            if not text_part:
                break
            character_count += len(text_part)
            collected_text.append(text_part)
            full_text = "".join(collected_text)
            if SPECIAL_TEXT in full_text:
                return full_text.split(SPECIAL_TEXT, 1)[0]

    return None


def check_binary_file(
    tokenizer: UrduBPETokenizer,
    binary_path: Path,
    text_path: Path,
    token_count: int,
    marker_count: int,
) -> None:
    if binary_path.stat().st_size != token_count * 2:
        raise ValueError("binary size is not two bytes for each token")

    if token_count:
        token_array = np.memmap(binary_path, dtype=np.uint16, mode="r")
        highest_id = 0
        found_markers = 0

        for start in range(0, len(token_array), READ_BACK_CHUNK_SIZE):
            part = token_array[start:start + READ_BACK_CHUNK_SIZE]
            highest_id = max(highest_id, int(part.max()))
            found_markers += int(np.count_nonzero(part == SPECIAL_ID))
    else:
        token_array = np.empty(0, dtype=np.uint16)
        highest_id = 0
        found_markers = 0

    if highest_id >= len(tokenizer.vocabulary):
        raise ValueError(f"token id {highest_id} is not in the vocabulary")
    if found_markers != marker_count:
        raise ValueError("special marker count in the binary does not match text")

    show_section("BINARY CHECK")
    show_value("Result", "Passed")
    show_value("Data type", "uint16")
    show_value("Tokens", f"{token_count:,}")
    show_value("Highest token id", highest_id)
    show_value("Special markers", f"{found_markers:,}")

    first_document = read_first_document(text_path)
    if first_document is None:
        show_value("First document", "Check skipped; no marker found")
        return

    document_end = None
    for start in range(0, len(token_array), READ_BACK_CHUNK_SIZE):
        part = token_array[start:start + READ_BACK_CHUNK_SIZE]
        marker_positions = np.flatnonzero(part == SPECIAL_ID)
        if len(marker_positions):
            document_end = start + int(marker_positions[0])
            break

    if document_end is None:
        show_value("First document", "Check skipped; document is too long")
        return

    decoded = tokenizer.decode(map(int, token_array[:document_end]))
    if decoded != first_document:
        raise ValueError("the first document did not decode back to its text")
    show_value("First document", "Round trip passed")


def encode_one_file(
    tokenizer: UrduBPETokenizer,
    input_path: Path,
    output_path: Path,
    check_every: int,
) -> tuple[int, int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(str(output_path) + ".tmp")
    text_bytes = input_path.stat().st_size
    token_count = 0
    marker_count = 0
    token_buffer = []

    with open_text_file(input_path) as input_file:
        with open(temporary_path, "wb") as output_file:
            with tqdm(
                total=text_bytes,
                unit="B",
                unit_scale=True,
                dynamic_ncols=True,
                bar_format=(
                    "{l_bar}{bar:30} | {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}]"
                ),
            ) as progress:
                for line_number, line in enumerate(input_file, start=1):
                    encoded_ids = tokenizer.encode(line)

                    if check_every and line_number % check_every == 0:
                        if tokenizer.decode(encoded_ids) != line:
                            raise ValueError(
                                f"line {line_number} failed its round trip"
                            )

                    marker_count += line.count(SPECIAL_TEXT)
                    token_buffer.extend(encoded_ids)
                    progress.update(
                        len(line.encode("utf-8", errors="surrogatepass"))
                    )

                    if len(token_buffer) >= OUTPUT_BUFFER_SIZE:
                        if max(token_buffer) >= len(tokenizer.vocabulary):
                            raise ValueError(
                                "an encoded token id is outside the vocabulary"
                            )
                        np.asarray(token_buffer, dtype=np.uint16).tofile(
                            output_file
                        )
                        token_count += len(token_buffer)
                        token_buffer.clear()

                if token_buffer:
                    if max(token_buffer) >= len(tokenizer.vocabulary):
                        raise ValueError(
                            "an encoded token id is outside the vocabulary"
                        )
                    np.asarray(token_buffer, dtype=np.uint16).tofile(output_file)
                    token_count += len(token_buffer)

    check_binary_file(
        tokenizer,
        temporary_path,
        input_path,
        token_count,
        marker_count,
    )
    os.replace(temporary_path, output_path)
    return token_count, marker_count, text_bytes


def encode_with_report(
    tokenizer_path: Path,
    input_path: Path,
    output_path: Path,
    report_path: Path,
    check_every: int,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8", newline="\n") as report:
        with redirect_stdout(Tee(sys.stdout, report)):
            start_clock = clock_time()
            start_timer = time.perf_counter()
            finished = False

            show_title("URDU BPE CORPUS ENCODING")
            show_section(f"INPUT: {input_path.name}")
            show_value("Started at", start_clock)
            show_value("Text file", input_path.resolve())
            show_value("Tokenizer", tokenizer_path.resolve())
            show_value("Binary output", output_path.resolve())
            show_value("Text report", report_path.resolve())

            try:
                tokenizer = UrduBPETokenizer.load(tokenizer_path)
                token_count, marker_count, text_bytes = encode_one_file(
                    tokenizer,
                    input_path,
                    output_path,
                    check_every,
                )
                show_section("ENCODING RESULTS")
                show_value("Tokens written", f"{token_count:,}")
                show_value("Special markers", f"{marker_count:,}")
                show_value(
                    "Text bytes per token",
                    f"{text_bytes / max(token_count, 1):.3f}",
                )
                finished = True
            finally:
                end_clock = clock_time()
                seconds = time.perf_counter() - start_timer
                show_section("RUN TIME")
                show_value("Status", "Completed" if finished else "Failed")
                show_value("Finished at", end_clock)
                show_value("Elapsed time", time_taken(seconds))


def main() -> int:
    script_folder = Path(__file__).resolve().parent
    project_folder = script_folder.parent
    default_tokenizer = project_folder / "models" / "tokenizer.json"
    default_inputs = [
        project_folder / "data" / "processed" / "train.txt",
        project_folder / "data" / "processed" / "val.txt",
    ]

    parser = argparse.ArgumentParser(
        description="Encode train.txt and val.txt as uint16 files."
    )
    parser.add_argument("--tok", default=str(default_tokenizer))
    parser.add_argument(
        "--inp",
        help="encode just this input file instead of train.txt and val.txt",
    )
    parser.add_argument("--out", help="binary output path; requires --inp")
    parser.add_argument("--log", help="text report path; requires --inp")
    parser.add_argument("--check-every", type=int, default=2_000)
    args = parser.parse_args()

    if args.check_every < 0:
        parser.error("--check-every cannot be negative")
    if not Path(args.tok).is_file():
        parser.error(f"tokenizer file does not exist: {args.tok}")

    if args.inp:
        input_paths = [Path(args.inp)]
    else:
        if args.out or args.log:
            parser.error("--out and --log can only be used with --inp")
        input_paths = default_inputs

    missing_paths = [str(path) for path in input_paths if not path.is_file()]
    if missing_paths:
        parser.error("input file(s) not found: " + ", ".join(missing_paths))

    for input_path in input_paths:
        output_path = (
            Path(args.out)
            if args.out
            else project_folder / "data" / "encoded" / f"{input_path.stem}.bin"
        )
        report_path = (
            Path(args.log)
            if args.log
            else project_folder
            / "logs"
            / f"encode_corpus_{input_path.stem}.log"
        )
        encode_with_report(
            Path(args.tok),
            input_path,
            output_path,
            report_path,
            args.check_every,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
