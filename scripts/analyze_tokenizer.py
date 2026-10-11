import argparse
import json
from pathlib import Path
import re
import sys

import tiktoken

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from analysis_merges import MergeAnalysis, merge_analysis
from analysis_metrics import (
    BAD_CASE_REASONS,
    HAND_INPUTS,
    TokenMetrics,
    markdown_cell,
    measure_test_file,
    measure_text,
)
from bpe import FIRST_MERGE_ID, UrduBPETokenizer


def create_report(
    tokenizer_path: Path,
    training_summary: str,
    tokenizer: UrduBPETokenizer,
    merge_data: MergeAnalysis,
    test_metrics: TokenMetrics,
    hand_metrics: list[tuple[str, str, TokenMetrics]],
) -> str:
    final_model = len(tokenizer.merges) == 49_743
    status = "Final 50,000-token tokenizer" if final_model else "PRELIMINARY trial tokenizer"
    lines = [
        "# Phase 3: Tokenizer Analysis",
        "",
        f"**Status:** {status} ({len(tokenizer.vocabulary):,} vocabulary entries).",
        f"**Tokenizer:** `{tokenizer_path}`.",
        "",
        "## Training Runtime",
        "",
        training_summary,
        "",
        "## 1. Merge Analysis",
        "",
        "| Merge | Input IDs | Bytes | Valid UTF-8 alone | Decoded token |",
        "|---:|---|---:|:---:|---|",
    ]
    for merge_number, pair, byte_length, valid_utf8, decoded in merge_data["rows"]:
        lines.append(
            f"| {merge_number:,} | {pair} | {byte_length} | {valid_utf8} | {markdown_cell(decoded)} |"
        )

    letters = ", ".join(
        f"{number}: {json.dumps(letter, ensure_ascii=False)}"
        for number, letter in merge_data["two_byte_letters"]
    ) or "None in the first 50 merges."
    invalid = ", ".join(map(str, merge_data["invalid_early"])) or "None in the first 50 merges."
    first_word = merge_data["first_whole_word"]
    first_word_text = (
        f"Merge {first_word[0]:,} first forms the multi-letter corpus word {json.dumps(first_word[1], ensure_ascii=False)}."
        if first_word
        else "No whole lexical word was found in the available training text."
    )
    lines.extend(
        [
            "",
            f"Two-byte Urdu letters rebuilt in the first 50: {letters}",
            f"Early merges invalid as standalone UTF-8 (first 50): {invalid}",
            f"Letter-forming merges rebuild UTF-8 Urdu characters; merges {invalid} are partial byte sequences and cannot be decoded alone.",
            first_word_text,
            "",
            "### 20 Longest Learned Tokens",
            "",
            "| Token ID | UTF-8 bytes | Token |",
            "|---:|---:|---|",
        ]
    )
    for token_id, byte_length, decoded in merge_data["longest"]:
        lines.append(f"| {token_id} | {byte_length} | {markdown_cell(decoded)} |")

    lines.extend(
        [
            "",
            "## 2. Tokens per Word",
            "",
            "Word counts use whitespace-delimited words; end-of-text markers are excluded.",
            "",
            "| Input | Words | Urdu BPE tokens/word | GPT-2 tokens/word | o200k_base tokens/word |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    all_metrics = [("test.txt", test_metrics)] + [
        (name, metrics) for name, _, metrics in hand_metrics
    ]
    for name, metrics in all_metrics:
        ratios = metrics["ratios"]
        lines.append(
            f"| {markdown_cell(name)} | {metrics['words']:,} | "
            f"{ratios['Urdu BPE']:.3f} | {ratios['GPT-2']:.3f} | {ratios['o200k_base']:.3f} |"
        )
    lines.extend(
        [
            "",
            (
                f"On test.txt, Urdu BPE used {test_metrics['ratios']['Urdu BPE']:.3f} tokens per word, "
                f"versus {test_metrics['ratios']['GPT-2']:.3f} for GPT-2 and "
                f"{test_metrics['ratios']['o200k_base']:.3f} for o200k_base."
            ),
            "Roman Urdu, rare spellings, and mixed-script inputs remain useful stress cases.",
            "",
            "## 3. Inputs Handled Poorly",
            "",
            "The three selected inputs with the highest Urdu-BPE token/word ratio are:",
            "",
            "| Input type | Urdu BPE tokens/word | Example | Interpretation |",
            "|---|---:|---|---|",
        ]
    )
    worst = sorted(
        hand_metrics,
        key=lambda item: item[2]["ratios"]["Urdu BPE"],
        reverse=True,
    )[:3]
    for name, text, metrics in worst:
        lines.append(
            f"| {name} | {metrics['ratios']['Urdu BPE']:.3f} | "
            f"{markdown_cell(json.dumps(text, ensure_ascii=False))} | "
            f"{BAD_CASE_REASONS[name]} |"
        )
    lines.extend(
        [
            "",
            f"{worst[0][0]} was hardest at {worst[0][2]['ratios']['Urdu BPE']:.3f} tokens per word.",
            "The examples are diagnostic rather than a representative evaluation set.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a trained Urdu BPE tokenizer.")
    parser.add_argument("--tokenizer", type=Path, default=PROJECT_DIR / "models" / "tokenizer.json")
    parser.add_argument("--train", type=Path, default=PROJECT_DIR / "data" / "processed" / "train.txt")
    parser.add_argument("--test", type=Path, default=PROJECT_DIR / "data" / "processed" / "test.txt")
    parser.add_argument("--training-log", type=Path, default=PROJECT_DIR / "logs" / "train_tokenizer.log")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "analysis" / "phase3_analysis.md")
    args = parser.parse_args()

    for path, label in (
        (args.tokenizer, "tokenizer"),
        (args.train, "training corpus"),
        (args.test, "test corpus"),
    ):
        if not path.is_file():
            parser.error(f"{label} file does not exist: {path}")

    tokenizer = UrduBPETokenizer.load(args.tokenizer)
    training_log = args.training_log.read_text(encoding="utf-8") if args.training_log.is_file() else ""

    def log_value(label: str, default: str = "not recorded") -> str:
        match = re.search(
            rf"^{re.escape(label)}[ \t]*:[ \t]*(.+)$",
            training_log,
            re.MULTILINE,
        )
        return match.group(1).strip() if match and match.group(1).strip() else default

    count_seconds = re.search(r"^Counting time\s*:\s*([\d.]+) seconds", training_log, re.MULTILINE)
    train_seconds = re.search(r"^Training time\s*:\s*([\d.]+) seconds", training_log, re.MULTILINE)
    if count_seconds and train_seconds:
        elapsed = float(count_seconds.group(1)) + float(train_seconds.group(1))
        runtime = (
            f"Full-corpus training used {float(count_seconds.group(1)):.1f}s for counting "
            f"and {float(train_seconds.group(1)):.1f}s for merge learning ({elapsed:.1f}s total). "
            f"Recorded system: {log_value('Computer')}; Python {log_value('Python version')}; "
            f"{log_value('CPU cores')} CPU cores; peak memory {log_value('Peak memory')}. "
            f"CPU model: {log_value('Processor', 'not reported by platform.processor()')}."
        )
    else:
        runtime = f"Training runtime metadata was not available in `{args.training_log}`."

    merge_data = merge_analysis(tokenizer, args.train)
    encodings = {
        "GPT-2": tiktoken.get_encoding("gpt2"),
        "o200k_base": tiktoken.get_encoding("o200k_base"),
    }
    test_metrics = measure_test_file(args.test, tokenizer, encodings)
    hand_metrics = [
        (name, text, measure_text(text, tokenizer, encodings))
        for name, text in HAND_INPUTS
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        create_report(args.tokenizer, runtime, tokenizer, merge_data, test_metrics, hand_metrics),
        encoding="utf-8",
    )
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Vocabulary size: {len(tokenizer.vocabulary):,}")
    print(f"Test words: {test_metrics['words']:,}")
    print(f"Analysis report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
