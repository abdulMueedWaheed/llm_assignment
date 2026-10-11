# Urdu Byte-Level BPE Tokenizer

CS 417 Assignment 1: a cleaned Urdu corpus and a byte-level BPE tokenizer trained to 50,000 vocabulary entries. The corpus cleaner keeps document boundaries; the tokenizer pre-tokenizes words, learns byte merges, and preserves input text during encode/decode. Separate analysis compares its merge behavior and tokens-per-word with GPT-2 and `o200k_base`.

## Requirements

- Python 3.10 or newer
- Packages in `requirements.txt`
- The corpus text files for training, encoding, and analysis; large data files are not tracked by Git

From the repository root, create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\\Scripts\\activate` instead. To fetch the prepared text files from the configured shared Drive folder, run:

```bash
python scripts/download_data.py --subfolder processed
```

This downloads text files into `data/processed/` if the folder is accessible. Otherwise obtain `train.txt`, `val.txt`, and `test.txt` from the course/shared-drive location and place them there. Training needs `train.txt`; encoding needs train and validation; analysis needs train and test. The current checkout has these processed splits, but not the Wikipedia or CC-100 raw inputs.

## Run and Test

Run commands from the repository root. The normalizer has a small built-in check; the BPE test covers the sanity example, Urdu-aware pre-tokenization, lossless round trips, serialization, and reference/fast equivalence on generated small cases.

```bash
python src/normalize.py
python scripts/test_bpe.py
```

Compare reference and optimized training on the first 1 MB of the corpus (2,000 merges; reference mode is intentionally slow):

```bash
python scripts/test_bpe.py --train data/processed/train.txt --ref-merges 2000
```

Check source files and the HumSub collection audit log with:

```bash
python scripts/verify_raw_data.py
```

That command only checks the sources listed in the script; it reports missing raw files rather than downloading them.

## Rebuild Outputs

Train exactly 50,000 vocabulary entries (256 byte IDs, special ID 256, then 49,743 merges). This reads only `train.txt` and writes the model plus a full run log:

```bash
python src/train_tokenizer.py --train data/processed/train.txt --out models/tokenizer.json --log logs/train_tokenizer.log --vocab-size 50000 --mode fast --min-freq 1 --checkpoint-every 5000
```

Encode train and validation to flat `uint16` arrays under `data/encoded/`:

```bash
python src/encode_corpus.py --tok models/tokenizer.json
```

Generate the detailed Part C analysis (requires the test split and `tiktoken`, installed from requirements):

```bash
python scripts/analyze_tokenizer.py --tokenizer models/tokenizer.json --train data/processed/train.txt --test data/processed/test.txt --output analysis/phase3_analysis.md
```

The polished assignment report is `reports/report.pdf`; its editable source is `reports/report.html`. Full-corpus training and encoding take time and need enough free disk space for the corpus, checkpoints, logs, and binary outputs.

## Corpus Rebuild

The cleaner expects these raw inputs:

- `data/raw/scraped/humsub_raw.txt`
- `data/raw/wikipedia/wikipedia_raw.txt`
- `data/raw/cc100/ur.txt`

After placing the sources at those paths, run:

```bash
python scripts/clean_corpus.py
```

It strips limited wiki markup, normalizes text, filters documents below a 60% Urdu-character ratio, deduplicates exact documents, shuffles with seed 42, and writes 98/1/1 document splits with `<|endoftext|>` delimiters. The HumSub scraper is `python scripts/scrape_corpus.py`; it uses the WordPress API and appends to its output. It does not automatically check `robots.txt`, so review the site’s current robots rules and terms before collecting; do not rerun against an existing raw file unless you intend to append. Raw-source licenses and redistribution terms must be verified before submission.

## Project Map

- `src/`: normalizer, tokenizer implementation, training and encoding entry points
- `scripts/`: corpus collection/cleaning, checks, and analysis
- `data/`: raw inputs, processed splits, and encoded arrays; large data is ignored by Git
- `models/`, `logs/`, `analysis/`, `reports/`: model and generated run/analysis artifacts
- `DATA_CARD.md`: corpus sources, sizes, cleaning, and known biases
- `docs/ASSIGNMENT_CHECKLIST.md`: requirement-by-requirement evidence and open actions
- `explanation/`: local module walkthroughs; intentionally ignored by Git

Module-level explanations are indexed in `explanation/README.md`.

## Submission

Archive the code, `models/tokenizer.json`, `DATA_CARD.md`, `README.md`, and `reports/report.pdf`. Provide a shared-drive link for `train.txt`, `val.txt`, `test.txt`, `train.bin`, and `val.bin`. Do not publish raw source text unless its terms explicitly allow redistribution.
