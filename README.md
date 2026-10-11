# CS 417 Urdu BPE Tokenizer

Assignment 1 implementation and deliverables for a byte-level Urdu BPE tokenizer.

## Status

- The 50,000-entry tokenizer is trained with all 49,743 merges.
- Recorded full training took 337.1s to count and 173.7s to learn merges (510.8s total), using 2,332 MiB peak memory on the logged Linux host (Python 3.14.8, 12 cores).
- The real 1 MiB reference/fast check matched all 2,000 merges (266.0s reference, 1.0s fast); round-trip passed.
- `data/encoded/train.bin` and `data/encoded/val.bin` are flat `uint16` arrays.
- The final Part C report is `reports/report.pdf` (3 pages).
- The single corpus card is `DATA_CARD.md`.
- Large corpus and binary files are kept outside the source archive for shared-drive hosting.

## Layout

```text
src/
  normalize.py          Unicode normalization
  bpe.py                public BPETokenizer API and smoke test
  bpe_tokenizer.py      corpus handling, encode/decode, and serialization
  bpe_training.py       reference and heap-optimized BPE training
  bpe_tokens.py         token IDs, Urdu regex, byte merge, heap ordering
  train_tokenizer.py    required training entry point
  encode_corpus.py      required uint16 encoder entry point
scripts/
  scrape_corpus.py      custom HumSub collection
  clean_corpus.py       cleaning, deduplication, and document splits
  verify_raw_data.py   source file verification
  download_data.py      prepared corpus downloader
  test_bpe.py           tokenizer tests and real-sample comparison
  analyze_tokenizer.py  Part C analysis CLI
  analysis_merges.py    merge, UTF-8, and token-length analysis
  analysis_metrics.py   tokens-per-word comparison and test inputs
  output_utils.py       shared CLI output formatting
data/
  raw/                  locally collected/source material; not archived
  processed/            train.txt, val.txt, test.txt; shared-drive deliverables
  encoded/               train.bin, val.bin; shared-drive deliverables
models/
  tokenizer.json        trained model with 49,743 ordered merges
logs/                    training, encoding, and test logs
analysis/                generated Part C Markdown
reports/
  report.pdf             final report, at most four pages
docs/
  ARCHITECTURE.md        code and data flow
  ASSIGNMENT_CHECKLIST.md requirements audit and remaining actions
DATA_CARD.md             single corpus data card
README.md                this guide
```

## Setup and Checks

Run from the repository root:

```bash
python -m pip install -r requirements.txt
python src/normalize.py
python scripts/test_bpe.py
python scripts/test_bpe.py --train data/processed/train.txt --ref-merges 2000
```

## Rebuild the Tokenizer

Exact full-corpus training command; it reads `train.txt` and writes the model and log separately:

```bash
python src/train_tokenizer.py --train data/processed/train.txt --out models/tokenizer.json --log logs/train_tokenizer.log --vocab-size 50000 --mode fast --min-freq 1 --checkpoint-every 5000
```

Encode the required splits:

```bash
python src/encode_corpus.py --tok models/tokenizer.json
```

Rebuild Part C analysis:

```bash
python scripts/analyze_tokenizer.py --tokenizer models/tokenizer.json --train data/processed/train.txt --test data/processed/test.txt --output analysis/phase3_analysis.md
```

The analyzer uses `tiktoken` only for the allowed GPT-2 and `o200k_base` comparison.

## Corpus Commands

```bash
python scripts/scrape_corpus.py
python scripts/clean_corpus.py
python scripts/verify_raw_data.py
python scripts/download_data.py
```

The cleaner expects source files under `data/raw/` and writes the document-level splits under `data/processed/`. It updates only the root `DATA_CARD.md`.

## Submission

Archive the source (`src/`, `scripts/`, `docs/`), `models/tokenizer.json`, `DATA_CARD.md`, `README.md`, and `reports/report.pdf`. Provide a shared-drive link for:

- `data/processed/train.txt`
- `data/processed/val.txt`
- `data/processed/test.txt`
- `data/encoded/train.bin`
- `data/encoded/val.bin`
