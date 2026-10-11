# Urdu BPE Tokenizer from Scratch

**Course**: CS 417: Large Language Model (Spring 2026)  
**Institution**: School of Electrical Engineering and Computer Science (SEECS), NUST  
**Assignment**: Assignment 1 — Build an Urdu BPE Tokenizer from Scratch  

This repository contains the code and documentation for building an Urdu Byte-Pair Encoding (BPE) tokenizer from the ground up, starting with corpus collection and Unicode normalization, followed by a custom BPE training implementation with a 50,000 token vocabulary.

---

## Current Status: Phase 1 Complete; Part B Verified; Part C In Progress

Phase 1 deliverables have been built and verified:
- **Corpus Volume**: 1.95 GB total cleaned Urdu text (~240 million words), exceeding the 500 MB minimum.
- **Source Diversity**: 3 sources across 3 distinct genres:
  1. **HumSub** (51.9 MB raw / 51.8 MB clean) — *Literature & Blogs* (collected via WordPress REST API with 5,279 articles logged as evidence).
  2. **Urdu Wikipedia** (359.6 MB raw / 351.3 MB clean) — *Encyclopedia* (extracted from Wikimedia dumps).
  3. **CC-100 Slice** (1,534.9 MB raw / 1,524.1 MB clean) — *Web Crawl* (Common Crawl).
- **Custom Normalizer (`normalize.py`)**: Handles Urdu keyboard inconsistencies (Arabic Yeh/Kaf/Heh mapping, Tatweel removal, ASCII digit standardization, and ZWNJ preservation).
- **Pipeline (`clean_corpus.py`)**: Strips markup, filters non-Urdu text (>=60% Urdu script threshold), deduplicates documents using SHA-256, and splits the data.
- **Splits**: Document-level split with `random.seed(42)` and `<|endoftext|>` delimiters:
  - `train.txt`: 98% (1,912.47 MB, 721,491 documents)
  - `val.txt`: 1% (19.49 MB, 7,362 documents)
  - `test.txt`: 1% (19.26 MB, 7,363 documents)
- **Data Card**: Full metrics and breakdown documented in [`DATA_CARD.md`](DATA_CARD.md).

---

## Repository Structure

```text
llm_assignment/
├── docs/
│   ├── Assignment 1 - Urdu BPE Tokenizer from Scratch.pdf  # Course assignment brief
│   └── EXECUTION_PLAN.md                                    # End-to-end implementation plan
├── reports/
│   ├── DATA_CARD.md                                        # Official 1-page data card
│   ├── raw_data_verify.txt                                 # Raw data verification log
│   └── scrape_log.json                                     # Custom scraping evidence log
├── scripts/
│   ├── scrape_corpus.py                                    # Scraper for HumSub REST API (>= 50 MB)
│   ├── clean_corpus.py                                     # Cleaning, deduplication, and splitting pipeline
│   └── verify_raw_data.py                                  # Raw dataset sanity and encoding checker
├── normalize.py                                            # Standalone Urdu Unicode normalizer
├── BPE_urdu/
│   ├── bpe.py                                              # Urdu-aware byte-level BPE tokenizer
│   ├── test_bpe.py                                         # Tokenizer and merge-equivalence tests
│   ├── train_tokenizer.py                                  # Train and save tokenizer.json
│   ├── encode_corpus.py                                    # Export train.bin / val.bin as uint16
│   └── analyze_tokenizer.py                                # Part C tokenizer analysis
├── requirements.txt                                        # Python dependencies
├── .gitignore                                              # Git ignore rules for datasets & cache
└── README.md                                               # Project overview and instructions
```

---

## Setup & Reproduction

### 1. Requirements

Install dependencies using Python 3.10+:

```bash
pip install -r requirements.txt
```

### 2. Verify Normalization

Run the built-in self-tests in `normalize.py`:

```bash
python normalize.py
```

### 3. Collect Custom Data (HumSub API)

To reproduce the custom collection of >= 50 MB raw Urdu articles:

```bash
python scripts/scrape_corpus.py
```

This writes raw text to `data/raw/scraped/humsub_raw.txt` and an audit trail to `reports/scrape_log.json`.

### 4. Run the Cleaning Pipeline

To clean the raw sources, apply normalization, deduplicate, and generate the final `train.txt`, `val.txt`, and `test.txt` splits:

```bash
python scripts/clean_corpus.py
```

This updates `DATA_CARD.md` with exact byte reductions and document totals.

### 5. Implement Part B (Urdu byte-level BPE)

Run these commands from the `BPE_urdu` directory in a Linux shell. The first
command runs the sanity and equivalence checks on the same 1 MB sample, then
automatically writes its report to `outputs/test_bpe_results.txt`. Run a
smaller measurement trial before committing to full-corpus training. Generated
Part B text reports and the tokenizer JSON are kept in `outputs/`; final
encoded arrays go in `bin_outputs/`.

```bash
cd BPE_urdu
python bpe.py
python test_bpe.py --train ../data/processed/train.txt --ref-merges 2000
python train_tokenizer.py --train ../data/processed/train.txt --out outputs/tok_trial.json --log outputs/train_tokenizer_trial_log.txt --max-bytes 50000000 --vocab-size 20000
```

Running `bpe.py` directly performs a small built-in test and saves
`outputs/bpe_test.txt`; it does not train on the corpus.
`test_bpe.py` saves `outputs/test_bpe_results.txt`.

After the trial, train the required 50,000-token vocabulary and encode the
training and validation splits. The tokenizer trainer writes
`outputs/train_tokenizer_log.txt`; the encoder automatically writes
`outputs/encode_corpus_train_log.txt` and
`outputs/encode_corpus_val_log.txt`. Without explicit output arguments,
training saves `outputs/tokenizer.json`; encoding saves `bin_outputs/train.bin`
and `bin_outputs/val.bin`.

```bash
python train_tokenizer.py
python encode_corpus.py
```

Run the Phase 3 analysis after training the final 50,000-token tokenizer:

```bash
python analyze_tokenizer.py
```

The available tokenizer is currently a 20,000-token trial. To reproduce the
preliminary analysis on it, run:

```bash
python analyze_tokenizer.py --tokenizer outputs/tok_trial.json --output ../reports/phase3_trial_analysis.md
```

With no arguments, `train_tokenizer.py` uses the project
`data/processed/train.txt`, writes the tokenizer to `outputs/tokenizer.json`,
and writes its log to `outputs/train_tokenizer_log.txt`. Pass `--train` to
select a different training file. With no arguments, `encode_corpus.py` loads
that tokenizer, encodes both `train.txt` and `val.txt`, and writes
`bin_outputs/train.bin` and `bin_outputs/val.bin` plus a separate report for
each split in `outputs/`. Each encoding report records the local start time,
finish time, and elapsed duration. Use `--inp` to encode only one input file.

Training uses `train.txt` only. Keep `--min-freq 1` to use all pre-tokens;
raising it changes the training distribution and must be disclosed in the report.
The tokenizer preserves text exactly and treats `<|endoftext|>` as reserved ID 256.

---

## Dataset Access

Due to size constraints (~1.95 GB), the raw and processed text files are hosted externally on Google Drive:
- **Processed Splits (`train.txt`, `val.txt`, `test.txt`)**: Available via shared project drive (`LLM-Project/data/processed/`).

---

## Remaining Work

- Train the 50,000-token vocabulary; confirm all 49,743 merges were learned.
- Encode `train.txt` and `val.txt`; retain the logs and report the binary token counts.
- Replace the preliminary Phase 3 analysis with results from the final 50,000-token tokenizer and produce the four-page PDF report.
