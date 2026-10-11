# Urdu BPE Tokenizer from Scratch

**Course**: CS 417: Large Language Model (Spring 2026)  
**Institution**: School of Electrical Engineering and Computer Science (SEECS), NUST  
**Assignment**: Assignment 1 — Build an Urdu BPE Tokenizer from Scratch  

This repository contains the code and documentation for building an Urdu Byte-Pair Encoding (BPE) tokenizer from the ground up, starting with corpus collection and Unicode normalization, followed by a custom BPE training implementation with a 50,000 token vocabulary.

---

## Current Status: Phase 1 (Corpus Collection & Cleaning) — Complete

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

---

## Dataset Access

Due to size constraints (~1.95 GB), the raw and processed text files are hosted externally on Google Drive:
- **Processed Splits (`train.txt`, `val.txt`, `test.txt`)**: Available via shared project drive (`LLM-Project/data/processed/`).

---

## Next Steps

- **Phase 2**: Pre-tokenization regular expression design for Urdu words, diacritics, and ZWNJ.
- **Phase 3**: Byte-level BPE tokenizer implementation from scratch (Reference and Fast versions) trained to 50,000 vocabulary tokens.
- **Phase 4**: Corpus encoding to binary flat arrays (`train.bin`, `val.bin`).
- **Phase 5**: Analysis, compression benchmarking vs. GPT-2/o200k_base, and final report.
