# Architecture and Code Map

## Repository Boundaries

The project uses directories by artifact role, not by assignment phase:

- `src/` contains the required tokenizer implementation and its public command-line entry points.
- `scripts/` contains collection, cleaning, verification, tests, and Part C analysis.
- `data/raw/` contains collected/downloaded sources; `data/processed/` contains the three text splits; `data/encoded/` contains their binary model inputs.
- `models/` contains the serialized tokenizer.
- `logs/` contains human-readable run and test logs.
- `analysis/` contains generated Part C tables and interpretations.
- `reports/` contains the final PDF.

## BPE Modules

The public surface is `src/bpe.py`. It defines `BPETokenizer` and preserves `UrduBPETokenizer` as an alias for existing callers. The implementation is separated into three related modules:

- `src/bpe_tokenizer.py` initializes the vocabulary, reads/pre-tokenizes corpus text, rebuilds token IDs, encodes/decodes strings, and saves/loads tokenizer JSON.
- `src/bpe_training.py` contains the slow reference trainer and the optimized frequency-weighted trainer. The fast trainer indexes affected words, incrementally updates pair counts, and selects merges from a lazy max-heap.
- `src/bpe_tokens.py` defines special IDs, the Urdu-aware regular expression, byte-level pair replacement, and the heap comparison order.

The reference and fast trainers use the same deterministic ordering: larger pair frequency first, then the greater pair of byte strings. Normalization is deliberately separate in `src/normalize.py`; tokenization itself is lossless and does not normalize input.

## Data Flow

1. `scripts/scrape_corpus.py` collects HumSub text under `data/raw/scraped/` and writes the article audit log to `logs/scrape_log.json`.
2. Published source files are placed in the matching `data/raw/` subdirectories. `scripts/clean_corpus.py` strips markup, normalizes, filters, deduplicates, splits by document, and updates the root `DATA_CARD.md`.
3. `src/train_tokenizer.py` reads only `data/processed/train.txt`, then writes `models/tokenizer.json` and a log under `logs/`.
4. `src/encode_corpus.py` reads `train.txt` and `val.txt`; its flat `uint16` arrays go to `data/encoded/`, with validation logs under `logs/`.
5. `scripts/analyze_tokenizer.py` loads `models/tokenizer.json`, compares the test split and seven selected inputs with GPT-2/o200k_base, and writes `analysis/phase3_analysis.md`. The submission PDF is `reports/report.pdf`.

## Reproducibility

Run commands from the repository root as shown in `README.md`. Training has no tokenizer-library dependency. `tiktoken` is imported by the Part C comparison only.
