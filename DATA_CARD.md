# DATA CARD: Urdu Monolingual Pretraining Corpus

---

## 1. Table of Sources

| Source | Genre | URL / Origin | Licence / Terms | Collection Date | Raw Size | Cleaned Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **HumSub** | Literature & Blogs | `https://www.humsub.com.pk` (REST API) | Site terms; collected for course research, not a redistribution license | October 11, 2026 | 51.92 MiB | 51.86 MiB |
| **Urdu Wikipedia** | Encyclopedia | `https://dumps.wikimedia.org/urwiki/` | CC BY-SA 4.0 | October 2026 | 359.64 MiB | 351.31 MiB |
| **CC-100 (Urdu)** | Web Crawl | `https://data.statmt.org/cc-100/` | Common Crawl / dataset terms | October 2026 | 1,534.89 MiB | 1,524.15 MiB |
| **Total** | **3 Sources, 3 Genres** | — | — | — | **1,946.45 MiB** | **1,927.33 MiB** |

*Evidence for Custom Collection*: `logs/scrape_log.json` records 5,279 collected articles, URLs, byte sizes, and timestamps.

---

## 2. Corpus Totals

- **Clean Corpus Size**: 1,927.33 MiB before document delimiters
- **Total Clean Documents**: 736,216
- **Total Word Count**: 240,192,534 words
- **Unique Word Count (Vocabulary)**: 2,985,664 distinct words
- **Document-Level Splits (seed=42)**:
  - `train.txt`: 98% (1,912.47 MiB, 721,491 documents)
  - `val.txt`: 1% (19.49 MiB, 7,362 documents)
  - `test.txt`: 1% (19.26 MiB, 7,363 documents)
- **Document Delimiter**: `<|endoftext|>`

Split files include document delimiters and separator newlines, so their combined size is larger than the cleaned document text total.

---

## 3. Cleaning Pipeline Reductions

| Pipeline Step | Output Size (MiB) | Text Removed (MiB) | Description |
| :--- | :--- | :--- | :--- |
| **0. Raw Input** | 1,946.45 MiB | — | Raw combined text across all three sources |
| **1. Markup Stripping** | 1,946.33 MiB | 0.12 MiB | Stripped wiki headers (`== ... ==`), links, HTML tags |
| **2. Unicode Normalization** | 1,942.14 MiB | 4.19 MiB | Custom `src/normalize.py`: Yeh/Kaf/Heh unified, Tatweel/BOM removed, digits to ASCII 0-9 |
| **3. Urdu Language Filter** | 1,933.16 MiB | 8.97 MiB | Dropped documents with <60% Urdu character ratio while retaining natural English code-mixing |
| **4. SHA-256 Deduplication** | 1,927.33 MiB | 5.83 MiB | Exact document deduplication using cryptographic SHA-256 hashes |

---

## 4. Known Gaps and Biases

1. **Genre Representation**: The corpus balances encyclopedia prose (Wikipedia), formal literary/blog essays (HumSub), and diverse web crawl (CC-100). Colloquial chat and social media slang are underrepresented relative to formal and journalistic Urdu.
2. **Orthographic Standardization**: Prevalent Arabic keyboard artifacts (`ي`, `ك`, `ه`) have been normalized to Urdu Unicode graphemes (`ی`, `ک`, `ہ`), preserving vocabulary compactness.
3. **Diacritics & Code-Mixing**: Rare diacritics (اعراب) have been retained to allow phonetic and poetic learning. Natural English code-mixing within Urdu sentences is preserved, while English-dominated spam was removed.
