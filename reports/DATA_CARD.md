# DATA CARD: Urdu Monolingual Pretraining Corpus

---

## 1. Table of Sources

| Source | Genre | URL / Origin | Licence | Collection Date | Raw Size | Cleaned Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **HumSub** | Literature & Blogs | `https://www.humsub.com.pk` (REST API) | Fair Academic Use | October 2026 | 51.92 MB | 51.86 MB |
| **Urdu Wikipedia** | Encyclopedia | Wikimedia Dump (`urwiki`) | CC BY-SA 3.0 / 4.0 | October 2026 | 359.64 MB | 351.31 MB |
| **CC-100 (Urdu)** | Web Crawl | Common Crawl (`cc100/ur.txt.xz`) | Common Crawl Terms | October 2026 | 1534.89 MB | 1524.15 MB |
| **Total** | **3 Sources, 3 Genres** | — | — | — | **1946.45 MB** | **1927.33 MB** |

*Evidence for Custom Collection*: `data/raw/scraped/scrape_log.json` logs 5,264 articles with article IDs, URLs, byte sizes, and timestamps.

---

## 2. Corpus Totals

- **Clean Corpus Size**: 1927.33 MB
- **Total Clean Documents**: 736,216
- **Total Word Count**: 240,192,534 words
- **Unique Word Count (Vocabulary)**: 2,985,664 distinct words
- **Document-Level Splits (seed=42)**:
  - `train.txt`: 98% (1912.47 MB, 721,491 documents)
  - `val.txt`: 1% (19.06 MB, 7,362 documents)
  - `test.txt`: 1% (19.26 MB, 7,363 documents)
- **Document Delimiter**: `<|endoftext|>`

---

## 3. Cleaning Pipeline Reductions

| Pipeline Step | Output Size (MB) | Text Removed (MB) | Description |
| :--- | :--- | :--- | :--- |
| **0. Raw Input** | 1946.45 MB | — | Raw combined text across all three sources |
| **1. Markup Stripping** | 1946.33 MB | 0.12 MB | Stripped wiki headers (`== ... ==`), links, HTML tags |
| **2. Unicode Normalization** | 1942.14 MB | 4.19 MB | Custom `normalize.py`: Yeh/Kaf/Heh unified, Tatweel/BOM removed, digits to ASCII 0-9 |
| **3. Urdu Language Filter** | 1933.16 MB | 8.97 MB | Dropped documents with <60% Urdu character ratio while retaining natural English code-mixing |
| **4. SHA-256 Deduplication** | 1927.33 MB | 5.83 MB | Exact document deduplication using cryptographic SHA-256 hashes |

---

## 4. Known Gaps and Biases

1. **Genre Representation**: The corpus balances encyclopedia prose (Wikipedia), formal literary/blog essays (HumSub), and diverse web crawl (CC-100). Colloquial chat and social media slang are underrepresented relative to formal and journalistic Urdu.
2. **Orthographic Standardization**: Prevalent Arabic keyboard artifacts (`ي`, `ك`, `ه`) have been normalized to Urdu Unicode graphemes (`ی`, `ک`, `ہ`), preserving vocabulary compactness.
3. **Diacritics & Code-Mixing**: Rare diacritics (اعراب) have been retained to allow phonetic and poetic learning. Natural English code-mixing within Urdu sentences is preserved, while English-dominated spam was removed.
