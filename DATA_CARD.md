# Urdu Corpus Data Card

| Source / genre | URL | Licence / terms | Collected | Raw → cleaned (MiB) |
|---|---|---|---|---:|
| HumSub, literature/blogs | humsub.com.pk | Site terms; no redistribution licence claimed | 2026-10-11 | 51.92 → 51.86 |
| Urdu Wikipedia, encyclopedia | dumps.wikimedia.org/urwiki | CC BY-SA 4.0 | Oct 2026 | 359.64 → 351.31 |
| CC-100 Urdu, web crawl | data.statmt.org/cc-100 | Common Crawl/dataset terms; verify applicable terms | Oct 2026 | 1,534.89 → 1,524.15 |

**Corpus:** 1,927.33 MiB cleaned text before delimiters; 736,216 documents; 240,192,534 words; 2,985,664 unique words. Fixed-seed (42), document-level splits: train 98% (1,912.47 MiB; 721,491 docs), validation 1% (19.49 MiB; 7,362 docs), test 1% (19.26 MiB; 7,363 docs). Each document ends with `<|endoftext|>`.

| Cleaning step | Size after step (MiB) | Removed (MiB) |
|---|---:|---:|
| Combined raw inputs | 1,946.45 | — |
| Strip wiki markup / links | 1,946.33 | 0.12 |
| Normalize Unicode, digits, whitespace | 1,942.14 | 4.19 |
| Urdu-character filter (<60%) | 1,933.16 | 8.97 |
| Exact-document SHA-256 deduplication | 1,927.33 | 5.83 |

Normalization maps Arabic yeh/kaf/heh variants to Urdu forms, removes tatweel and BOM, and maps digits to ASCII. Natural English code-mixing and diacritics are retained.

**Gaps and bias:** CC-100 contributes about 79% of cleaned bytes; social, conversational, and regional Urdu are underrepresented. HumSub provides 5,279 scraped articles (54,529,631 raw bytes; see `logs/scrape_log.json`). Only the HumSub raw file is present locally; Wikipedia and CC-100 raw inputs are absent, so the full cleaning run is not reproducible from this checkout. Verify each source's current terms before redistribution; the corpus and raw text are not included in the submission archive.