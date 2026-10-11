# Assignment 1 Requirements Review

This checklist follows the deliverables and numbered analysis prompts in the supplied CS 417 brief. The brief says “five questions” but then lists three numbered questions; the report addresses all subparts of those three listed questions.

## Part A: Corpus

| Requirement | Current evidence | Status / action |
|---|---|---|
| At least 500 MB cleaned Urdu text | Recorded card total is 1,927.33 MiB; all three processed splits exist locally | Met |
| Three sources and two genres | HumSub, Urdu Wikipedia, CC-100 across literature/blogs, encyclopedia, and web crawl | Met; retain source terms/URLs in the card |
| At least 50 MB collected directly | HumSub raw file is about 52 MiB; audit records 5,279 articles | Met; preserve `logs/scrape_log.json` as evidence |
| Cleaning, normalization, deduplication, document split | Implemented by `scripts/clean_corpus.py` and `src/normalize.py`; fixed seed 42 | Implemented; rerun only when all raw inputs are available |
| One-page `DATA_CARD.md` | Single root card contains source sizes, totals, reductions, and biases | Verify printed page count before submission; do not publish raw copyrighted sources |

## Part B: Tokenizer

| Requirement | Current evidence | Status / action |
|---|---|---|
| 256 byte IDs, special ID 256, exactly 49,743 ordered merges | `models/tokenizer.json`: 50,000 vocabulary entries, 49,743 merges | Met |
| Urdu pre-tokenization retains marks and ZWNJ | Pattern is in `src/bpe_tokens.py`; tests cover diacritics and ZWNJ | Met |
| Reference and optimized trainers | Both implementations remain in `src/bpe_training.py` | Met |
| Fast/reference equivalence | Real 1 MiB comparison: 2,000/2,000 merges matched; reference 266.0s, fast 1.0s; round trip passed | Met; result is in `logs/test_bpe.log` |
| Encode/decode and serialized model | Round-trip, save/load, and final model load tests pass | Met |
| Full-corpus runtime/machine | Preserved in `logs/train_tokenizer.log` | Report parser now includes it in Part C output |
| `train.bin`/`val.bin` as `uint16` | 278,671,208 and 2,846,148 tokens; maximum ID 49,999; first-document round trips pass | Met |

## Part C: Analysis and Report

| Requirement | Current evidence | Status / action |
|---|---|---|
| Merge 1-50, every 5,000th merge, UTF-8 validity, first whole word, 20 longest tokens | `analysis/phase3_analysis.md` | Met; explain “whole word” as a multi-letter corpus pre-token in the report |
| Test split and seven hand inputs vs GPT-2 and o200k_base | Metrics table and interpretations in the analysis | Met; tokens/word uses whitespace-delimited words, as stated in the report |
| Three weak inputs and explanations | Highest-fragmentation hand inputs are listed with explanations | Met; characterize these as selected diagnostics, not a general benchmark |
| PDF no longer than four pages | `reports/report.pdf` is three pages | Met |

## Before Submission

1. Upload `data/processed/{train,val,test}.txt` and `data/encoded/{train,val}.bin` to the shared drive; add the share URL to the submission instructions.
2. Check `DATA_CARD.md` renders on one page and verify all source terms accurately describe permission to use the collected text.
3. Rebuild `submission.zip` from the exact file list in `README.md`; exclude the large linked text/binary files and raw sources.
