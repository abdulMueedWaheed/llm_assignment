# beginner-friendly Urdu BPE

This folder is a separate, easier-to-read version of the code in `BPE_urdu`.
It uses the same text input files and the same tokenizer JSON format.

## what the two training versions do

- `train_reference` keeps every word occurrence and recounts every pair
  throughout the full training text after each merge. It is the simple,
  slower version used as the correctness reference.
- `train_fast` keeps one copy of each word and uses that word's frequency when
  counting pairs. It also keeps an index from each pair to the different words
  that contain it, so a merge only updates affected words. A lazy max-heap
  selects the highest-count pair without scanning the full pair-count table.

Both versions use the same tie-breaking rule, so they should learn the same
merges from the same text.

## run from this folder

Install the project requirements first. Run commands from `BPE_URDU_EASY`:

```bash
python bpe.py
python test_bpe.py
python test_bpe.py --train ../data/processed/train.txt --ref-merges 2000
python train_tokenizer.py --max-bytes 50000000 --vocab-size 20000
python train_tokenizer.py
python encode_corpus.py
```

The one-megabyte comparison and full corpus encoding can take time. The final
training command uses all of `train.txt`; run it only after the small checks.

## files and outputs

- `bpe.py`: pre-tokenization, reference training, fast training, encode,
  decode, and tokenizer save/load.
- `test_bpe.py`: sanity, text round-trip, save/load, and trainer comparison
  tests. It can compare the trainers on a one-megabyte sample of `train.txt`.
- `train_tokenizer.py`: trains and saves `outputs/tokenizer.json`.
- `encode_corpus.py`: encodes both corpus splits by default.
- `output_utils.py`: shared readable terminal and report formatting.
- `download_data.py`: downloads prepared corpus files from the shared drive.
- `outputs/`: tokenizer JSON and text reports.
- `bin_outputs/`: `train.bin` and `val.bin` as `uint16` arrays.
