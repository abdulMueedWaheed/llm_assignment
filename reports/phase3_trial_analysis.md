# Phase 3: Tokenizer Analysis

**Status:** PRELIMINARY trial tokenizer (20,000 vocabulary entries).
**Tokenizer:** `BPE_urdu/outputs/tok_trial.json`. Regenerate with the final 50,000-token model before submission.

## 1. Merge Analysis

| Merge | Input IDs | Bytes | Valid UTF-8 alone | Decoded token |
|---:|---|---:|:---:|---|
| 1 | 216 + 167 | 2 | yes | "ا" |
| 2 | 219 + 140 | 2 | yes | "ی" |
| 3 | 32 + 216 | 2 | no | invalid UTF-8: b' \xd8' |
| 4 | 218 + 169 | 2 | yes | "ک" |
| 5 | 32 + 217 | 2 | no | invalid UTF-8: b' \xd9' |
| 6 | 216 + 177 | 2 | yes | "ر" |
| 7 | 217 + 136 | 2 | yes | "و" |
| 8 | 219 + 129 | 2 | yes | "ہ" |
| 9 | 219 + 146 | 2 | yes | "ے" |
| 10 | 32 + 260 | 3 | yes | " ک" |
| 11 | 217 + 134 | 2 | yes | "ن" |
| 12 | 257 + 216 | 3 | no | invalid UTF-8: b'\xd8\xa7\xd8' |
| 13 | 217 + 132 | 2 | yes | "ل" |
| 14 | 217 + 133 | 2 | yes | "م" |
| 15 | 261 + 133 | 3 | yes | " م" |
| 16 | 218 + 186 | 2 | yes | "ں" |
| 17 | 32 + 257 | 3 | yes | " ا" |
| 18 | 216 + 170 | 2 | yes | "ت" |
| 19 | 32 + 264 | 3 | yes | " ہ" |
| 20 | 258 + 272 | 4 | yes | "یں" |
| 21 | 216 + 179 | 2 | yes | "س" |
| 22 | 216 + 175 | 2 | yes | "د" |
| 23 | 13 + 10 | 2 | yes | "\r\n" |
| 24 | 218 + 190 | 2 | yes | "ھ" |
| 25 | 259 + 179 | 3 | yes | " س" |
| 26 | 259 + 168 | 3 | yes | " ب" |
| 27 | 218 + 175 | 2 | yes | "گ" |
| 28 | 266 + 258 | 5 | yes | " کی" |
| 29 | 216 + 168 | 2 | yes | "ب" |
| 30 | 266 + 265 | 5 | yes | " کے" |
| 31 | 261 + 190 | 3 | yes | " پ" |
| 32 | 257 + 267 | 4 | yes | "ان" |
| 33 | 261 + 134 | 3 | yes | " ن" |
| 34 | 263 + 262 | 4 | yes | "ور" |
| 35 | 259 + 170 | 3 | yes | " ت" |
| 36 | 219 + 148 | 2 | yes | "۔" |
| 37 | 259 + 172 | 3 | yes | " ج" |
| 38 | 257 + 262 | 4 | yes | "ار" |
| 39 | 32 + 268 | 4 | no | invalid UTF-8: b' \xd8\xa7\xd8' |
| 40 | 271 + 276 | 7 | yes | " میں" |
| 41 | 259 + 175 | 3 | yes | " د" |
| 42 | 217 + 129 | 2 | yes | "ف" |
| 43 | 217 + 185 | 2 | yes | "ٹ" |
| 44 | 217 + 130 | 2 | yes | "ق" |
| 45 | 261 + 136 | 3 | yes | " و" |
| 46 | 216 + 185 | 2 | yes | "ع" |
| 47 | 257 + 269 | 4 | yes | "ال" |
| 48 | 268 + 166 | 4 | yes | "ائ" |
| 49 | 266 + 262 | 5 | yes | " کر" |
| 50 | 259 + 177 | 3 | yes | " ر" |
| 5,000 | 496 + 372 | 9 | yes | " بھلا" |
| 10,000 | 328 + 467 | 7 | yes | " آرم" |
| 15,000 | 1463 + 300 | 6 | yes | "محق" |
| 19,743 | 485 + 315 | 6 | yes | "مرز" |

Two-byte Urdu letters rebuilt in the first 50: 1: "ا", 2: "ی", 4: "ک", 6: "ر", 7: "و", 8: "ہ", 9: "ے", 11: "ن", 13: "ل", 14: "م", 16: "ں", 18: "ت", 21: "س", 22: "د", 24: "ھ", 27: "گ", 29: "ب", 42: "ف", 43: "ٹ", 44: "ق", 46: "ع"
Early merges invalid as standalone UTF-8 (first 50): 3, 5, 12, 39
Merge 20 first forms the multi-letter corpus word "یں".

### 20 Longest Learned Tokens

| Token ID | UTF-8 bytes | Token |
|---:|---:|---|
| 19920 | 29 | " انسائیکلوپیڈیا" |
| 10426 | 27 | " انانکارپوریٹڈ" |
| 18395 | 27 | " خیبرپختونخواہ" |
| 13736 | 26 | " -------------------------" |
| 7162 | 25 | " خیبرپختونخوا" |
| 15273 | 24 | "خیبرپختونخوا" |
| 5921 | 23 | " مینوفیکچررز" |
| 8491 | 23 | " طاہرالقادری" |
| 9630 | 23 | " یونیورسٹیوں" |
| 13720 | 23 | " انفراسٹرکچر" |
| 13860 | 23 | " انکارپوریشن" |
| 14086 | 23 | " مینوفیکچرنگ" |
| 15342 | 23 | " میٹروپولیٹن" |
| 16944 | 23 | " ایڈمنسٹریشن" |
| 17409 | 23 | " مائیکروسافٹ" |
| 18247 | 23 | " ہندوستانیوں" |
| 10383 | 22 | "انکارپوریٹڈ" |
| 3191 | 21 | " دارالحکومت" |
| 4004 | 21 | " اداکارائیں" |
| 4140 | 21 | " پاکستانیوں" |

## 2. Tokens per Word

Word counts use whitespace-delimited words. `<|endoftext|>` separators are excluded from the test-corpus comparison.

| Input | Words | Urdu BPE tokens/word | GPT-2 tokens/word | o200k_base tokens/word |
|---|---:|---:|---:|---:|
| test.txt | 2,367,063 | 1.275 | 6.102 | 1.638 |
| News | 15 | 1.133 | 5.733 | 1.533 |
| Poetry | 15 | 1.267 | 6.133 | 1.533 |
| Urdu with English | 12 | 1.917 | 4.000 | 1.333 |
| Roman Urdu | 9 | 3.667 | 2.667 | 2.000 |
| Numbers and dates | 11 | 2.636 | 6.091 | 2.455 |
| Diacritics | 7 | 3.429 | 7.857 | 3.857 |
| Personal names | 8 | 1.750 | 7.750 | 2.625 |

On test.txt, this tokenizer used 1.275 tokens per word, compared with 6.102 for GPT-2 and 1.638 for o200k_base. Roman Urdu, rare spellings, and mixed-script inputs remain useful stress cases.

## 3. Inputs Handled Poorly

The following are the three hand-picked inputs with the highest Urdu-BPE token/word ratio:

| Input type | Urdu BPE tokens/word | Example | Interpretation |
|---|---:|---|---|
| Roman Urdu | 3.667 | "Aaj mausam bohat khubsurat hai, chalo Islamabad chalte hain." | Romanized spellings vary and do not share the Urdu-script merges learned by this tokenizer. |
| Diacritics | 3.429 | "اَللّٰهُ أَكْبَرُ، اُردُو کے اعراب اور حرکات" | Combining marks add UTF-8 bytes and are relatively sparse, so words may remain split. |
| Numbers and dates | 2.636 | "قیمت ۱۲۵۰ روپے ہے؛ تاریخ 11-10-2026 اور وقت 9:45 بجے ہے۔" | Digits and punctuation form combinations that are not reusable word-like units. |

These are small diagnostic examples, not a representative evaluation set. Replace them with the final tokenizer results before submitting.
