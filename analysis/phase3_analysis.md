# Phase 3: Tokenizer Analysis

**Status:** Final 50,000-token tokenizer (50,000 vocabulary entries).
**Tokenizer:** `models/tokenizer.json`.

## Training Runtime

Full-corpus training used 337.1s for counting and 173.7s for merge learning (510.8s total). Recorded system: Linux-7.2.9-200.fc44.x86_64-x86_64-with-glibc2.43; Python 3.14.8; 12 CPU cores; peak memory 2332 MB. CPU model: not reported by platform.processor().

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
| 24 | 259 + 179 | 3 | yes | " س" |
| 25 | 218 + 190 | 2 | yes | "ھ" |
| 26 | 259 + 168 | 3 | yes | " ب" |
| 27 | 218 + 175 | 2 | yes | "گ" |
| 28 | 266 + 258 | 5 | yes | " کی" |
| 29 | 266 + 265 | 5 | yes | " کے" |
| 30 | 216 + 168 | 2 | yes | "ب" |
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
| 43 | 217 + 130 | 2 | yes | "ق" |
| 44 | 217 + 185 | 2 | yes | "ٹ" |
| 45 | 261 + 136 | 3 | yes | " و" |
| 46 | 216 + 185 | 2 | yes | "ع" |
| 47 | 257 + 269 | 4 | yes | "ال" |
| 48 | 275 + 265 | 5 | yes | " ہے" |
| 49 | 268 + 166 | 4 | yes | "ائ" |
| 50 | 259 + 177 | 3 | yes | " ر" |
| 5,000 | 1710 + 4736 | 15 | yes | " پریشانی" |
| 10,000 | 301 + 8576 | 9 | yes | " ورحم" |
| 15,000 | 338 + 14363 | 13 | yes | " انوکھی" |
| 20,000 | 645 + 392 | 11 | yes | " کاملہ" |
| 25,000 | 388 + 277 | 5 | yes | " ٹس" |
| 30,000 | 854 + 378 | 8 | yes | "منون" |
| 35,000 | 4606 + 688 | 11 | yes | " چکانا" |
| 40,000 | 500 + 35410 | 9 | yes | " میَں" |
| 45,000 | 13341 + 115 | 4 | yes | "atis" |
| 49,743 | 480 + 308 | 5 | yes | " غپ" |

Two-byte Urdu letters rebuilt in the first 50: 1: "ا", 2: "ی", 4: "ک", 6: "ر", 7: "و", 8: "ہ", 9: "ے", 11: "ن", 13: "ل", 14: "م", 16: "ں", 18: "ت", 21: "س", 22: "د", 25: "ھ", 27: "گ", 30: "ب", 42: "ف", 43: "ق", 44: "ٹ", 46: "ع"
Early merges invalid as standalone UTF-8 (first 50): 3, 5, 12, 39
Letter-forming merges rebuild UTF-8 Urdu characters; merges 3, 5, 12, 39 are partial byte sequences and cannot be decoded alone.
Merge 20 first forms the multi-letter corpus word "یں".

### 20 Longest Learned Tokens

| Token ID | UTF-8 bytes | Token |
|---:|---:|---|
| 26755 | 32 | "۔۔۔۔۔۔۔۔۔۔۔۔۔۔۔۔" |
| 35817 | 30 | "چندرابابونائیڈو" |
| 19038 | 29 | " انسائیکلوپیڈیا" |
| 36509 | 29 | " انٹرکانٹینینٹل" |
| 29505 | 28 | "نارتھیمپٹنشائر" |
| 10120 | 27 | " انانکارپوریٹڈ" |
| 18324 | 27 | " خیبرپختونخواہ" |
| 31708 | 27 | " پارلیمنٹیرینز" |
| 38710 | 27 | " انگیزانکشافات" |
| 42258 | 27 | " لاہورہائیکورٹ" |
| 49866 | 27 | " غیرجانبدارانہ" |
| 26680 | 26 | " -------------------------" |
| 37511 | 26 | "خیبرپختونخواہ" |
| 47785 | 26 | "لاہورہائیکورٹ" |
| 7682 | 25 | " خیبرپختونخوا" |
| 23648 | 25 | " امیرالمومنین" |
| 26325 | 25 | " امیرالمؤمنین" |
| 27489 | 25 | " اورنامورمصنف" |
| 32915 | 25 | " دارالحکومتوں" |
| 37547 | 25 | " انفرااسٹرکچر" |

## 2. Tokens per Word

Word counts use whitespace-delimited words; end-of-text markers are excluded.

| Input | Words | Urdu BPE tokens/word | GPT-2 tokens/word | o200k_base tokens/word |
|---|---:|---:|---:|---:|
| test.txt | 2,367,063 | 1.222 | 6.102 | 1.638 |
| News | 15 | 1.133 | 5.733 | 1.533 |
| Poetry | 15 | 1.133 | 6.133 | 1.533 |
| Urdu with English | 12 | 1.500 | 4.000 | 1.333 |
| Roman Urdu | 9 | 3.000 | 2.667 | 2.000 |
| Numbers and dates | 11 | 2.455 | 6.091 | 2.455 |
| Diacritics | 7 | 2.429 | 7.857 | 3.857 |
| Personal names | 8 | 1.500 | 7.750 | 2.625 |

On test.txt, Urdu BPE used 1.222 tokens per word, versus 6.102 for GPT-2 and 1.638 for o200k_base.
Roman Urdu, rare spellings, and mixed-script inputs remain useful stress cases.

## 3. Inputs Handled Poorly

The three selected inputs with the highest Urdu-BPE token/word ratio are:

| Input type | Urdu BPE tokens/word | Example | Interpretation |
|---|---:|---|---|
| Roman Urdu | 3.000 | "Aaj mausam bohat khubsurat hai, chalo Islamabad chalte hain." | Romanized spellings vary and do not share the Urdu-script merges learned by this tokenizer. |
| Numbers and dates | 2.455 | "قیمت ۱۲۵۰ روپے ہے؛ تاریخ 11-10-2026 اور وقت 9:45 بجے ہے۔" | Digits and punctuation form combinations that are not reusable word-like units. |
| Diacritics | 2.429 | "اَللّٰهُ أَكْبَرُ، اُردُو کے اعراب اور حرکات" | Combining marks add UTF-8 bytes and are relatively sparse, so words may remain split. |

Roman Urdu was hardest at 3.000 tokens per word.
The examples are diagnostic rather than a representative evaluation set.
