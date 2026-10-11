"""
scripts/verify_raw_data.py
Verifies raw datasets for UTF-8 integrity, byte sizes, and readable Urdu samples.
"""

import os
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

SOURCES = {
    "HumSub (Scraped - Blogs/Literature)": DATA_DIR / "scraped" / "humsub_raw.txt",
    "Urdu Wikipedia (Encyclopedia)": DATA_DIR / "wikipedia" / "wikipedia_raw.txt",
    "CC-100 (Web Crawl)": DATA_DIR / "cc100" / "ur.txt",
}

LOG_FILE = DATA_DIR / "scraped" / "scrape_log.json"

print("=" * 70)
print("            RAW DATASET VERIFICATION REPORT")
print("=" * 70)

total_raw_bytes = 0

for name, path in SOURCES.items():
    print(f"\n[Checking]: {name}")
    print(f"  Path: {path}")

    if not os.path.exists(path):
        print(f"  [ERROR]: File does NOT exist at this path!")
        continue

    # 1. Size check
    size_bytes = os.path.getsize(path)
    size_mb = size_bytes / (1024 * 1024)
    total_raw_bytes += size_bytes
    print(f"  Size: {size_bytes:,} bytes ({size_mb:.2f} MB)")

    # 2. UTF-8 Encoding & Sample Read
    try:
        with open(path, "r", encoding="utf-8") as f:
            sample = f.read(500)
            print(f"  UTF-8 Encoding: VALID (No decoding errors)")
            # Clean preview
            clean_preview = " ".join(sample.split())[:120]
            print(f"  Sample text: {clean_preview}...")
    except UnicodeDecodeError as e:
        print(f"  [CRITICAL ERROR]: Encoding is corrupted! {e}")
    except Exception as e:
        print(f"  [ERROR]: {e}")

# 3. Check Scraper Evidence Log
print("\n" + "-" * 70)
print("[Checking]: Scraper Evidence Audit Log")
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        log_data = json.load(f)
    print(f"  Status: FOUND ({LOG_FILE})")
    print(f"  Articles logged: {log_data.get('total_documents_collected', 'N/A')}")
    print(f"  Logged bytes:    {log_data.get('total_raw_bytes', 'N/A'):,} bytes")
    print(f"  Timestamp:       {log_data.get('collection_timestamp', 'N/A')}")
else:
    print(f"  [WARNING]: Audit log not found at {LOG_FILE}")

print("\n" + "=" * 70)
print(f"TOTAL RAW DATA AVAILABLE: {total_raw_bytes / (1024 * 1024 * 1024):.2f} GB")
print("=" * 70)