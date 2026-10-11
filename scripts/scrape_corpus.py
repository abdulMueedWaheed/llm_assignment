"""
scrape_corpus.py
Scrapes Urdu blogs and articles from HumSub via its WordPress REST API.
Saves raw text and logs metadata (URL, bytes, timestamp) as evidence for Part A.
"""

import os
import json
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime

TARGET_DIR = r"G:\My Drive\LLM-Project\data\raw\scraped"
OUTPUT_FILE = os.path.join(TARGET_DIR, "humsub_raw.txt")
LOG_FILE = os.path.join(TARGET_DIR, "scrape_log.json")
TARGET_BYTES = 52 * 1024 * 1024  # Target ~52 MB to comfortably exceed 50 MB threshold

os.makedirs(TARGET_DIR, exist_ok=True)

API_URL = "https://www.humsub.com.pk/wp-json/wp/v2/posts"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) StudentProject/1.0"
}

audit_log = []
total_bytes = os.path.getsize(OUTPUT_FILE) if os.path.exists(OUTPUT_FILE) else 0
page = 1

print(f"Target size: {TARGET_BYTES / (1024 * 1024):.1f} MB")
print(f"Current file size: {total_bytes / (1024 * 1024):.2f} MB")

with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
    while total_bytes < TARGET_BYTES:
        params = {"per_page": 100, "page": page}
        try:
            resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
            
            if resp.status_code == 400:
                print(f"Reached last page at page {page}.")
                break
                
            if resp.status_code != 200:
                print(f"Page {page} returned status {resp.status_code}. Waiting 5 seconds...")
                time.sleep(5)
                continue

            posts = resp.json()
            if not posts:
                print(f"No posts returned for page {page}.")
                break

            page_bytes = 0
            for post in posts:
                post_id = post.get("id")
                link = post.get("link")
                title_html = post.get("title", {}).get("rendered", "")
                content_html = post.get("content", {}).get("rendered", "")

                title = BeautifulSoup(title_html, "html.parser").get_text(strip=True)
                content = BeautifulSoup(content_html, "html.parser").get_text(separator="\n", strip=True)

                if content and len(content) > 50:
                    doc = f"{title}\n\n{content}\n\n<|endoftext|>\n\n"
                    doc_bytes = len(doc.encode("utf-8"))
                    out_f.write(doc)
                    out_f.flush()
                    total_bytes += doc_bytes
                    page_bytes += doc_bytes

                    audit_log.append({
                        "id": post_id,
                        "url": link,
                        "bytes": doc_bytes,
                        "timestamp": datetime.now().isoformat()
                    })

                    if total_bytes >= TARGET_BYTES:
                        break

            mb_done = total_bytes / (1024 * 1024)
            pct = (total_bytes / TARGET_BYTES) * 100
            print(f"Page {page:03d} | +{page_bytes / 1024:.1f} KB | Total: {mb_done:.2f} / 52.00 MB ({pct:.1f}%)")
            
            page += 1
            time.sleep(0.3)

        except requests.exceptions.RequestException as e:
            print(f"Network error on page {page}: {e}. Retrying...")
            time.sleep(5)
        except Exception as e:
            print(f"Error on page {page}: {e}. Retrying...")
            time.sleep(5)

# Save evidence log
with open(LOG_FILE, "w", encoding="utf-8") as log_f:
    json.dump({
        "source": "HumSub (humsub.com.pk)",
        "genre": "Literature & Blogs",
        "license": "Fair Academic Use",
        "collection_method": "WordPress REST API v2",
        "total_documents": len(audit_log),
        "total_bytes": total_bytes,
        "collected_at": datetime.now().isoformat(),
        "articles": audit_log
    }, log_f, indent=2)

print(f"\nDone. Final size: {total_bytes / (1024 * 1024):.2f} MB")
print(f"Log written to: {LOG_FILE}")
