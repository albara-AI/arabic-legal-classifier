# -*- coding: utf-8 -*-
"""
AL-BAYAN# | Step 0: Fetch Data from Jordanian Official Gazette
==============================================================
Scrapes PDF documents from moj.gov.jo and saves metadata to Excel.

Usage:
    python 01_fetch_data.py
    python 01_fetch_data.py --pages 50 --output data/raw
"""

import os
import time
import argparse
import requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from tqdm import tqdm

# ── Settings ──────────────────────────────────────────────────
BASE_URL    = "https://www.moj.gov.jo/AR/Pages/Official_Gazette"
PDF_DIR     = "data/pdfs"
OUTPUT_FILE = "data/gazette_metadata.xlsx"
DELAY_SEC   = 1.5       # polite delay between requests
MAX_RETRIES = 3
HEADERS     = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)


# ── 1. Fetch page HTML ─────────────────────────────────────────
def fetch_page(url, retries=MAX_RETRIES):
    """Fetch raw HTML from a URL with retry logic."""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            r.encoding = "utf-8"
            return r.text
        except requests.RequestException as e:
            print(f"  Retry {attempt+1}/{retries}: {e}")
            time.sleep(DELAY_SEC * 2)
    return None


# ── 2. Parse gazette listing page ─────────────────────────────
def parse_gazette_list(html, base_url):
    """
    Extract gazette issue metadata from listing page.
    Returns list of dicts: {title, date, issue_number, pdf_url}
    """
    soup  = BeautifulSoup(html, "html.parser")
    items = []

    # look for links ending in .pdf or containing 'gazette'
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        text = a.get_text(strip=True)

        if not href:
            continue

        # build absolute URL
        if href.startswith("http"):
            full_url = href
        else:
            full_url = urljoin(base_url, href)

        # detect PDF or gazette document links
        is_pdf = href.lower().endswith(".pdf")
        is_doc = any(k in href.lower() for k in
                     ["gazette", "jarida", "جريدة", "official"])

        if is_pdf or is_doc:
            items.append({
                "title":        text or "Untitled",
                "pdf_url":      full_url,
                "issue_number": extract_issue_number(href),
                "year":         extract_year(href),
                "local_path":   "",
                "status":       "pending",
            })

    return items


# ── 3. Extract issue number from URL ──────────────────────────
def extract_issue_number(url_or_text):
    """Try to pull gazette issue number from URL string."""
    import re
    match = re.search(r'(\d{4,5})', url_or_text)
    return match.group(1) if match else "unknown"


# ── 4. Extract year from URL ──────────────────────────────────
def extract_year(url_or_text):
    """Extract 4-digit year from URL or text."""
    import re
    match = re.search(r'(20\d{2}|19\d{2})', url_or_text)
    return match.group(1) if match else "unknown"


# ── 5. Download a single PDF ──────────────────────────────────
def download_pdf(pdf_url, save_dir, filename=None):
    """
    Download PDF to save_dir. Returns local path or None on failure.
    Skips if file already exists (resume support).
    """
    if not filename:
        filename = Path(urlparse(pdf_url).path).name or "document.pdf"
        if not filename.endswith(".pdf"):
            filename += ".pdf"

    local_path = os.path.join(save_dir, filename)

    # skip already downloaded
    if os.path.exists(local_path):
        return local_path

    try:
        r = requests.get(pdf_url, headers=HEADERS,
                         stream=True, timeout=30)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return local_path
    except Exception as e:
        print(f"  Download failed: {pdf_url} → {e}")
        return None


# ── 6. Scrape multiple pages ──────────────────────────────────
def scrape_gazette(max_pages=10, start_page=1):
    """
    Crawl gazette listing pages and collect all document metadata.
    Supports pagination via ?page=N pattern.
    """
    all_items = []

    for page_num in range(start_page, start_page + max_pages):
        page_url = f"{BASE_URL}?page={page_num}"
        print(f"\n[Page {page_num}] {page_url}")

        html = fetch_page(page_url)
        if not html:
            print(f"  Skipping page {page_num} (fetch failed)")
            continue

        items = parse_gazette_list(html, page_url)
        print(f"  Found {len(items)} documents")

        if not items:
            print("  No more items — stopping pagination")
            break

        all_items.extend(items)
        time.sleep(DELAY_SEC)

    return all_items


# ── 7. Download all PDFs ──────────────────────────────────────
def download_all(items, pdf_dir=PDF_DIR):
    """
    Download PDFs for all items. Updates 'local_path' and 'status'.
    Shows progress bar.
    """
    for item in tqdm(items, desc="Downloading PDFs"):
        if not item.get("pdf_url"):
            item["status"] = "no_url"
            continue

        filename = f"gazette_{item['year']}_{item['issue_number']}.pdf"
        local    = download_pdf(item["pdf_url"], pdf_dir, filename)

        if local:
            item["local_path"] = local
            item["status"]     = "downloaded"
        else:
            item["status"]     = "failed"

        time.sleep(DELAY_SEC)

    return items


# ── 8. Save metadata ──────────────────────────────────────────
def save_metadata(items, output_file=OUTPUT_FILE):
    """Save scraped metadata to Excel for downstream processing."""
    df = pd.DataFrame(items)
    df.to_excel(output_file, index=False, engine="openpyxl")
    print(f"\nMetadata saved → {output_file}")
    print(f"  Total documents : {len(df)}")
    print(f"  Downloaded      : {(df.status == 'downloaded').sum()}")
    print(f"  Failed          : {(df.status == 'failed').sum()}")
    return df


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Scrape Jordanian Official Gazette")
    parser.add_argument("--pages",  type=int, default=10,
                        help="Number of listing pages to scrape")
    parser.add_argument("--start",  type=int, default=1,
                        help="Start page number")
    parser.add_argument("--output", type=str, default=PDF_DIR,
                        help="Output directory for PDFs")
    parser.add_argument("--no-download", action="store_true",
                        help="Collect metadata only, skip PDF download")
    args = parser.parse_args()

    print("=" * 60)
    print("AL-BAYAN# | Gazette Scraper")
    print("=" * 60)

    # step 1: scrape metadata
    items = scrape_gazette(max_pages=args.pages, start_page=args.start)
    print(f"\nTotal documents found: {len(items)}")

    if not items:
        print("No documents found. Check BASE_URL and site structure.")
        return

    # step 2: download PDFs
    if not args.no_download:
        items = download_all(items, pdf_dir=args.output)

    # step 3: save metadata
    save_metadata(items)


if __name__ == "__main__":
    main()
