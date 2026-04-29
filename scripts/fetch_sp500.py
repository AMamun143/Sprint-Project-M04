#!/usr/bin/env python3
"""Fetch the current S&P 500 company list (name + sector) from Wikipedia
and save it to data/sp500.csv.

Usage:
    python scripts/fetch_sp500.py
"""

import csv
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# Wikipedia's S&P 500 list page (updated regularly)
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "sp500.csv"


def fetch_sp500() -> list[dict]:
    """Read the S&P 500 constituents table from Wikipedia and return
    a list of dicts with keys [name, sector]."""
    headers = {
        "User-Agent": "sp500-fetcher/1.0 (educational project; Python/requests)",
    }
    resp = requests.get(WIKI_URL, headers=headers, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # The first <table> with class 'wikitable' is the S&P 500 list
    table = soup.find("table", {"class": "wikitable"})
    if table is None:
        raise RuntimeError("Could not find S&P 500 table on Wikipedia page")

    rows = table.find_all("tr")

    # Parse header to find the column indices we need
    header = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
    name_idx = next(i for i, h in enumerate(header) if "security" in h.lower())
    sector_idx = next(
        i for i, h in enumerate(header) if "gics" in h.lower() and "sector" in h.lower()
    )

    records = []
    for row in rows[1:]:
        cells = row.find_all(["td", "th"])
        if len(cells) <= max(name_idx, sector_idx):
            continue
        name = cells[name_idx].get_text(strip=True)
        sector = cells[sector_idx].get_text(strip=True)
        # Skip footer / empty rows
        if not name or not sector:
            continue
        records.append({"name": name, "sector": sector})

    # Sort by sector then name for readability
    records.sort(key=lambda r: (r["sector"], r["name"]))
    return records


def main():
    print(f"Fetching S&P 500 list from {WIKI_URL} ...")
    records = fetch_sp500()
    print(f"Found {len(records)} companies across {len({r['sector'] for r in records})} sectors")

    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "sector"])
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved to {OUTPUT_PATH}")

    # Print a quick summary
    from collections import Counter

    counts = Counter(r["sector"] for r in records)
    print("\nCompanies per sector:")
    for sector in sorted(counts):
        print(f"  {sector}: {counts[sector]}")


if __name__ == "__main__":
    main()