#!/usr/bin/env python3
"""Fetch recent news from RSS feeds (no API key required).

Usage:
    python fetch_news.py [options]

Options:
    --source SOURCE  News source: bbc, guardian, npr, hackernews, techcrunch, nature
                     (default: bbc). Use 'all' to fetch from all sources.
    --query KEYWORD  Keep only articles whose title/description contains KEYWORD
    --limit N        Max number of articles (default: 10)
    --output FILE    Write output to FILE instead of stdout
    --format         text or json (default: text)

Examples:
    python fetch_news.py --source bbc --limit 5
    python fetch_news.py --source hackernews --query "AI" --limit 10
    python fetch_news.py --source all --query "climate" --output news.txt
"""

import json
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime


SOURCES = {
    "bbc":        "http://feeds.bbci.co.uk/news/world/rss.xml",
    "guardian":   "https://www.theguardian.com/world/rss",
    "npr":        "https://feeds.npr.org/1001/rss.xml",
    "hackernews": "https://hnrss.org/frontpage",
    "techcrunch": "https://techcrunch.com/feed/",
    "nature":     "https://www.nature.com/nature.rss",
}


def fetch_rss(url: str) -> list[dict]:
    """Fetch and parse an RSS feed, returning a list of article dicts."""
    req = urllib.request.Request(url, headers={"User-Agent": "student-tool/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw = resp.read()

    root = ET.fromstring(raw)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    # Handle both RSS 2.0 and Atom feeds
    items = root.findall(".//item") or root.findall(".//atom:entry", ns)

    results = []
    for item in items:
        def text(tag: str) -> str:
            el = item.find(tag)
            if el is None:
                el = item.find(f"atom:{tag}", ns)
            if el is None:
                return ""
            return (el.text or "").strip()

        title = text("title")
        link = text("link") or text("url")
        # Atom feeds store link in href attribute
        if not link:
            el = item.find("atom:link", ns)
            link = (el.attrib.get("href") or "") if el is not None else ""
        description = text("description") or text("summary") or text("content")
        pub_date = text("pubDate") or text("published") or text("updated")

        if title:
            results.append({
                "title": title,
                "link": link,
                "description": _strip_html(description)[:500],
                "published": _normalize_date(pub_date),
            })
    return results


def _strip_html(text: str) -> str:
    """Remove basic HTML tags from text."""
    import re
    return re.sub(r"<[^>]+>", "", text).strip()


def _normalize_date(raw: str) -> str:
    """Try to return a clean date string; fall back to raw."""
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z",
                "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(raw.strip(), fmt).strftime("%Y-%m-%d %H:%M UTC")
        except (ValueError, AttributeError):
            pass
    return raw.strip()


def format_text(articles: list[dict]) -> str:
    lines = []
    for i, a in enumerate(articles, 1):
        lines.append(f"[{i}] {a['title']}")
        if a["published"]:
            lines.append(f"    Date : {a['published']}")
        if a["link"]:
            lines.append(f"    URL  : {a['link']}")
        if a["description"]:
            desc = a["description"]
            if len(desc) > 200:
                desc = desc[:197] + "..."
            lines.append(f"    Desc : {desc}")
        lines.append("")
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fetch recent news from RSS feeds.")
    parser.add_argument("--source", default="bbc",
                        help=f"Source name or 'all'. Available: {', '.join(SOURCES)}")
    parser.add_argument("--query", default=None, metavar="KEYWORD",
                        help="Filter articles by keyword (case-insensitive)")
    parser.add_argument("--limit", type=int, default=10, metavar="N")
    parser.add_argument("--output", metavar="FILE", default=None)
    parser.add_argument("--format", choices=["text", "json"], default="text", dest="fmt")
    args = parser.parse_args()

    # Resolve sources
    if args.source == "all":
        source_list = list(SOURCES.keys())
    elif args.source in SOURCES:
        source_list = [args.source]
    else:
        sys.exit(f"Unknown source '{args.source}'. Available: {', '.join(SOURCES)}, all")

    articles = []
    for src in source_list:
        try:
            fetched = fetch_rss(SOURCES[src])
            for a in fetched:
                a["source"] = src
            articles.extend(fetched)
        except Exception as e:
            print(f"Warning: could not fetch {src}: {e}", file=sys.stderr)

    if not articles:
        sys.exit("No articles fetched.")

    # Filter by keyword
    if args.query:
        kw = args.query.lower()
        articles = [
            a for a in articles
            if kw in a["title"].lower() or kw in a["description"].lower()
        ]
        if not articles:
            sys.exit(f"No articles matched query '{args.query}'.")

    articles = articles[: args.limit]

    if args.fmt == "json":
        output = json.dumps(articles, indent=2, ensure_ascii=False)
    else:
        output = format_text(articles)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Saved {len(articles)} article(s) to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
