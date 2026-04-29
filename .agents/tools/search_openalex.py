#!/usr/bin/env python3
"""Search OpenAlex for academic papers (no API key required).

Usage:
    python search_openalex.py "transformer attention" [options]

Options:
    --limit N        Number of results (default: 10, max: 50)
    --from-year YYYY Filter papers published >= YYYY
    --sort FIELD     cited_by_count or publication_date (default: cited_by_count)
    --output FILE    Write output to FILE instead of stdout
    --format         text or json (default: text)

Examples:
    python search_openalex.py "large language models" --limit 5
    python search_openalex.py "CRISPR gene editing" --from-year 2020 --output results.txt
    python search_openalex.py "climate change" --format json --output results.json
"""

import json
import sys
import urllib.parse
import urllib.request


def reconstruct_abstract(inverted_index: dict | None) -> str:
    """Reconstruct abstract text from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    words = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words[i] for i in sorted(words))


def search_openalex(query: str, limit: int = 10, from_year: int | None = None, sort: str = "cited_by_count") -> list[dict]:
    """Query the OpenAlex works API and return a list of paper dicts."""
    params = {
        "search": query,
        "per-page": min(limit, 50),
        "sort": f"{sort}:desc",
        "select": "title,authorships,publication_year,abstract_inverted_index,doi,cited_by_count,open_access,primary_location",
    }
    if from_year:
        params["filter"] = f"publication_year:>{from_year - 1}"

    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "student-tool/1.0 (mailto:example@example.com)"})

    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode())

    results = []
    for work in data.get("results", []):
        authors = [
            a["author"]["display_name"]
            for a in work.get("authorships", [])
            if a.get("author", {}).get("display_name")
        ]
        oa = work.get("open_access", {})
        results.append({
            "title": work.get("title") or "",
            "authors": authors[:5],  # cap at 5
            "year": work.get("publication_year"),
            "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
            "doi": work.get("doi") or "",
            "cited_by_count": work.get("cited_by_count") or 0,
            "open_access_url": oa.get("oa_url") or "",
        })
    return results


def format_text(results: list[dict]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        authors = ", ".join(r["authors"]) if r["authors"] else "Unknown"
        if len(r["authors"]) == 5:
            authors += " et al."
        lines.append(f"[{i}] {r['title']}")
        lines.append(f"    Authors : {authors}")
        lines.append(f"    Year    : {r['year']}")
        lines.append(f"    Cited   : {r['cited_by_count']}")
        if r["doi"]:
            lines.append(f"    DOI     : {r['doi']}")
        if r["open_access_url"]:
            lines.append(f"    PDF     : {r['open_access_url']}")
        if r["abstract"]:
            abstract = r["abstract"]
            if len(abstract) > 300:
                abstract = abstract[:297] + "..."
            lines.append(f"    Abstract: {abstract}")
        lines.append("")
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Search OpenAlex for academic papers.")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--limit", type=int, default=10, metavar="N")
    parser.add_argument("--from-year", type=int, default=None, metavar="YYYY")
    parser.add_argument("--sort", choices=["cited_by_count", "publication_date"], default="cited_by_count")
    parser.add_argument("--output", metavar="FILE", default=None)
    parser.add_argument("--format", choices=["text", "json"], default="text", dest="fmt")
    args = parser.parse_args()

    try:
        results = search_openalex(args.query, limit=args.limit, from_year=args.from_year, sort=args.sort)
    except Exception as e:
        sys.exit(f"Error fetching from OpenAlex: {e}")

    if not results:
        sys.exit("No results found.")

    if args.fmt == "json":
        output = json.dumps(results, indent=2, ensure_ascii=False)
    else:
        output = format_text(results)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Saved {len(results)} result(s) to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
