# Shared Tools

Reusable CLI scripts for all skills. Run with `uv run python3 ../../tools/<script>.py` from inside a skill folder.

---

## search_openalex.py — Search academic papers

Queries the [OpenAlex](https://openalex.org) API. No API key required.

```bash
uv run python3 .agents/tools/search_openalex.py "query" [options]

Options:
  --limit N        Number of results (default: 10, max: 50)
  --from-year YYYY Only papers published >= YYYY
  --sort FIELD     cited_by_count or publication_date (default: cited_by_count)
  --output FILE    Save to file
  --format         text or json (default: text)
```

**Examples:**
```bash
uv run python3 .agents/tools/search_openalex.py "large language models" --limit 5
uv run python3 .agents/tools/search_openalex.py "CRISPR" --from-year 2022 --output results.txt
uv run python3 .agents/tools/search_openalex.py "climate change" --format json --output papers.json
```

---

## fetch_news.py — Fetch recent news via RSS

Fetches from free public RSS feeds. No API key required.

```bash
uv run python3 .agents/tools/fetch_news.py [options]

Options:
  --source SOURCE  bbc, guardian, npr, hackernews, techcrunch, nature, or all
                   (default: bbc)
  --query KEYWORD  Filter by keyword (case-insensitive)
  --limit N        Max articles (default: 10)
  --output FILE    Save to file
  --format         text or json (default: text)
```

**Examples:**
```bash
uv run python3 .agents/tools/fetch_news.py --source bbc --limit 5
uv run python3 .agents/tools/fetch_news.py --source hackernews --query "AI" --limit 10
uv run python3 .agents/tools/fetch_news.py --source all --query "climate" --output news.txt
```
