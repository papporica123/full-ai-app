from ddgs import DDGS


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Free web search, no API key required. Returns [{title, href, body}, ...]."""
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


def format_results(results: list[dict]) -> str:
    if not results:
        return "No search results found."
    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(f"[{i}] {r.get('title', '')}\n{r.get('href', '')}\n{r.get('body', '')}")
    return "\n\n".join(lines)
