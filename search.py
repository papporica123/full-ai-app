from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from ddgs import DDGS

_executor = ThreadPoolExecutor(max_workers=4)


def _text_search(query: str, max_results: int) -> list[dict]:
    with DDGS(timeout=5) as ddgs:
        return list(ddgs.text(query, max_results=max_results))


def web_search(query: str, max_results: int = 5, timeout: float = 10) -> list[dict]:
    """Free web search, no API key required. Returns [{title, href, body}, ...].

    Enforces a hard wall-clock timeout so a stalled connection can't hang
    the caller indefinitely (the ddgs library's own timeout only bounds
    individual socket operations, not the overall call).
    """
    future = _executor.submit(_text_search, query, max_results)
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError:
        raise TimeoutError(f"Web search timed out after {timeout}s")


def format_results(results: list[dict]) -> str:
    if not results:
        return "No search results found."
    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(f"[{i}] {r.get('title', '')}\n{r.get('href', '')}\n{r.get('body', '')}")
    return "\n\n".join(lines)
