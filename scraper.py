"""
General purpose Reddit scraper using Playwright (real Chrome browser).
No API credentials required.
"""

import argparse
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from urllib.parse import quote_plus

import httpx
from playwright.sync_api import sync_playwright
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

SORT_OPTIONS = ["relevance", "hot", "top", "new", "comments"]
TIME_OPTIONS = ["hour", "day", "week", "month", "year", "all"]
OLLAMA_DEFAULT_MODEL = "gemma4:31b-cloud"
OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/api/chat"


def _storage_state_kwargs() -> dict:
    """If REDDIT_STORAGE_STATE points to a saved session file (from login_reddit.py),
    launch the browser logged-in — to get past the datacenter-IP block page."""
    path = os.environ.get("REDDIT_STORAGE_STATE", "").strip()
    if path and os.path.exists(path):
        return {"storage_state": path}
    return {}


# ── URL builders ─────────────────────────────────────────────────────────────

def build_search_url(query: str, sort: str, time_filter: str) -> str:
    return (
        f"https://www.reddit.com/search/"
        f"?q={quote_plus(query)}&sort={sort}&t={time_filter}"
    )


def build_subreddit_url(subreddit: str, sort: str, time_filter: str) -> str:
    sort_path = sort if sort != "relevance" else "hot"
    return f"https://www.reddit.com/r/{subreddit}/{sort_path}/?t={time_filter}"


# ── Ollama: generate query variants ──────────────────────────────────────────

def get_query_variants(query: str, model: str = OLLAMA_DEFAULT_MODEL) -> list[str]:
    prompt = (
        f'I want to search Reddit for posts about: "{query}"\n\n'
        f'Give me exactly 2 alternative search phrases that would surface similar or related '
        f'discussions on Reddit. Write them as complete natural language phrases — the kind '
        f'a person would type into a search box — not bare keywords. '
        f'Return only the 2 phrases, one per line, with no numbering, bullet points, '
        f'explanations, or any other text.'
    )

    console.print(f"[dim]Asking {model} for query variants...[/dim]")
    try:
        response = httpx.post(
            OLLAMA_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=60,
        )
        response.raise_for_status()
        text = response.json()["message"]["content"].strip()
    except httpx.ConnectError:
        console.print("[red]Cannot reach Ollama at localhost:11434 — is it running?[/red]")
        return []
    except Exception as e:
        console.print(f"[red]Ollama error:[/red] {e}")
        return []

    # Strip any accidental numbering / bullets the model adds
    lines = [
        re.sub(r'^[\d\.\-\*\)\s]+', '', line).strip()
        for line in text.splitlines()
        if line.strip()
    ]
    variants = [l for l in lines if l][:2]

    for i, v in enumerate(variants, 1):
        console.print(f"[dim]  Variant {i}:[/dim] {v}")

    return variants


# ── Playwright scraper ────────────────────────────────────────────────────────

def scrape_posts(url: str, limit: int = 50, quiet: bool = False) -> list[dict]:
    def log(msg: str) -> None:
        if not quiet:
            console.print(msg)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
            **_storage_state_kwargs(),
        )
        page = context.new_page()

        log(f"[dim]Opening:[/dim] {url}")
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        time.sleep(3)

        COUNT_JS = """
            () => {
                const seen = new Set();
                document.querySelectorAll(
                    'search-telemetry-tracker a[href*="/comments/"], shreddit-post[permalink]'
                ).forEach(el => {
                    const key = el.getAttribute('href') || el.getAttribute('permalink') || '';
                    if (key) seen.add(key);
                });
                return seen.size;
            }
        """
        prev_count = 0
        stalled = 0
        for attempt in range(20):
            current_count = page.evaluate(COUNT_JS)
            if not quiet:
                console.print(f"[dim]  Scroll {attempt + 1}: {current_count} posts...[/dim]", end="\r")
            if current_count >= limit:
                break
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(2.5)
            if current_count == prev_count:
                stalled += 1
                if stalled >= 3:
                    break
            else:
                stalled = 0
            prev_count = current_count

        if not quiet:
            console.print()  # newline after \r

        posts = page.evaluate(f"""
            () => {{
                const results = [];
                const seen = new Set();
                const limit = {limit};

                // Search results page: search-telemetry-tracker
                document.querySelectorAll('search-telemetry-tracker').forEach(el => {{
                    if (results.length >= limit) return;
                    const titleEl = el.querySelector('h1,h2,h3');
                    const title = titleEl?.innerText.trim() || '';
                    if (!title) return;
                    const linkEl = el.querySelector('a[href*="/comments/"]');
                    const href = linkEl?.getAttribute('href') || '';
                    if (!href) return;
                    const url = href.startsWith('http') ? href : 'https://reddit.com' + href;
                    if (!url.includes('/comments/') || seen.has(url)) return;
                    seen.add(url);
                    const subEl = el.querySelector('a[href*="/r/"]:not([href*="/comments/"])');
                    const subHref = subEl?.getAttribute('href') || '';
                    const subMatch = subHref.match(/\\/r\\/([^/?#]+)/);
                    const subreddit = subMatch ? 'r/' + subMatch[1] : '?';
                    const numbers = [...el.querySelectorAll('faceplate-number')].map(n => n.getAttribute('number') || n.innerText.trim());
                    results.push({{ title, subreddit, score: numbers[0] || '?', comments: numbers[1] || '?', url }});
                }});

                // Subreddit page: shreddit-post
                if (results.length === 0) {{
                    document.querySelectorAll('shreddit-post').forEach(el => {{
                        if (results.length >= limit) return;
                        const permalink = el.getAttribute('permalink') || el.getAttribute('content-href') || '';
                        if (!permalink || !permalink.includes('/comments/')) return;
                        const url = 'https://reddit.com' + permalink;
                        if (seen.has(url)) return;
                        seen.add(url);
                        const title = el.getAttribute('post-title') || el.querySelector('h1,h2,h3')?.innerText.trim() || '';
                        const subreddit = el.getAttribute('subreddit-prefixed-name') || el.getAttribute('subreddit-name') || '?';
                        results.push({{ title, subreddit, score: el.getAttribute('score') || '?', comments: el.getAttribute('comment-count') || '?', url }});
                    }});
                }}

                // Generic fallback
                if (results.length === 0) {{
                    document.querySelectorAll('a[href*="/comments/"]').forEach(el => {{
                        if (results.length >= limit) return;
                        const href = el.getAttribute('href') || '';
                        const url = href.startsWith('http') ? href : 'https://reddit.com' + href;
                        if (seen.has(url)) return;
                        seen.add(url);
                        const title = el.innerText.trim() || el.getAttribute('aria-label') || '';
                        const subMatch = href.match(/\\/r\\/([^/?#]+)/);
                        if (title) results.push({{ title, subreddit: subMatch ? 'r/' + subMatch[1] : '?', score: '?', comments: '?', url }});
                    }});
                }}

                return results;
            }}
        """)

        browser.close()

    return posts


# ── Display ───────────────────────────────────────────────────────────────────

def display_posts(posts: list[dict], title: str, show_query_col: bool = False) -> None:
    if not posts:
        console.print("[yellow]No posts found.[/yellow]")
        return

    table = Table(title=title, box=box.ROUNDED, show_lines=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=3, justify="right")
    if show_query_col:
        table.add_column("Query", style="magenta", min_width=18, max_width=28)
    table.add_column("Subreddit", style="green", min_width=16)
    table.add_column("Score", justify="right", width=7)
    table.add_column("Comments", justify="right", width=9)
    table.add_column("Title", min_width=40)
    table.add_column("Link", style="blue", min_width=30)

    for i, post in enumerate(posts, 1):
        row = [str(i)]
        if show_query_col:
            row.append(post.get("found_by", "")[:28])
        row += [
            post.get("subreddit", "?"),
            str(post.get("score", "?")),
            str(post.get("comments", "?")),
            post.get("title", "")[:80],
            post.get("url", ""),
        ]
        table.add_row(*row)

    console.print(table)
    console.print(f"\n[dim]Total: {len(posts)} posts[/dim]\n")


# ── Deep search: parallel scrape + aggregate ──────────────────────────────────

def deep_search(
    query: str,
    sort: str,
    time_filter: str,
    limit: int,
    model: str,
) -> None:
    variants = get_query_variants(query, model)
    if not variants:
        console.print("[yellow]No variants returned — running single search.[/yellow]")
        variants = []

    all_queries = [query] + variants
    console.print(
        f"\n[bold]Deep search — {len(all_queries)} queries in parallel:[/bold]"
    )
    for q in all_queries:
        console.print(f"  • {q}")
    console.print()

    def run(q: str) -> tuple[str, list[dict]]:
        url = build_search_url(q, sort, time_filter)
        posts = scrape_posts(url, limit, quiet=True)
        return q, posts

    seen_urls: set[str] = set()
    aggregated: list[dict] = []

    with ThreadPoolExecutor(max_workers=len(all_queries)) as executor:
        futures = {executor.submit(run, q): q for q in all_queries}
        completed = 0
        for future in as_completed(futures):
            q, posts = future.result()
            completed += 1
            new_posts = 0
            for post in posts:
                url = post.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    post["found_by"] = q
                    aggregated.append(post)
                    new_posts += 1
            console.print(
                f"[dim]  [{completed}/{len(all_queries)}] \"{q}\" → "
                f"{len(posts)} posts scraped, {new_posts} new[/dim]"
            )

    console.print(
        f"\n[bold green]Aggregated {len(aggregated)} unique posts "
        f"across {len(all_queries)} searches[/bold green]\n"
    )
    display_posts(aggregated, f'Deep search: "{query}"', show_query_col=True)
    return aggregated


# ── Thread fetcher ────────────────────────────────────────────────────────────

def fetch_thread_playwright(post_url: str) -> Optional[dict]:
    """Scrape a Reddit thread page (post body + comments) using a real browser."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
            **_storage_state_kwargs(),
        )
        page = context.new_page()
        try:
            page.goto(post_url, wait_until="domcontentloaded", timeout=25000)
            time.sleep(3)
        except Exception:
            browser.close()
            return None

        data = page.evaluate("""
            () => {
                // Title
                const titleEl = document.querySelector('h1');
                const title = titleEl ? titleEl.innerText.trim() : document.title;

                // Post body (shreddit new layout)
                const bodyEl = document.querySelector(
                    '[slot="text-body"], [data-testid="post-content"] [data-click-id="text"], .md'
                );
                const selftext = bodyEl ? bodyEl.innerText.trim().slice(0, 600) : '';

                // Score
                const scoreEl = document.querySelector(
                    'shreddit-post[score], [data-testid="vote-arrows"] [id*="vote-arrows"]'
                );
                const score = scoreEl
                    ? parseInt(scoreEl.getAttribute('score') || '0', 10)
                    : 0;

                // Subreddit
                const subEl = document.querySelector('a[href^="/r/"]:not([href*="/comments/"])');
                const subHref = subEl ? subEl.getAttribute('href') : '';
                const subMatch = subHref.match(/\\/r\\/([^/?#]+)/);
                const subreddit = subMatch ? 'r/' + subMatch[1] : '?';

                // Comments — shreddit-comment elements
                const comments = [];
                document.querySelectorAll('shreddit-comment').forEach(el => {
                    const body = el.querySelector(
                        '[slot="comment"], p, .md'
                    );
                    const text = body ? body.innerText.trim().slice(0, 400) : '';
                    if (!text || text === '[deleted]' || text === '[removed]') return;
                    const depth = parseInt(el.getAttribute('depth') || '0', 10);
                    const voteEl = el.querySelector('faceplate-number');
                    const voteScore = voteEl
                        ? parseInt(voteEl.getAttribute('number') || '0', 10)
                        : 0;
                    comments.push({ depth, score: voteScore, body: text });
                });

                // Sort top-level comments by score
                const topLevel = comments
                    .filter(c => c.depth === 0)
                    .sort((a, b) => b.score - a.score)
                    .slice(0, 15);

                return { title, selftext, score, subreddit, comments: topLevel };
            }
        """)
        browser.close()

    if not data or not data.get("title"):
        return None

    return {
        "title": data["title"],
        "subreddit": data["subreddit"],
        "score": data["score"],
        "selftext": data["selftext"],
        "url": post_url,
        "top_comments": [{"score": c["score"], "body": c["body"]} for c in data["comments"]],
    }


def format_thread_for_prompt(i: int, thread: dict) -> str:
    lines = [
        f"--- Thread {i}: {thread['title']} ---",
        f"Subreddit: {thread['subreddit']}  |  Score: {thread['score']}",
    ]
    if thread["selftext"]:
        lines.append(f"Post body: {thread['selftext']}")
    lines.append("Top comments:")
    for c in thread["top_comments"][:10]:
        lines.append(f"  [{c['score']}] {c['body']}")
    return "\n".join(lines)


# ── LLM report ────────────────────────────────────────────────────────────────

def generate_report(query: str, posts: list[dict], model: str, min_score: int = 5, max_threads: int = 30, instructions: str = "") -> None:
    def score_key(p: dict) -> int:
        try:
            return int(p.get("score", 0))
        except (ValueError, TypeError):
            return 0

    qualified = [p for p in posts if score_key(p) >= min_score]
    top_posts = sorted(qualified, key=score_key, reverse=True)[:max_threads]

    console.print(
        f"\n[dim]{len(posts)} total posts → "
        f"{len(qualified)} with score ≥ {min_score} → "
        f"fetching top {len(top_posts)}[/dim]"
    )

    console.print(f"\n[bold]Fetching top {len(top_posts)} threads for analysis (parallel)...[/bold]\n")

    def fetch_with_label(post: dict) -> Optional[dict]:
        return fetch_thread_playwright(post.get("url", ""))

    threads = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_with_label, post): post for post in top_posts}
        done = 0
        for future in as_completed(futures):
            done += 1
            post = futures[future]
            thread = future.result()
            if thread:
                threads.append(thread)
                console.print(f"[dim]  [{done}/{len(top_posts)}] ✓ {post.get('url', '')}[/dim]")
            else:
                console.print(f"[yellow]  [{done}/{len(top_posts)}] ✗ skipped: {post.get('url', '')}[/yellow]")

    if not threads:
        console.print("[red]No threads could be fetched.[/red]")
        return

    threads_text = "\n\n".join(format_thread_for_prompt(i, t) for i, t in enumerate(threads, 1))

    user_focus = (
        f'\n\nUser instructions — prioritise these in your report:\n{instructions.strip()}'
        if instructions.strip() else ""
    )

    prompt = (
        f'You are a research analyst. A user searched Reddit for: "{query}"\n\n'
        f'Below are the top {len(threads)} Reddit threads on this topic, including post content and top comments.\n\n'
        f'{threads_text}{user_focus}\n\n'
        f'Generate an actionable intelligence report covering:\n'
        f'1. Key themes and patterns across the threads\n'
        f'2. Specific places, names, or recommendations that appear\n'
        f'3. Overall sentiment and consensus\n'
        f'4. Any notable disagreements or caveats\n'
        f'5. A short list of actionable takeaways for someone new to this topic\n\n'
        f'Be concise and direct. Format with clear sections.'
    )

    console.print(f"\n[bold cyan]Generating report with {model}...[/bold cyan]\n")
    console.rule()

    try:
        with httpx.stream(
            "POST",
            OLLAMA_URL,
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True},
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            import json as _json
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = _json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    print(token, end="", flush=True)
    except httpx.ConnectError:
        console.print("[red]Cannot reach Ollama at localhost:11434[/red]")
        return

    console.rule()
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="General purpose Reddit scraper (Playwright / real browser)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scraper.py search "Best pubs in NYC"
  python scraper.py search "Best pubs in NYC" --deep-search
  python scraper.py search "Python tips" --sort top --time month --limit 10
  python scraper.py subreddit nyc --sort hot --limit 20
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="Search Reddit for a query")
    search_parser.add_argument("query", help='Search query e.g. "Best pubs in NYC"')
    search_parser.add_argument("--sort", choices=SORT_OPTIONS, default="relevance")
    search_parser.add_argument("--time", choices=TIME_OPTIONS, default="week", dest="time_filter")
    search_parser.add_argument("--limit", type=int, default=50, metavar="N")
    search_parser.add_argument(
        "--deep-search",
        action="store_true",
        help="Use Ollama to generate 2 query variants and run 3 searches in parallel",
    )
    search_parser.add_argument(
        "--model",
        default=OLLAMA_DEFAULT_MODEL,
        metavar="MODEL",
        help=f"Ollama model for --deep-search / --report (default: {OLLAMA_DEFAULT_MODEL})",
    )
    search_parser.add_argument(
        "--report",
        action="store_true",
        help="After scraping, fetch qualifying threads and generate an LLM intelligence report",
    )
    search_parser.add_argument(
        "--min-score",
        type=int,
        default=5,
        metavar="N",
        help="Minimum post score to include in the report (default: 5)",
    )
    search_parser.add_argument(
        "--instructions",
        type=str,
        default="",
        metavar="TEXT",
        help='Custom focus instructions injected into the report prompt e.g. "focus on budget-friendly options in Brooklyn"',
    )

    sub_parser = subparsers.add_parser("subreddit", help="Browse a specific subreddit")
    sub_parser.add_argument("name", help="Subreddit name (without r/)")
    sub_parser.add_argument("--sort", choices=SORT_OPTIONS, default="hot")
    sub_parser.add_argument("--time", choices=TIME_OPTIONS, default="week", dest="time_filter")
    sub_parser.add_argument("--limit", type=int, default=50, metavar="N")

    args = parser.parse_args()

    try:
        if args.command == "search":
            if args.deep_search:
                posts = deep_search(
                    args.query,
                    args.sort,
                    args.time_filter,
                    args.limit,
                    args.model,
                )
            else:
                console.print(
                    f'\n[bold]Searching Reddit for:[/bold] "{args.query}"  '
                    f"[dim]sort={args.sort} time={args.time_filter} limit={args.limit}[/dim]\n"
                )
                posts = scrape_posts(build_search_url(args.query, args.sort, args.time_filter), args.limit)
                display_posts(posts, f'Search: "{args.query}"')

            if args.report and posts:
                generate_report(args.query, posts, args.model, min_score=args.min_score, instructions=args.instructions)

        elif args.command == "subreddit":
            console.print(
                f"\n[bold]Browsing:[/bold] r/{args.name}  "
                f"[dim]sort={args.sort} time={args.time_filter} limit={args.limit}[/dim]\n"
            )
            posts = scrape_posts(build_subreddit_url(args.name, args.sort, args.time_filter), args.limit)
            display_posts(posts, f"r/{args.name}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()
