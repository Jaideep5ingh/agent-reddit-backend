"""
Async pipeline orchestrator. Runs as an arq worker task after job enqueue.

Stages:
  scrape  → fetch ALL candidate threads (one shared browser, pooled tabs)
          → AI-score every thread on its full content (gemma, structured output)
          → select the top-N by score
          → map-reduce streaming report (gpt-oss)

All progress is emitted via Redis Streams to the SSE endpoint.
"""

import asyncio

from api import events
from api.jobs import JobStatus, update_job
from api.models import ScrapeRequest
from api.embedding import select_top_threads
from api.report import stream_report_to_queue
from scraper import build_search_url, fetch_threads_batch, get_query_variants, scrape_posts

# Query-variant generation is a light rephrasing task — use a fast, cheap model
# rather than the (heavy) report model. Relevance ranking is now done by a local
# embedding model (see api.embedding), not a cloud chat model. The report runs on
# req.model (e.g. gpt-oss:120b).
VARIANT_MODEL = "gemma4:31b-cloud"

# We fetch the latest MAX_FETCH candidates and rank them on full content. MAX_FETCH
# is kept under Reddit's ~100/min .json limit so the whole fetch finishes in ONE
# rate window (fast, no 429s, no cross-window pacing). FETCH_CONCURRENCY is the
# parallel .json request count (plain HTTP, no browser).
FETCH_CONCURRENCY = 12
MAX_FETCH = 90


async def _emit(job_id: str, event: str, data: dict) -> None:
    await events.emit(job_id, event, data)


async def _scrape_all(req: ScrapeRequest, job_id: str):
    """Run 1 or 3 searches (deep_search) in parallel and return deduplicated posts."""
    if req.deep_search:
        await _emit(job_id, "progress", {"step": "variants", "message": "Generating query variants via Ollama..."})
        variants = await asyncio.to_thread(get_query_variants, req.query, VARIANT_MODEL)
        all_queries = [req.query] + variants
        await _emit(job_id, "variants", {"queries": all_queries})
        await _emit(job_id, "progress", {
            "step": "scraping",
            "message": f"Running {len(all_queries)} searches in parallel...",
        })

        urls = [build_search_url(q, req.sort, req.time_filter) for q in all_queries]
        results = await asyncio.gather(*[
            asyncio.to_thread(scrape_posts, url, req.limit, True) for url in urls
        ], return_exceptions=True)

        seen = set()
        posts = []
        for batch, query in zip(results, all_queries):
            if isinstance(batch, Exception):
                continue
            for post in batch:
                url = post.get("url", "")
                if url and url not in seen:
                    seen.add(url)
                    post["found_by"] = query
                    posts.append(post)
        return posts

    else:
        url = build_search_url(req.query, req.sort, req.time_filter)
        await _emit(job_id, "progress", {"step": "scraping", "message": "Scraping Reddit..."})
        return await asyncio.to_thread(scrape_posts, url, req.limit, True)


def _comment_count(p: dict) -> int:
    """Scraped comment count, or -1 when the scrape couldn't read a number ('?')."""
    try:
        return int(p.get("comments", 0))
    except (ValueError, TypeError):
        return -1


def _has_discussion(p: dict) -> bool:
    """A candidate must have discussion to analyse. Drop posts the scrape shows
    have ZERO comments; keep posts with known-positive comments or an unreadable
    count ('?'). No upvote-based filtering — relevance is the AI scorer's job."""
    return _comment_count(p) != 0


async def run_job_task(job_id: str, req: ScrapeRequest) -> None:
    """Entry point for the arq worker. Runs scrape → fetch → score → select → report."""
    await update_job(job_id, status=JobStatus.RUNNING)
    threads_fetched = 0
    report_text = ""  # hoisted so the `done` event can carry the full report (safety net)

    try:
        # ── Stage 1: Scrape candidate posts (deduped by URL) ──────────────────
        posts = await _scrape_all(req, job_id)
        await update_job(job_id, posts_count=len(posts))
        await _emit(job_id, "posts", {"posts": posts, "count": len(posts)})
        await _emit(job_id, "progress", {
            "step": "scraped",
            "message": f"Scraped {len(posts)} unique posts.",
        })

        top = []
        if req.report and posts:
            # Candidate = a post with discussion. Drop zero-comment posts, then keep
            # the LATEST MAX_FETCH by post timestamp. `created` (ISO 8601) comes free
            # from the search results — no fetch needed — and ISO strings sort
            # chronologically as plain strings. Capping at MAX_FETCH (<100) keeps the
            # whole fetch inside one Reddit rate-limit window.
            candidates = [p for p in posts if _has_discussion(p)]
            candidates.sort(key=lambda p: p.get("created", ""), reverse=True)
            candidates = candidates[:MAX_FETCH]

            # ── Stage 2: Fetch ALL candidate content (.json REST, header-paced) ─
            # We fetch every candidate and rank on full content. The fetcher is
            # rate-limit aware: it paces itself across Reddit's ~100/min window
            # rather than dropping threads, so large batches complete (just slower).
            await _emit(job_id, "progress", {
                "step": "fetching",
                "message": f"Fetching full content of {len(candidates)} threads...",
            })

            fetched_count = 0
            total = len(candidates)

            async def on_done(url: str, ok: bool):
                nonlocal fetched_count
                fetched_count += 1
                await _emit(job_id, "thread_fetched", {
                    "url": url, "index": fetched_count, "total": total, "success": ok,
                })

            threads = await fetch_threads_batch(
                [p["url"] for p in candidates],
                concurrency=FETCH_CONCURRENCY,
                on_done=on_done,
            )

            # A thread that turned out to have no usable comments after fetch has
            # nothing to analyse — drop it (covers posts whose scraped count was '?').
            threads = [t for t in threads if t.get("top_comments")]

            # ── Stage 3: Embed + select top-N by relevance (local, no LLM) ────
            # nomic-embed-text ranks threads by cosine relevance and dedups
            # near-identical ones — a few seconds, no cloud rate limit.
            top = await select_top_threads(
                threads, req.query, req.instructions, req.max_threads, job_id,
            )
            threads_fetched = len(top)
            await _emit(job_id, "progress", {
                "step": "selected",
                "message": f"Selected top {len(top)} threads by semantic relevance.",
            })
            await _emit(job_id, "threads_selected", {
                "threads": [
                    {"url": t["url"], "title": t.get("title", ""),
                     "ai_score": t.get("ai_score", 0), "ai_reason": t.get("ai_reason", "")}
                    for t in top
                ],
            })

        # ── Stage 5: Map-reduce streaming report ─────────────────────────────
        if req.report and top:
            report_text = await stream_report_to_queue(
                query=req.query,
                threads=top,
                model=req.model,
                instructions=req.instructions,
                job_id=job_id,
            )
            await update_job(job_id, report=report_text)

        await update_job(job_id, status=JobStatus.DONE)
        # Carry the full report in `done` too: if any report_token was missed live
        # (slow client, trimmed stream), the frontend can overwrite with this copy.
        await _emit(job_id, "done", {
            "job_id": job_id,
            "threads_fetched": threads_fetched,
            "report": report_text,
        })

    except asyncio.CancelledError:
        # Job aborted (POST /jobs/{id}/abort → arq cancels this task). Mark it and
        # tell the client, then re-raise so arq records the cancellation. The finally
        # block still closes the stream and the async-with blocks release Chromium/httpx.
        await update_job(job_id, status=JobStatus.ABORTED)
        await _emit(job_id, "aborted", {"message": "Job stopped."})
        raise

    except Exception as exc:
        await update_job(job_id, status=JobStatus.ERROR, error=str(exc))
        await _emit(job_id, "error", {"message": str(exc)})

    finally:
        # Sentinel — tells the SSE reader the stream is finished
        await events.close_stream(job_id)
