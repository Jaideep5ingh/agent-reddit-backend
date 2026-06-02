"""
Async pipeline orchestrator. Runs as a background task after POST /jobs.
Stages: scrape → (optional) agent thread fetch → (optional) streaming report.
All progress is emitted to job.queue as SSE events.
"""

import asyncio

from api.agent import run_thread_agent
from api.jobs import Job, JobStatus
from api.models import ScrapeRequest
from api.report import stream_report_to_queue
from scraper import build_search_url, get_query_variants, scrape_posts

# Query-variant generation is a trivial rephrasing task — use a fast, cheap,
# non-"thinking" model rather than the (possibly heavy) report model the user
# picked. gemma4 returns clean output in ~1s; gpt-oss "thinks" and often emits
# nothing for short structured prompts.
VARIANT_MODEL = "gemma4:31b-cloud"


async def _emit(job: Job, event: str, data: dict) -> None:
    await job.queue.put({"event": event, "data": data})


async def _scrape_all(req: ScrapeRequest, job: Job) -> list[dict]:
    """Run 1 or 3 searches (deep_search) in parallel and return deduplicated posts."""
    if req.deep_search:
        await _emit(job, "progress", {"step": "variants", "message": "Generating query variants via Ollama..."})
        variants = await asyncio.to_thread(get_query_variants, req.query, VARIANT_MODEL)
        all_queries = [req.query] + variants
        await _emit(job, "variants", {"queries": all_queries})
        await _emit(job, "progress", {
            "step": "scraping",
            "message": f"Running {len(all_queries)} searches in parallel...",
        })

        urls = [build_search_url(q, req.sort, req.time_filter) for q in all_queries]
        results = await asyncio.gather(*[
            asyncio.to_thread(scrape_posts, url, req.limit, True) for url in urls
        ], return_exceptions=True)

        seen: set[str] = set()
        posts: list[dict] = []
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
        await _emit(job, "progress", {"step": "scraping", "message": f"Scraping Reddit..."})
        return await asyncio.to_thread(scrape_posts, url, req.limit, True)


async def run_job(job: Job, req: ScrapeRequest) -> None:
    job.status = JobStatus.RUNNING
    threads_fetched = 0

    try:
        # ── Stage 1: Scrape posts ─────────────────────────────────────────────
        posts = await _scrape_all(req, job)
        job.posts = posts

        await _emit(job, "posts", {"posts": posts, "count": len(posts)})
        await _emit(job, "progress", {
            "step": "scraped",
            "message": f"Scraped {len(posts)} unique posts.",
        })

        # ── Stage 2: Agent thread selection + parallel fetch ──────────────────
        threads: list[dict] = []
        if req.report and posts:
            # Filter by min_score before handing to agent (reduces noise in context)
            def _score(p: dict) -> int:
                try:
                    return int(p.get("score", 0))
                except (ValueError, TypeError):
                    return 0

            qualified = [p for p in posts if _score(p) >= req.min_score]
            await _emit(job, "progress", {
                "step": "agent",
                "message": (
                    f"{len(qualified)} posts with score ≥ {req.min_score} "
                    f"— asking agent to select up to {req.max_threads} threads..."
                ),
            })

            threads = await run_thread_agent(
                query=req.query,
                posts=qualified,
                model=req.model,
                instructions=req.instructions,
                job=job,
                max_threads=req.max_threads,
            )
            threads_fetched = len(threads)

        # ── Stage 3: Streaming report ─────────────────────────────────────────
        if req.report and threads:
            await _emit(job, "progress", {
                "step": "report",
                "message": f"Generating report from {len(threads)} threads...",
            })
            await stream_report_to_queue(
                query=req.query,
                threads=threads,
                model=req.model,
                instructions=req.instructions,
                job=job,
            )

        job.status = JobStatus.DONE
        await _emit(job, "done", {"job_id": job.job_id, "threads_fetched": threads_fetched})

    except Exception as exc:
        job.status = JobStatus.ERROR
        job.error = str(exc)
        await _emit(job, "error", {"message": str(exc)})

    finally:
        # Sentinel — tells the SSE generator the stream is finished
        await job.queue.put(None)
