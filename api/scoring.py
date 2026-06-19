"""
AI-native relevance scoring (the "AI" in front of selection).

A fast/cheap model (gemma4) reads each fetched thread's full content and scores
it 0-100 for relevance to the query + the user's custom instructions. Scoring is
batched (many threads per call) and parallelised, and uses Ollama structured
output (a Pydantic JSON schema + temperature 0) so the result is deterministic
and shape-guaranteed — the Python equivalent of a Zod-validated response.

Returns the threads annotated with `ai_score` and `ai_reason`; the pipeline then
sorts by `ai_score` and keeps the top N.
"""

import asyncio
import re

import httpx

from api import events
from api.models import ThreadScoreBatch
from api.ollama_client import ollama_chat

# Full thread bodies are large, so keep batches modest to stay within context.
# NOTE: scoring throughput is bound by the Ollama cloud rate limit, NOT by our
# concurrency. Raising MAX_PARALLEL trips the quota and sends the client into
# exponential backoff (measured: 4→6 made a 133s score take 557s). To go faster,
# pack more threads per call (bigger BATCH_SIZE) so there are FEWER calls — don't
# add parallelism. Bodies are compacted in _thread_block to keep batches in budget.
BATCH_SIZE = 14
MAX_PARALLEL = 4

_ID_RE = re.compile(r"/comments/([a-z0-9]+)", re.I)


def _extract_json(content: str) -> str:
    """Cloud models wrap JSON in ```json fences (and the `format` schema is ignored).
    Grab the outermost {...} so Pydantic can validate it."""
    i, j = content.find("{"), content.rfind("}")
    return content[i:j + 1] if i != -1 and j != -1 else content


def _post_id(url: str) -> str:
    """The stable Reddit post id from a permalink — robust matching key even if the
    model echoes the url with a different domain/casing."""
    m = _ID_RE.search(url or "")
    return m.group(1).lower() if m else (url or "").lower()


def _thread_block(i: int, t: dict) -> str:
    # Compact body (relevance rarely hinges on the full post), but keep ALL
    # comments — the discussion is the strongest relevance signal — each trimmed
    # to a snippet so bigger batches still fit.
    cs = t.get("top_comments", [])
    comments = "\n".join(f"      - {(c.get('body') or '')[:250]}" for c in cs)
    return (
        f"[{i}] url: {t['url']}\n"
        f"    title: {t.get('title', '')}\n"
        f"    subreddit: {t.get('subreddit', '?')} | upvotes: {t.get('score', '?')}\n"
        f"    body: {(t.get('selftext') or '')[:200]}\n"
        f"    comments ({len(cs)}):\n{comments or '      (none)'}"
    )


async def _score_batch(client, batch, query, instructions, model) -> dict:
    listing = "\n\n".join(_thread_block(i, t) for i, t in enumerate(batch))
    focus = (
        f"\nWeigh these user instructions heavily: {instructions.strip()}"
        if instructions.strip() else ""
    )
    prompt = (
        f'Rate how relevant and valuable each Reddit thread below is for a research '
        f'report about: "{query}".{focus}\n\n'
        f'Score each thread from 0 to 100 (100 = directly on-topic with substantive '
        f'discussion; 0 = off-topic, empty, or low-value). Judge by the title, body, '
        f'and comments — NOT just upvotes. Return one score for EVERY thread, copying '
        f'its url exactly.\n\n'
        f'{listing}\n\n'
        f'Respond ONLY with a JSON object of this exact shape (no prose):\n'
        f'{{"scores": [{{"url": "<exact url>", "relevance": <0-100>, "reason": "<one line>"}}]}}'
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "format": ThreadScoreBatch.model_json_schema(),
        "options": {"temperature": 0},
    }
    data = await ollama_chat(client, payload)
    content = (data.get("message") or {}).get("content", "") or "{}"
    try:
        parsed = ThreadScoreBatch.model_validate_json(_extract_json(content))
    except Exception:
        return {}
    return {_post_id(s.url): (s.relevance, s.reason) for s in parsed.scores}


async def score_threads(threads, query, instructions, model, job_id) -> list[dict]:
    """Annotate each thread with `ai_score` (0-100) and `ai_reason`. Threads the
    model fails to score default to 0 so they sort to the bottom."""
    if not threads:
        return threads

    batches = [threads[i:i + BATCH_SIZE] for i in range(0, len(threads), BATCH_SIZE)]
    sem = asyncio.Semaphore(MAX_PARALLEL)
    done = 0

    async with httpx.AsyncClient(timeout=120) as client:
        async def run(batch):
            nonlocal done
            async with sem:
                result = await _score_batch(client, batch, query, instructions, model)
            done += len(batch)
            await events.emit(job_id, "progress", {
                "step": "scoring",
                "message": f"Scored {done}/{len(threads)} threads...",
            })
            return result

        results = await asyncio.gather(*[run(b) for b in batches], return_exceptions=True)

    scores: dict = {}
    for r in results:
        if isinstance(r, dict):
            scores.update(r)

    for t in threads:
        sc = scores.get(_post_id(t.get("url", "")))
        t["ai_score"], t["ai_reason"] = sc if sc else (0, "unscored")

    return threads
