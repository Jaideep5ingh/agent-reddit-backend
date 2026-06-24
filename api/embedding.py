"""
Embedding-based candidate selection (replaces LLM relevance scoring).

A lightweight LOCAL embedding model (nomic-embed-text) turns the query + each
fetched thread into vectors; cosine similarity ranks threads by relevance. Runs
on localhost with no cloud round-trip, so there's no rate-limit/throttle — the
whole selection is a few seconds for ~200 threads.

Also does semantic dedup: near-identical threads (crossposts/reposts with
different URLs) are dropped so the report doesn't see the same content twice.

Threads are annotated with `ai_score` (0-100 relevance) and `ai_reason` so the
downstream `threads_selected` event and report stages stay unchanged.
"""

import math
import os
import re
from collections import Counter

import httpx

from api import events

OLLAMA_EMBED_URL = (
    os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/api/embed"
)
EMBED_MODEL = "nomic-embed-text"

# Two threads more similar than this are treated as the same content.
DEDUP_THRESHOLD = 0.95
# Cap embed input per thread — nomic context is ~2k tokens; title + body excerpt
# + top comments is plenty of relevance signal.
EMBED_CHAR_CAP = 1000
# Only the first N comments feed the ranking vector. Embedding on the VM's CPU is
# token-bound (~600 tok/s, no GPU), so fewer comments = faster scoring. NOTE: the
# EMBED_CHAR_CAP above is still the binding limit for comment-heavy threads — lower
# it too if scoring latency needs a bigger cut.
EMBED_TOP_COMMENTS = 10


def _thread_text(t: dict) -> str:
    """Compact, relevance-bearing text for one thread: title + body excerpt + comments."""
    parts = [t.get("title", "")]
    body = (t.get("selftext") or "")[:400]
    if body:
        parts.append(body)
    for c in t.get("top_comments", [])[:EMBED_TOP_COMMENTS]:
        b = (c.get("body") or "").strip()
        if b:
            parts.append(b)
    return "\n".join(parts)[:EMBED_CHAR_CAP]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-9)


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(s: str) -> list[str]:
    return _TOKEN_RE.findall(s.lower())


def _bm25_scores(q_tokens, doc_tokens, k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Classic BM25. Rewards exact query-term matches and, via IDF, weights rare
    terms (e.g. 'jersey', 'city') far above common ones ('best', 'bar') — exactly
    the geo/entity specificity embeddings wash out. Pure Python, no deps."""
    n = len(doc_tokens)
    if n == 0:
        return []
    avgdl = (sum(len(d) for d in doc_tokens) / n) or 1.0
    df: Counter = Counter()
    for d in doc_tokens:
        for term in set(d):
            df[term] += 1
    q_unique = set(q_tokens)
    idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in q_unique if df.get(t)}
    scores = []
    for d in doc_tokens:
        tf = Counter(d)
        dl = len(d)
        s = 0.0
        for t in q_unique:
            f = tf.get(t, 0)
            if not f or t not in idf:
                continue
            s += idf[t] * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    return scores


def _rrf_fuse(*score_lists, k: int = 60) -> list[float]:
    """Reciprocal Rank Fusion: combine several rankings by rank position, no weight
    tuning or score normalisation. An item ranked high in EITHER list scores well."""
    n = len(score_lists[0])
    fused = [0.0] * n
    for scores in score_lists:
        order = sorted(range(n), key=lambda i: scores[i], reverse=True)
        for pos, i in enumerate(order):
            fused[i] += 1.0 / (k + pos)
    return fused


async def _embed(client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
    # keep_alive pins nomic in RAM between jobs so we don't pay a cold model reload
    # on the first embed of each job (the daemon's default evicts it after 5 min idle).
    resp = await client.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBED_MODEL, "input": texts, "keep_alive": "30m"},
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


async def select_top_threads(threads, query, instructions, k, job_id) -> list[dict]:
    """Embed query + threads, dedup near-identical, return the top-k by relevance.

    Each returned thread carries `ai_score` (0-100) and `ai_reason`. Pure local
    embeddings — no cloud chat model, no rate limit.

    NOTE: `instructions` are intentionally NOT folded into the ranking query.
    Concatenating them dilutes the core query vector and injects noise (generic
    words like "find"/"rank"/"higher" drag in off-topic threads). Instructions are
    applied at the report stage instead, where the LLM handles them well. A future
    improvement could extract location/entity terms from instructions and apply a
    targeted BM25 boost without polluting the embedding query.
    """
    if not threads:
        return []

    query_text = query
    texts = [query_text] + [_thread_text(t) for t in threads]

    await events.emit(job_id, "progress", {
        "step": "scoring",
        "message": f"Embedding {len(threads)} threads & ranking by relevance...",
    })

    # Cold start loads the model (~8s); warm calls embed ~200 texts in ~1-2s.
    async with httpx.AsyncClient(timeout=120) as client:
        embeddings = await _embed(client, texts)

    qv, tvs = embeddings[0], embeddings[1:]
    embed_scores = [_cosine(qv, v) for v in tvs]

    # BM25 over the same thread texts (texts[1:]) — catches exact entities/locations
    # the embedding blurs. Fuse the two rankings with RRF (no weight tuning).
    q_tokens = _tokenize(query_text)
    bm25_scores = _bm25_scores(q_tokens, [_tokenize(x) for x in texts[1:]])
    fused = _rrf_fuse(embed_scores, bm25_scores)

    # Normalise fused score to 0-100 for a readable, well-spread relevance number.
    lo, hi = min(fused), max(fused)
    span = (hi - lo) or 1.0
    for t, v, f in zip(threads, tvs, fused):
        t["_emb"] = v
        t["ai_score"] = round(100 * (f - lo) / span)
        t["ai_reason"] = "semantic + keyword relevance"

    ranked = [t for t, _ in sorted(zip(threads, fused), key=lambda p: p[1], reverse=True)]

    # Greedy semantic dedup while filling the top-k: skip a thread if it's near
    # identical to one we've already kept.
    kept: list[dict] = []
    dropped = 0
    for t in ranked:
        if any(_cosine(t["_emb"], u["_emb"]) > DEDUP_THRESHOLD for u in kept):
            dropped += 1
            continue
        kept.append(t)
        if len(kept) >= k:
            break

    for t in threads:
        t.pop("_emb", None)

    if dropped:
        await events.emit(job_id, "progress", {
            "step": "scoring",
            "message": f"Dropped {dropped} near-duplicate threads.",
        })

    return kept


async def prerank_by_title(posts, query, k, job_id) -> list[dict]:
    """Rank candidate posts by their TITLE (already scraped — no fetch needed) and
    return the top-k. Used to pick which threads are worth fetching, so we fetch
    only ~k instead of all candidates — Reddit's .json endpoint rate-limits bursts
    (~100 requests), so fetching everything fails at scale. Titles are descriptive
    enough on Reddit to be a strong pre-filter; the full-content rerank happens
    after fetch via select_top_threads.
    """
    if len(posts) <= k:
        return posts

    texts = [query] + [p.get("title", "") for p in posts]
    await events.emit(job_id, "progress", {
        "step": "prerank",
        "message": f"Pre-ranking {len(posts)} candidates by title → fetching top {k}...",
    })

    async with httpx.AsyncClient(timeout=120) as client:
        embeddings = await _embed(client, texts)

    qv, tvs = embeddings[0], embeddings[1:]
    embed_scores = [_cosine(qv, v) for v in tvs]
    q_tokens = _tokenize(query)
    bm25_scores = _bm25_scores(q_tokens, [_tokenize(x) for x in texts[1:]])
    fused = _rrf_fuse(embed_scores, bm25_scores)

    ranked = [p for p, _ in sorted(zip(posts, fused), key=lambda x: x[1], reverse=True)]
    return ranked[:k]
