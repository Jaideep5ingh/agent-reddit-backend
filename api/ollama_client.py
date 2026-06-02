"""
Shared Ollama HTTP helpers with retry/backoff and real error surfacing.

Ollama Cloud's free tier rate/quota-limits bursty or heavy usage and returns
429 / 5xx. The bare `raise_for_status()` we used before discarded the response
body, so the UI only ever saw "500 Internal Server Error" with no reason. These
helpers retry transient gateway errors and, on final failure, raise with the
actual Ollama error text so it propagates to the SSE `error` event.
"""

import asyncio
import json
import os

import httpx

# Ollama runs on the same host as the API (the VM). Override with OLLAMA_BASE_URL.
OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/api/chat"

# Status codes worth retrying — cloud gateway / rate-limit hiccups.
RETRY_STATUS = {429, 500, 502, 503, 504}


def _error_text(body: str) -> str:
    """Pull Ollama's `{"error": "..."}` message out of a response body."""
    try:
        parsed = json.loads(body)
        return parsed.get("error") or body[:400]
    except Exception:
        return (body or "")[:400]


async def ollama_chat(
    client: httpx.AsyncClient,
    payload: dict,
    *,
    retries: int = 3,
    base_delay: float = 2.0,
) -> dict:
    """
    Non-streaming /api/chat call with exponential backoff on transient errors.
    Returns the parsed JSON response. Raises httpx.HTTPStatusError with the real
    Ollama error message after retries are exhausted (or on a non-retryable code).
    """
    payload = {**payload, "stream": False}
    for attempt in range(retries):
        resp = await client.post(OLLAMA_URL, json=payload)
        if resp.is_success:
            return resp.json()
        if resp.status_code in RETRY_STATUS and attempt < retries - 1:
            await asyncio.sleep(base_delay * (2 ** attempt))
            continue
        raise httpx.HTTPStatusError(
            f"Ollama error {resp.status_code}: {_error_text(resp.text)}",
            request=resp.request,
            response=resp,
        )
    # Unreachable, but keeps type-checkers happy.
    raise RuntimeError("ollama_chat: exhausted retries without raising")


async def ollama_stream_tokens(
    client: httpx.AsyncClient,
    payload: dict,
    *,
    retries: int = 3,
    base_delay: float = 2.0,
):
    """
    Streaming /api/chat call yielding content token strings.

    Retries happen only *before* the first token is yielded (gateway rejections
    arrive fast, before any generation), so callers never receive a partial
    stream followed by a retry. On final failure, raises with the real error text.
    """
    payload = {**payload, "stream": True}
    for attempt in range(retries):
        async with client.stream("POST", OLLAMA_URL, json=payload) as resp:
            if not resp.is_success:
                body = (await resp.aread()).decode("utf-8", "replace")
                if resp.status_code in RETRY_STATUS and attempt < retries - 1:
                    await asyncio.sleep(base_delay * (2 ** attempt))
                    continue
                raise httpx.HTTPStatusError(
                    f"Ollama error {resp.status_code}: {_error_text(body)}",
                    request=resp.request,
                    response=resp,
                )
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
            return
