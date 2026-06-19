"""
Map-reduce report generator.

For large thread counts, all threads no longer fit in a single prompt. So:
  MAP    — split threads into chunks, summarise each chunk in parallel (one LLM
           call per chunk) into a dense set of notes.
  REDUCE — feed all chunk-summaries into one final call that collates them into
           the report, streamed token-by-token to the UI as report_token events.

For small counts (a single chunk) this degrades gracefully to one map call + the
reduce pass over its notes.
"""

import asyncio

import httpx

from api import events
from api.ollama_client import ollama_chat, ollama_stream_tokens
from scraper import format_thread_for_prompt

# Threads per map chunk — small enough that a chunk + its summary fit comfortably.
CHUNK_SIZE = 10


def _map_prompt(query: str, threads, instructions: str) -> str:
    threads_text = "\n\n".join(
        format_thread_for_prompt(i, t) for i, t in enumerate(threads, 1)
    )
    focus = (
        f"\n\nThe user especially cares about: {instructions.strip()}"
        if instructions.strip() else ""
    )
    return (
        f'You are analysing a batch of Reddit threads for a report about: "{query}".{focus}\n\n'
        f'{threads_text}\n\n'
        f'Extract dense, factual notes from THIS batch only:\n'
        f'- recurring themes and patterns\n'
        f'- specific names, places, products, or recommendations mentioned\n'
        f'- notable opinions, agreements, and disagreements (with rough sentiment)\n'
        f'Be concise and bulleted. Do not write an intro or conclusion — just the notes.'
    )


def _reduce_prompt(query: str, notes: list[str], instructions: str) -> str:
    notes_text = "\n\n".join(f"--- Notes batch {i} ---\n{n}" for i, n in enumerate(notes, 1))
    focus = (
        f"\n\nUser instructions — prioritise these in your report:\n{instructions.strip()}"
        if instructions.strip() else ""
    )
    return (
        f'You are a research analyst. A user searched Reddit for: "{query}"\n\n'
        f'Below are notes extracted from several batches of Reddit threads on this '
        f'topic. Synthesise them into ONE coherent intelligence report — merge '
        f'overlapping points, resolve the overall picture across all batches.\n\n'
        f'{notes_text}{focus}\n\n'
        f'Generate an actionable report covering:\n'
        f'1. Key themes and patterns\n'
        f'2. Specific places, names, or recommendations that appear\n'
        f'3. Overall sentiment and consensus\n'
        f'4. Notable disagreements or caveats\n'
        f'5. A short list of actionable takeaways for someone new to this topic\n\n'
        f'Be concise and direct. Format with clear sections.'
    )


async def stream_report_to_queue(
    query: str,
    threads,
    model: str,
    instructions: str,
    job_id: str,
) -> str:
    """Map-reduce the threads into a streamed report. Returns the full report text."""
    chunks = [threads[i:i + CHUNK_SIZE] for i in range(0, len(threads), CHUNK_SIZE)]

    async with httpx.AsyncClient(timeout=180) as client:
        # ── MAP: summarise each chunk in parallel ────────────────────────────
        await events.emit(job_id, "progress", {
            "step": "report_map",
            "message": f"Summarising {len(threads)} threads in {len(chunks)} parallel batches...",
        })

        async def summarise(chunk):
            data = await ollama_chat(client, {
                "model": model,
                "messages": [{"role": "user", "content": _map_prompt(query, chunk, instructions)}],
            })
            return (data.get("message") or {}).get("content", "") or ""

        notes = await asyncio.gather(*[summarise(c) for c in chunks])
        notes = [n for n in notes if n.strip()]

        if not notes:
            return ""

        # ── REDUCE: collate notes into the final streamed report ─────────────
        await events.emit(job_id, "progress", {
            "step": "report_reduce",
            "message": "Collating batch summaries into the final report...",
        })

        report_text = ""
        token_stream = ollama_stream_tokens(client, {
            "model": model,
            "messages": [{"role": "user", "content": _reduce_prompt(query, notes, instructions)}],
        })
        async for token in token_stream:
            report_text += token
            await events.emit(job_id, "report_token", {"token": token})

    return report_text
