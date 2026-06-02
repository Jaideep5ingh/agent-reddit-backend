"""
Async Ollama streaming report generator.
Routes tokens into the job queue as report_token SSE events
instead of printing to stdout.
"""

import httpx

from api.jobs import Job
from api.ollama_client import ollama_stream_tokens
from scraper import format_thread_for_prompt


def _build_prompt(query: str, threads: list[dict], instructions: str) -> str:
    threads_text = "\n\n".join(
        format_thread_for_prompt(i, t) for i, t in enumerate(threads, 1)
    )
    user_focus = (
        f"\n\nUser instructions — prioritise these in your report:\n{instructions.strip()}"
        if instructions.strip()
        else ""
    )
    return (
        f'You are a research analyst. A user searched Reddit for: "{query}"\n\n'
        f"Below are the top {len(threads)} Reddit threads on this topic, "
        f"including post content and top comments.\n\n"
        f"{threads_text}{user_focus}\n\n"
        f"Generate an actionable intelligence report covering:\n"
        f"1. Key themes and patterns across the threads\n"
        f"2. Specific places, names, or recommendations that appear\n"
        f"3. Overall sentiment and consensus\n"
        f"4. Any notable disagreements or caveats\n"
        f"5. A short list of actionable takeaways for someone new to this topic\n\n"
        f"Be concise and direct. Format with clear sections."
    )


async def stream_report_to_queue(
    query: str,
    threads: list[dict],
    model: str,
    instructions: str,
    job: Job,
) -> None:
    """Stream Ollama report tokens into job.queue as report_token events."""
    prompt = _build_prompt(query, threads, instructions)

    async with httpx.AsyncClient(timeout=180) as client:
        token_stream = ollama_stream_tokens(
            client,
            {"model": model, "messages": [{"role": "user", "content": prompt}]},
        )
        async for token in token_stream:
            job.report += token
            await job.queue.put({
                "event": "report_token",
                "data": {"token": token},
            })
