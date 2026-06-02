"""
Ollama-powered agent that uses tool use to select and concurrently fetch
the most relevant Reddit threads from a scraped post list.
"""

import asyncio
import json
from typing import Optional

import httpx

from api.jobs import Job
from api.ollama_client import ollama_chat

FETCH_THREAD_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_thread",
        "description": (
            "Fetch the full content of a Reddit thread including post body and top comments. "
            "Call this for posts that look most relevant and substantive for the query. "
            "You may call it multiple times in a single response to fetch threads in parallel."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full Reddit post URL"},
            },
            "required": ["url"],
        },
    },
}


async def run_thread_agent(
    query: str,
    posts: list[dict],
    model: str,
    instructions: str,
    job: Job,
    max_threads: int = 15,
) -> list[dict]:
    """
    Shows Ollama the full post list and lets it call fetch_thread() on the
    most relevant ones. All tool calls within a single response are executed
    concurrently via asyncio.gather. Returns list of fetched thread dicts.
    """
    # Import here to avoid circular import at module level
    from scraper import fetch_thread_playwright

    post_list = "\n".join(
        f"{i + 1}. [{p.get('score', '?')} pts] {p.get('title', '')} "
        f"— {p.get('subreddit', '?')} — {p.get('url', '')}"
        for i, p in enumerate(posts)
    )

    focus = f"\nUser focus: {instructions.strip()}" if instructions.strip() else ""
    system_msg = (
        f"You are a research assistant selecting Reddit threads to fetch for a report about: \"{query}\".{focus}\n\n"
        f"Use the fetch_thread tool to retrieve the most relevant, substantive threads. "
        f"Prefer threads with real discussion (comments) over link-only posts. "
        f"Fetch at most {max_threads} threads total. "
        f"When you have fetched enough, stop calling the tool and say DONE."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": f"Here are the scraped posts:\n\n{post_list}\n\nFetch the most relevant threads now.",
        },
    ]

    fetched: dict[str, dict] = {}  # url → thread data, prevents double-fetching

    async def execute_tool_call(tc: dict) -> dict:
        """Fetch one thread and emit a thread_fetched SSE event."""
        args = tc.get("function", {}).get("arguments", {})
        # Ollama may return arguments as a string or dict
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        url = args.get("url", "")
        call_id = tc.get("id", "")

        thread = None
        if url and url not in fetched:
            thread = await asyncio.to_thread(fetch_thread_playwright, url)
            if thread:
                fetched[url] = thread

        await job.queue.put({
            "event": "thread_fetched",
            "data": {
                "url": url,
                "index": len(fetched),
                "total": max_threads,
                "success": thread is not None,
            },
        })

        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(
                {"status": "ok", "title": thread["title"], "comments": len(thread["top_comments"])}
                if thread
                else {"status": "error", "message": "Could not fetch thread"}
            ),
        }

    await job.queue.put({
        "event": "agent_thinking",
        "data": {"message": f"Agent reviewing {len(posts)} posts to select the most relevant threads..."},
    })

    async with httpx.AsyncClient(timeout=90) as client:
        for _ in range(10):  # max agent turns
            if len(fetched) >= max_threads:
                break

            data = await ollama_chat(
                client,
                {
                    "model": model,
                    "messages": messages,
                    "tools": [FETCH_THREAD_TOOL],
                },
            )
            msg = data.get("message", {})
            tool_calls = msg.get("tool_calls") or []

            # Capture any text the model emitted alongside tool calls
            text_content = msg.get("content", "")
            if text_content and text_content.strip():
                await job.queue.put({
                    "event": "agent_thinking",
                    "data": {"message": text_content.strip()[:300]},
                })

            if not tool_calls:
                break

            # Execute all tool calls from this turn in parallel
            tool_results = await asyncio.gather(*[execute_tool_call(tc) for tc in tool_calls])

            # Append assistant turn and tool results for next loop
            messages.append({"role": "assistant", "content": text_content, "tool_calls": tool_calls})
            messages.extend(tool_results)

            if data.get("done_reason") == "stop" and not tool_calls:
                break

    await job.queue.put({
        "event": "progress",
        "data": {"step": "agent_done", "message": f"Agent fetched {len(fetched)} relevant threads"},
    })

    return list(fetched.values())
