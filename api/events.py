"""Redis Streams event bus. Replaces the in-process asyncio.Queue so the SSE endpoint
(web process) and the pipeline (arq worker process) can communicate across processes.
A Stream is a persisted, replayable log: a client connecting late or reconnecting can
replay from offset 0 or from its Last-Event-ID. The hash TTL does not cover this key,
so we EXPIRE it explicitly."""
import json
from typing import AsyncIterator, Tuple

from api.jobs import JOB_TTL_SECONDS
from api.redis_client import get_redis

END_EVENT = "__end__"
# Report streams one entry PER TOKEN, and a full report can be a few thousand tokens.
# The frontend rebuilds the report by concatenating report_token events (incl. on a
# reload/reconnect that replays from offset 0), so MAXLEN must comfortably exceed the
# worst-case event count or the start of the report gets trimmed → gappy output. Entries
# are tiny strings and the key is TTL'd (1h), so a high cap costs almost nothing.
_MAXLEN = 20000


def _key(job_id: str) -> str:
    return "events:" + job_id


async def emit(job_id: str, event: str, data: dict) -> None:
    """Append one event to the job's stream and refresh the stream's TTL. Batched into a
    single round trip (matters for report_token, which emits one event per token)."""
    r = get_redis()
    key = _key(job_id)
    async with r.pipeline(transaction=False) as pipe:
        pipe.xadd(key, {"event": event, "data": json.dumps(data)}, maxlen=_MAXLEN, approximate=True)
        pipe.expire(key, JOB_TTL_SECONDS)
        await pipe.execute()


async def close_stream(job_id: str) -> None:
    """Emit the end marker so readers know the stream is finished."""
    await emit(job_id, END_EVENT, {})


async def read_events(job_id: str, last_id: str = "0") -> AsyncIterator[Tuple[str, str, str]]:
    """Yield (entry_id, event_name, data_json) tuples as they arrive, starting AFTER
    last_id ("0" = from the beginning, replaying history). Blocks up to 25s waiting for
    new entries; on timeout yields a heartbeat tuple (entry_id="", event="__ping__",
    data="{}") so the caller can keep the HTTP connection alive and detect disconnects.
    Returns (stops) once the END_EVENT marker is seen."""
    r = get_redis()
    cursor = last_id if last_id else "0"
    while True:
        resp = await r.xread({_key(job_id): cursor}, block=25000, count=50)
        if not resp:
            # timeout — heartbeat, then keep waiting
            yield ("", "__ping__", "{}")
            continue
        for _stream_key, entries in resp:
            for entry_id, fields in entries:
                cursor = entry_id
                event = fields.get("event", "")
                if event == END_EVENT:
                    return
                yield (entry_id, event, fields.get("data", "{}"))
