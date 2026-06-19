"""Redis-backed job store. Each job is a hash at key `job:{job_id}` with TTL
(JOB_TTL_SECONDS). Replaces the old in-process dict so state survives restarts and is
readable by the separate arq worker process. The live event stream is separate — see
api/events.py."""
import os
import time
import uuid
from typing import Optional

from api.redis_client import get_redis

JOB_TTL_SECONDS = int(os.environ.get("JOB_TTL_SECONDS", str(60 * 60)))  # 1 hour


class JobStatus:
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    ABORTED = "aborted"


def _key(job_id: str) -> str:
    return "job:" + job_id


async def make_job() -> str:
    """Create a new job hash (status=pending) with TTL. Returns the new job_id."""
    job_id = str(uuid.uuid4())
    r = get_redis()
    key = _key(job_id)
    await r.hset(key, mapping={
        "status": JobStatus.PENDING,
        "created_at": str(time.time()),
        "posts_count": "0",
        "report": "",
        "error": "",
    })
    await r.expire(key, JOB_TTL_SECONDS)
    return job_id


async def job_exists(job_id: str) -> bool:
    r = get_redis()
    return bool(await r.exists(_key(job_id)))


async def get_job(job_id: str) -> Optional[dict]:
    """Return the job hash as a dict, or None if it doesn't exist/expired."""
    r = get_redis()
    data = await r.hgetall(_key(job_id))
    return data or None


async def update_job(job_id: str, **fields) -> None:
    """Set one or more hash fields. Refreshes TTL so active jobs don't expire mid-run."""
    if not fields:
        return
    r = get_redis()
    key = _key(job_id)
    await r.hset(key, mapping={k: ("" if v is None else str(v)) for k, v in fields.items()})
    await r.expire(key, JOB_TTL_SECONDS)
