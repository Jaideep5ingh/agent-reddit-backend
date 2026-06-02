import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

# This is a transient, in-memory store (single process). Jobs are evicted by age
# and capped in count so a long-running server doesn't leak memory. Override via env.
JOB_TTL_SECONDS = int(os.environ.get("JOB_TTL_SECONDS", str(60 * 60)))  # 1 hour
MAX_JOBS = int(os.environ.get("MAX_JOBS", "200"))


class JobStatus:
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class Job:
    job_id: str
    status: str = JobStatus.PENDING
    # asyncio.Queue bridges background coroutine (puts events) and SSE generator (drains events)
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue())
    posts: list = field(default_factory=list)
    report: str = ""   # full assembled report, built token by token
    error: Optional[str] = None
    created_at: float = field(default_factory=time.monotonic)


_jobs: dict[str, Job] = {}


def _prune() -> None:
    """Evict expired jobs, then enforce the hard cap (oldest first)."""
    now = time.monotonic()
    for jid in [jid for jid, j in _jobs.items() if now - j.created_at > JOB_TTL_SECONDS]:
        _jobs.pop(jid, None)
    if len(_jobs) > MAX_JOBS:
        oldest = sorted(_jobs.items(), key=lambda kv: kv[1].created_at)[: len(_jobs) - MAX_JOBS]
        for jid, _ in oldest:
            _jobs.pop(jid, None)


def make_job() -> Job:
    _prune()
    job_id = str(uuid.uuid4())
    job = Job(job_id=job_id)
    _jobs[job_id] = job
    return job


def get_job(job_id: str) -> Optional[Job]:
    return _jobs.get(job_id)
