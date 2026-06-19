"""Shared async Redis client. Lazily created so it binds to the running event loop
(constructing at import time can bind to the wrong/absent loop). One client is reused
process-wide; redis.asyncio is safe to share across coroutines."""
import os
from typing import Optional
from redis.asyncio import Redis

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

_redis = None  # type: Optional[Redis]


def get_redis():
    global _redis
    if _redis is None:
        _redis = Redis.from_url(REDIS_URL, decode_responses=True)
    return _redis
