"""arq worker entrypoint. Run with:  arq api.worker.WorkerSettings
Pulls jobs enqueued by POST /jobs and runs the scrape→agent→report pipeline in this
separate process, writing state to Redis and events to the Redis Stream. max_jobs caps
how many jobs run concurrently in one worker (replaces the old in-process Semaphore)."""
import os

from dotenv import load_dotenv
load_dotenv()

from arq.connections import RedisSettings

from api.models import ScrapeRequest
from api.pipeline import run_job_task
from api.redis_client import REDIS_URL

MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "3"))


async def run_scrape(ctx, job_id: str, req_dict: dict) -> None:
    req = ScrapeRequest(**req_dict)
    await run_job_task(job_id, req)


class WorkerSettings:
    functions = [run_scrape]
    redis_settings = RedisSettings.from_dsn(REDIS_URL)
    max_jobs = MAX_CONCURRENT_JOBS
    job_timeout = 600  # seconds; a stuck scrape/LLM call fails instead of hanging forever
    keep_result = 0    # we store results in our own job hash; don't keep arq's result blob
