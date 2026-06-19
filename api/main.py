import os

from dotenv import load_dotenv

# Load .env before importing modules that read env at import time (api.jobs,
# api.ollama_client, scraper) and before the CORS config below.
load_dotenv()

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from slowapi.errors import RateLimitExceeded
from arq import create_pool
from arq.connections import RedisSettings

from api.auth import verify_turnstile
from api.jobs import get_job, job_exists, make_job
from api.models import JobResponse, JobStatusResponse, ScrapeRequest
from api.redis_client import REDIS_URL
from api import events
from api.ratelimit import limiter, rate_limit_handler

app = FastAPI(title="Reddit Scraper API", version="1.0.0")

# Rate limiting (slowapi). Per-IP limits live on each route via @limiter.limit(...);
# the limiter object must be on app.state and its 429 exception handler registered here.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)

# The browser calls this API cross-origin (Vercel page → VM), so CORS is required.
# Set ALLOWED_ORIGINS to a comma-separated list of allowed origins in production
# (e.g. "https://your-app.vercel.app"). Defaults to "*" for local dev.
_origins = os.environ.get("ALLOWED_ORIGINS", "*").strip()
allow_origins = ["*"] if _origins == "*" else [o.strip() for o in _origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup():
    app.state.arq = await create_pool(RedisSettings.from_dsn(REDIS_URL))


@app.on_event("shutdown")
async def _shutdown():
    pool = getattr(app.state, "arq", None)
    if pool is not None:
        await pool.aclose()


@app.get("/health")
@limiter.limit("25/minute")
async def health(request: Request):
    return {"status": "ok"}


@app.post(
    "/jobs",
    response_model=JobResponse,
    status_code=202,
    dependencies=[Depends(verify_turnstile)],
)
@limiter.limit("3/minute")
async def create_job(request: Request, req: ScrapeRequest):
    """
    Start a scrape job. Returns job_id immediately.
    Stream progress via GET /jobs/{job_id}/stream.
    Requires a valid Cloudflare Turnstile token (cf-turnstile-response header).
    """
    job_id = await make_job()
    # Emit "queued" immediately so the UI shows a queued state until the worker picks it
    # up and emits its first real progress event. If a worker slot is free the worker
    # starts in milliseconds and this flashes briefly; if all workers are busy it
    # correctly persists until one frees.
    await events.emit(job_id, "progress", {"step": "queued",
        "message": "Queued — your job will start shortly…"})
    # _job_id ties arq's job record to our job hash key (one shared identity).
    req_dict = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    await request.app.state.arq.enqueue_job("run_scrape", job_id, req_dict, _job_id=job_id)
    return JobResponse(job_id=job_id)


@app.get("/jobs/{job_id}/stream")
@limiter.limit("10/minute")
async def stream_job(request: Request, job_id: str):
    """
    SSE stream for a job. Events: progress, variants, posts,
    agent_thinking, thread_fetched, report_token, done, error.
    Supports Last-Event-ID header for reconnection/resume.
    """
    if not await job_exists(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    last_id = request.headers.get("Last-Event-ID", "0")

    async def event_generator():
        async for entry_id, event, data in events.read_events(job_id, last_id):
            if event == "__ping__":
                yield ": ping\n\n"   # SSE comment — keeps the connection alive
                continue
            yield "id: {}\nevent: {}\ndata: {}\n\n".format(entry_id, event, data)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
@limiter.limit("10/minute")
async def get_job_status(request: Request, job_id: str):
    """Polling fallback — returns current job status and post count."""
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job_id,
        status=job.get("status", ""),
        posts_count=int(job.get("posts_count", 0) or 0),
        report=job.get("report") or None,
        error=job.get("error") or None,
    )
