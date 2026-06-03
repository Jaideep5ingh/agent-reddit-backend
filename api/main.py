import json
import os

from dotenv import load_dotenv

# Load .env before importing modules that read env at import time (api.jobs,
# api.ollama_client, scraper) and before the CORS config below.
load_dotenv()

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api.auth import verify_turnstile
from api.jobs import JobStatus, get_job, make_job
from api.models import JobResponse, JobStatusResponse, ScrapeRequest
from api.pipeline import run_job

app = FastAPI(title="Reddit Scraper API", version="1.0.0")

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


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post(
    "/jobs",
    response_model=JobResponse,
    status_code=202,
    dependencies=[Depends(verify_turnstile)],
)
async def create_job(req: ScrapeRequest, background_tasks: BackgroundTasks):
    """
    Start a scrape job. Returns job_id immediately.
    Stream progress via GET /jobs/{job_id}/stream.
    Requires a valid Cloudflare Turnstile token (cf-turnstile-response header).
    """
    job = make_job()
    background_tasks.add_task(run_job, job, req)
    return JobResponse(job_id=job.job_id)


@app.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    """
    SSE stream for a job. Events: progress, variants, posts,
    agent_thinking, thread_fetched, report_token, done, error.
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        while True:
            item = await job.queue.get()
            if item is None:
                # Sentinel — pipeline finished
                return
            yield f"event: {item['event']}\ndata: {json.dumps(item['data'])}\n\n"

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
async def get_job_status(job_id: str):
    """Polling fallback — returns current job status and post count."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        posts_count=len(job.posts),
        report=job.report or None,
        error=job.error,
    )
