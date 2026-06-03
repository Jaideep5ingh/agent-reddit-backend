"""
App-level rate limiting (slowapi) — caps how OFTEN a given client can hit the
expensive endpoints. This is a different control from the other two layers:

  Turnstile (Step 2)        "is this a real browser on our site?"   (proves humanity)
  Rate limit (this, Step 3) "how OFTEN, per client, over time?"      (caps volume)
  Concurrency cap (Step 4)  "how MANY jobs at the same instant?"     (protects the VM now)

Keyed on the REAL client IP via api.netutil.client_ip (CF-Connecting-IP behind the
tunnel), so users aren't all collapsed into the single 127.0.0.1 bucket uvicorn sees.

Storage is IN-MEMORY, which is correct for the current SINGLE uvicorn worker (one
process => one shared counter). NOTE: when Slice B adds multiple workers, in-memory
becomes per-process and the effective limit becomes N x the configured value — at that
point switch to Redis: Limiter(key_func=..., storage_uri="redis://localhost:6379").
Counters also reset on restart; acceptable since systemd restarts are rare.
"""

from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request
from starlette.responses import JSONResponse

from api.netutil import client_ip

limiter = Limiter(key_func=client_ip)


def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """429 response the frontend can render (it reads `detail`), with a Retry-After hint."""
    resp = JSONResponse(
        status_code=429,
        content={"detail": "Too many requests — please slow down and try again shortly."},
    )
    resp.headers["Retry-After"] = "60"
    return resp
