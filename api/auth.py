"""
Cloudflare Turnstile verification — gates POST /jobs so only a real browser that
solved a challenge on our site can start a job. This protects the Ollama quota, the
logged-in Reddit session (account-ban risk), and the VM (Chromium spawns).

SECURE BY DEFAULT: verification is ENFORCED whenever the app runs. The ONLY way to
skip it is an explicit opt-out (TURNSTILE_DISABLED=1) for local dev. A missing/typo'd
secret does NOT silently disable auth — it fails the request closed. (A dropped env
var must never quietly re-open the endpoint we built this to close.)
"""

import os
from typing import Optional

import httpx
from fastapi import Header, HTTPException, Request

# Server-to-server verification endpoint (the SECRET key never leaves the backend).
SITEVERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"

TURNSTILE_SECRET = os.environ.get("TURNSTILE_SECRET_KEY", "")
# Explicit local-dev opt-out ONLY. Absence of the secret is NOT an opt-out.
TURNSTILE_DISABLED = os.environ.get("TURNSTILE_DISABLED", "").strip().lower() in {"1", "true", "yes"}

# If siteverify is unreachable we FAIL CLOSED (reject). A brief outage for users is
# better than a window where the endpoint is unprotected.
_VERIFY_TIMEOUT = 5.0


async def verify_turnstile(
    request: Request,
    cf_turnstile_response: Optional[str] = Header(default=None),
) -> None:
    """FastAPI dependency. Returns None if the challenge is valid; raises 403/503 otherwise.

    The frontend sends the widget token in the `cf-turnstile-response` header (FastAPI
    maps the snake_case param to that hyphenated header automatically).
    """
    # Explicit dev opt-out — the only bypass.
    if TURNSTILE_DISABLED:
        return

    if not TURNSTILE_SECRET:
        # Misconfiguration: fail closed and loudly, never silently open up.
        raise HTTPException(status_code=503, detail="Server auth misconfigured (no Turnstile secret).")

    if not cf_turnstile_response:
        raise HTTPException(status_code=403, detail="Missing Turnstile token.")

    # Real client IP: behind the Cloudflare tunnel, request.client.host is always
    # 127.0.0.1 (cloudflared → uvicorn on localhost). Cloudflare injects the true
    # client IP as CF-Connecting-IP. remoteip is optional for siteverify but a useful signal.
    data = {"secret": TURNSTILE_SECRET, "response": cf_turnstile_response}
    client_ip = request.headers.get("CF-Connecting-IP")
    if client_ip:
        data["remoteip"] = client_ip

    try:
        async with httpx.AsyncClient(timeout=_VERIFY_TIMEOUT) as client:
            resp = await client.post(SITEVERIFY_URL, data=data)  # form-encoded, not JSON
            result = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        # siteverify unreachable or returned non-JSON → fail closed.
        raise HTTPException(status_code=503, detail="Could not verify Turnstile challenge.") from exc

    if not result.get("success"):
        # Common error-codes: timeout-or-duplicate (replay / expired), invalid-input-response.
        codes = ", ".join(result.get("error-codes", [])) or "unknown"
        raise HTTPException(status_code=403, detail=f"Turnstile verification failed: {codes}")
