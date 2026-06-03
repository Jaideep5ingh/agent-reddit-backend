"""
Network helpers. The one thing here matters more than its size: deriving the REAL
client IP when we run behind the Cloudflare tunnel.

uvicorn listens on 127.0.0.1 and cloudflared connects to it over localhost, so
request.client.host is 127.0.0.1 for EVERY request. Anything that keys on the client
(auth signals, rate limiting) must instead read CF-Connecting-IP, the true client IP
Cloudflare injects. Centralized here so auth and rate limiting can't drift apart.
"""

from fastapi import Request


def client_ip(request: Request) -> str:
    """Best-effort real client IP: CF-Connecting-IP (behind the tunnel), else the socket peer."""
    cf_ip = request.headers.get("CF-Connecting-IP")
    if cf_ip:
        return cf_ip.strip()
    return request.client.host if request.client else "unknown"
