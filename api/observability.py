"""
Optional Sentry error tracking. Fully no-op unless SENTRY_DSN is set, and degrades
to a no-op if sentry-sdk isn't installed — so local dev and pre-DSN deploys are safe.

Wired in two places (separate processes, each needs its own init):
- api/main.py  → init_sentry("api")     (FastAPI integration auto-captures route errors)
- api/worker.py → init_sentry("worker")  (job errors are captured explicitly via capture_exc,
                  because run_job_task handles exceptions instead of re-raising)
"""

import os


def init_sentry(component: str) -> None:
    """Initialise Sentry for this process. No-op when SENTRY_DSN is unset/empty or the
    SDK isn't installed. Errors-only (no performance sampling) to stay in the free tier."""
    dsn = os.environ.get("SENTRY_DSN", "").strip()
    if not dsn:
        return
    try:
        import sentry_sdk
    except ImportError:
        return
    sentry_sdk.init(
        dsn=dsn,
        environment=os.environ.get("SENTRY_ENV", "production"),
        traces_sample_rate=0.0,    # errors only — keeps us within the free event budget
        send_default_pii=False,
    )
    sentry_sdk.set_tag("component", component)


def capture_exc(exc: BaseException) -> None:
    """Report an exception to Sentry if it's active; otherwise do nothing."""
    try:
        import sentry_sdk
        sentry_sdk.capture_exception(exc)
    except Exception:
        pass
