"""
Simple fixed-window rate limiting middleware.

No external dependency -- keeps a per-client request count in memory and
resets it every window. This is intentionally simple: it's per-process state,
so it resets on restart and doesn't share counts across multiple replicas of
this service. That's fine for a single-instance deployment (the common case
for this AI layer); if you scale to multiple replicas behind a load balancer,
replace this with a Redis-backed limiter so counts are shared.

Configure via environment variables:
    RATE_LIMIT_REQUESTS        max requests per window per client (default 60)
    RATE_LIMIT_WINDOW_SECONDS  window length in seconds (default 60)

Clients are identified by IP address (X-Forwarded-For if present, since this
sits behind a reverse proxy in production -- see nginx/Caddy config -- else
the direct connection IP).
"""
from __future__ import annotations

import os
import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from utils.logger import get_logger

logger = get_logger(__name__)


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.max_requests = int(os.environ.get("RATE_LIMIT_REQUESTS", "60"))
        self.window_seconds = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))
        # client_id -> (window_start_timestamp, count)
        self._buckets: dict[str, tuple[float, int]] = defaultdict(lambda: (0.0, 0))
        logger.info(
            f"Rate limiting enabled: {self.max_requests} requests / {self.window_seconds}s per client."
        )

    async def dispatch(self, request: Request, call_next):
        # Don't rate-limit docs/openapi so Swagger UI stays usable during testing.
        if request.url.path in ("/docs", "/openapi.json", "/redoc", "/health", "/"):
            return await call_next(request)

        client_id = _get_client_ip(request)
        now = time.time()
        window_start, count = self._buckets[client_id]

        if now - window_start >= self.window_seconds:
            window_start, count = now, 0

        count += 1
        self._buckets[client_id] = (window_start, count)

        if count > self.max_requests:
            retry_after = max(0, int(self.window_seconds - (now - window_start)))
            logger.warning(f"Rate limit exceeded for {client_id} ({count}/{self.max_requests}).")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "RateLimitExceeded",
                    "message": f"Too many requests. Limit is {self.max_requests} per {self.window_seconds}s.",
                },
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)
