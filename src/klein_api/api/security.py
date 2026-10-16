"""Request guards applied before routing: bearer token, body size, and the busy gate."""

from __future__ import annotations

import hmac
import re
from typing import TYPE_CHECKING

from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from klein_api.api.errors import busy_response, error_response

if TYPE_CHECKING:
    from klein_api.jobs.queue import JobQueue

IDEMPOTENCY_HEADER = "Idempotency-Key"
IDEMPOTENCY_KEY_PATTERN = r"^[A-Za-z0-9._:-]{1,128}$"
_SUBMIT_PATH = re.compile(r"^/v1/workflows/([^/]+)/jobs$")
_KEY = re.compile(IDEMPOTENCY_KEY_PATTERN)
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
PROXY_HEADERS = (
    "forwarded",
    "via",
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-proto",
    "x-forwarded-port",
    "x-forwarded-server",
    "x-real-ip",
    "x-client-ip",
    "cf-connecting-ip",
    "true-client-ip",
)


def host_name(host_header: str) -> str:
    """``[::1]:8767`` -> ``::1``, ``localhost:8767`` -> ``localhost``."""
    if host_header.startswith("["):
        end = host_header.find("]")
        rest = host_header[end + 1 :] if end > 0 else ""
        # only an optional ":port" may follow the bracketed address
        if end < 0 or (rest and not (rest.startswith(":") and rest[1:].isdigit())):
            return host_header
        return host_header[1:end]
    return host_header.rsplit(":", 1)[0] if host_header.count(":") == 1 else host_header


class LoopbackOnlyMiddleware(BaseHTTPMiddleware):
    """Without a token the API trusts "this machine" — make sure that is what it gets.

    - A Host header that is not a loopback name means a browser was pointed at us through
      DNS rebinding (``attacker.example`` resolving to 127.0.0.1): refuse it.
    - Proxy headers mean a reverse proxy is republishing a token-less API: refuse it.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if host_name(request.headers.get("host", "")).lower() not in LOOPBACK_HOSTS:
            return error_response(403, "forbidden_host", "use 127.0.0.1 or localhost")
        if any(header in request.headers for header in PROXY_HEADERS):
            return error_response(
                403, "proxy_requires_token", "serving through a proxy requires KLEIN_API_TOKEN"
            )
        return await call_next(request)


class BusyGateMiddleware(BaseHTTPMiddleware):
    """Answer ``429 busy`` for a submit before its body (up to ``max_body_mb``) is read.

    The queue re-checks capacity when the job is actually created; this only saves the
    upload and JSON parsing when the answer is already known. A resubmission whose
    Idempotency-Key already has a job passes through (it gets that job back, not a 429).
    """

    def __init__(self, app: ASGIApp, queue: JobQueue) -> None:
        super().__init__(app)
        self._queue = queue

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        match = _SUBMIT_PATH.match(request.url.path)
        if request.method == "POST" and match is not None:
            key = request.headers.get(IDEMPOTENCY_HEADER)
            if key is not None and _KEY.match(key) is None:
                key = None  # the route rejects it with a 422 after this
            # the queue's lock may briefly wait on disk I/O: keep it off the event loop
            retry_after = await run_in_threadpool(self._queue.refusal, match.group(1), key)
            if retry_after is not None:
                return busy_response(retry_after)
        return await call_next(request)


class BearerAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, token: str) -> None:
        super().__init__(app)
        self._expected = token.encode()

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        scheme, _, supplied = request.headers.get("authorization", "").partition(" ")
        valid = scheme.lower() == "bearer" and hmac.compare_digest(
            supplied.strip().encode("latin-1", "replace"), self._expected
        )
        if not valid:
            return error_response(
                401,
                "unauthorized",
                "missing or invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject oversized bodies up front; any Transfer-Encoding (chunked) is refused."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        super().__init__(app)
        self._max = max_bytes

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.method in ("POST", "PUT", "PATCH"):
            length = request.headers.get("content-length")
            # Any Transfer-Encoding (even next to a Content-Length, which the HTTP parser
            # would then ignore) could stream past the limit: only a plain length is accepted.
            if "transfer-encoding" in request.headers:
                return error_response(
                    411, "length_required", "send a Content-Length without Transfer-Encoding"
                )
            # no length and no Transfer-Encoding means an empty body (rejected by the route)
            if length is not None and (not length.isdigit() or int(length) > self._max):
                return error_response(
                    413,
                    "payload_too_large",
                    f"request body over {self._max // (1024 * 1024)} MB",
                )
        return await call_next(request)
