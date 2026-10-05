"""Redis-backed fixed-window request limiting for API deployments."""
from __future__ import annotations

import hashlib
import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Request, Response
from fastapi.responses import JSONResponse


_INCREMENT_SCRIPT = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then redis.call('EXPIRE', KEYS[1], ARGV[1]) end
return count
"""


class RedisRateLimiter:
    def __init__(self, redis_url: str, requests_per_minute: int,
                 fail_open: bool = False, redis_client: Any | None = None,
                 trusted_api_keys: set[str] | None = None) -> None:
        self.redis_url = redis_url
        self.requests_per_minute = requests_per_minute
        self.fail_open = fail_open
        self.redis = redis_client
        self.trusted_api_keys = trusted_api_keys or set()

    @staticmethod
    def identity_key(api_key: str | None, client_host: str | None) -> str:
        identity = f"key:{api_key}" if api_key else f"ip:{client_host or 'unknown'}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    async def _increment(self, identity: str, now: float | None = None) -> int:
        if self.redis is None:
            from redis.asyncio import Redis
            self.redis = Redis.from_url(self.redis_url, socket_connect_timeout=2,
                                        socket_timeout=2, decode_responses=True)
        current_time = time.time() if now is None else now
        window = int(current_time // 60)
        key = f"api-rate:{window}:{identity}"
        return int(await self.redis.eval(_INCREMENT_SCRIPT, 1, key, 60))

    async def dispatch(self, request: Request,
                       call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if self.requests_per_minute <= 0 or request.url.path in {"/health", "/readyz"}:
            return await call_next(request)
        client_host = request.client.host if request.client else None
        candidate_key = request.headers.get("x-api-key")
        valid_key = candidate_key if candidate_key in self.trusted_api_keys else None
        identity = self.identity_key(valid_key, client_host)
        try:
            count = await self._increment(identity)
        except Exception:
            if self.fail_open:
                return await call_next(request)
            return JSONResponse(status_code=503, content={"detail": "Rate limiting service is unavailable"})
        if count > self.requests_per_minute:
            retry_after = max(1, 60 - int(time.time()) % 60)
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"},
                                headers={"Retry-After": str(retry_after)})
        return await call_next(request)

    async def close(self) -> None:
        if self.redis is not None and hasattr(self.redis, "aclose"):
            await self.redis.aclose()
