from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.rate_limit import RedisRateLimiter


class MemoryRedis:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    async def eval(self, script, numkeys, key, ttl):
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]


class FailingRedis:
    async def eval(self, *args):
        raise ConnectionError("Redis is down")


def _app(limiter: RedisRateLimiter) -> TestClient:
    app = FastAPI()
    app.middleware("http")(limiter.dispatch)

    @app.get("/resource")
    def resource():
        return {"ok": True}

    return TestClient(app)


def test_rate_limit_uses_valid_api_keys_and_returns_retry_after() -> None:
    limiter = RedisRateLimiter("redis://unused", 1, redis_client=MemoryRedis(),
                               trusted_api_keys={"tenant-a-key", "tenant-b-key"})
    client = _app(limiter)

    assert client.get("/resource", headers={"X-API-Key": "tenant-a-key"}).status_code == 200
    assert client.get("/resource", headers={"X-API-Key": "tenant-b-key"}).status_code == 200
    limited = client.get("/resource", headers={"X-API-Key": "tenant-a-key"})

    assert limited.status_code == 429
    assert "Retry-After" in limited.headers


def test_invalid_api_keys_cannot_create_independent_rate_limit_buckets() -> None:
    limiter = RedisRateLimiter("redis://unused", 1, redis_client=MemoryRedis(),
                               trusted_api_keys={"actual-key"})
    client = _app(limiter)

    assert client.get("/resource", headers={"X-API-Key": "fake-one"}).status_code == 200
    assert client.get("/resource", headers={"X-API-Key": "fake-two"}).status_code == 429


def test_redis_failure_fails_closed_by_default_and_can_be_configured_open() -> None:
    fail_closed = _app(RedisRateLimiter("redis://unused", 10, redis_client=FailingRedis()))
    fail_open = _app(RedisRateLimiter("redis://unused", 10, fail_open=True, redis_client=FailingRedis()))

    assert fail_closed.get("/resource").status_code == 503
    assert fail_open.get("/resource").status_code == 200
