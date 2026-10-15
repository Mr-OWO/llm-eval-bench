import asyncio
import time
import pytest
from llm_eval_bench.pool import AsyncWorkerPool, TokenBucketRateLimiter


@pytest.mark.asyncio
async def test_token_bucket_rate_limiter():
    # Rate of 20 tokens per second, capacity of 5
    limiter = TokenBucketRateLimiter(rate=20.0, capacity=5.0)

    # Acquire initial burst
    start = time.monotonic()
    for _ in range(5):
        await limiter.acquire(1.0)
    elapsed = time.monotonic() - start
    assert elapsed < 0.2  # initial burst should be immediate

    # Next acquire should throttle slightly
    await limiter.acquire(1.0)
    elapsed_after = time.monotonic() - start
    assert elapsed_after >= 0.04


@pytest.mark.asyncio
async def test_async_worker_pool_concurrency():
    pool = AsyncWorkerPool(concurrency=5, requests_per_sec=100.0)
    active_count = 0
    max_active = 0
    lock = asyncio.Lock()

    async def mock_task(item: int) -> int:
        nonlocal active_count, max_active
        async with lock:
            active_count += 1
            if active_count > max_active:
                max_active = active_count

        await asyncio.sleep(0.05)

        async with lock:
            active_count -= 1
        return item * 2

    results = await pool.map(mock_task, list(range(15)))
    assert len(results) == 15
    assert results == [i * 2 for i in range(15)]
    assert max_active <= 5
