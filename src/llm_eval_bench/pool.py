import asyncio
import time
from typing import Any, Awaitable, Callable, List, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class TokenBucketRateLimiter:
    """Thread-safe and async-safe token bucket rate limiter."""

    def __init__(self, rate: float, capacity: float):
        """
        Args:
            rate: Token generation rate per second.
            capacity: Maximum bucket capacity.
        """
        self.rate = float(rate)
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        """Wait until sufficient tokens are available in the bucket."""
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self.last_refill
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.last_refill = now

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return

                # Calculate required sleep time
                deficit = tokens - self.tokens
                sleep_duration = deficit / self.rate

            await asyncio.sleep(max(0.001, sleep_duration))


class AsyncWorkerPool:
    """High-throughput async execution harness bounded by Semaphore and RateLimiter."""

    def __init__(self, concurrency: int = 10, requests_per_sec: float = 50.0):
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)
        self.rate_limiter = TokenBucketRateLimiter(rate=requests_per_sec, capacity=requests_per_sec)

    async def execute(self, func: Callable[[T], Awaitable[R]], item: T) -> R:
        """Executes a single async unit bounded by concurrency semaphore and rate limiter."""
        await self.rate_limiter.acquire(1.0)
        async with self.semaphore:
            return await func(item)

    async def map(self, func: Callable[[T], Awaitable[R]], items: List[T]) -> List[R]:
        """Maps an async task across an input list concurrently."""
        tasks = [self.execute(func, item) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=False)
