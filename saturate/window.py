"""Admission window: an asyncio gate with a runtime-adjustable limit."""

from __future__ import annotations

import asyncio


class Window:
    def __init__(self, limit: int):
        self.limit = limit
        self.inflight = 0
        self._cond = asyncio.Condition()

    async def acquire(self) -> None:
        async with self._cond:
            while self.inflight >= self.limit:
                await self._cond.wait()
            self.inflight += 1

    async def release(self) -> None:
        async with self._cond:
            self.inflight -= 1
            self._cond.notify_all()

    async def set_limit(self, n: int) -> None:
        if n == self.limit:
            return
        async with self._cond:
            self.limit = n
            self._cond.notify_all()
