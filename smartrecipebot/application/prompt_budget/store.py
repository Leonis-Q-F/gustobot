from __future__ import annotations

import json
from typing import Optional

from loguru import logger

try:  # pragma: no cover - optional dependency in minimal test envs
    from redis.asyncio import Redis
except Exception:  # pragma: no cover - graceful fallback
    Redis = None  # type: ignore[assignment]

from smartrecipebot.config import settings

from .types import PromptMemoryState


class RedisPromptMemoryStore:
    """基于 Redis 的 prompt_memory 持久化存储。

    运作流程：
    1. 使用 session_id 生成单 key blob。
    2. 读取时反序列化为 PromptMemoryState。
    3. 写入时整体覆盖并刷新 TTL。
    4. Redis 不可用时自动降级，不阻塞主链路。
    """

    def __init__(
        self,
        *,
        redis_client: Optional[Redis] = None,
        redis_url: Optional[str] = None,
        prefix: Optional[str] = None,
        ttl: Optional[int] = None,
    ) -> None:
        """初始化 Redis 客户端、命名空间前缀和过期时间。"""
        self._redis = redis_client
        if self._redis is None and Redis is not None:
            self._redis = Redis.from_url(
                redis_url or settings.REDIS_URL,
                decode_responses=False,
            )
        self.prefix = prefix or settings.PROMPT_BUDGET_REDIS_PREFIX
        self.ttl = ttl if ttl is not None else settings.CONVERSATION_HISTORY_TTL

    def _key(self, session_id: str) -> str:
        """生成某个会话在 Redis 中的存储 key。"""
        return f"{self.prefix}:{session_id}"

    async def load(self, session_id: str) -> Optional[PromptMemoryState]:
        """读取并反序列化指定会话的记忆状态。"""
        if self._redis is None:
            return None
        try:
            raw = await self._redis.get(self._key(session_id))
        except Exception as exc:
            logger.warning("Failed to load prompt memory from Redis: {}", exc)
            return None

        if not raw:
            return None

        try:
            return json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        except Exception as exc:
            logger.warning("Failed to decode prompt memory payload: {}", exc)
            return None

    async def save(self, session_id: str, memory_state: PromptMemoryState) -> bool:
        """序列化并写入指定会话的记忆状态。"""
        if self._redis is None:
            return False
        try:
            await self._redis.set(
                self._key(session_id),
                json.dumps(memory_state, ensure_ascii=False),
                ex=self.ttl,
            )
            return True
        except Exception as exc:
            logger.warning("Failed to save prompt memory to Redis: {}", exc)
            return False

    async def clear(self, session_id: str) -> None:
        """删除指定会话的记忆状态。"""
        if self._redis is None:
            return
        try:
            await self._redis.delete(self._key(session_id))
        except Exception as exc:
            logger.warning("Failed to clear prompt memory in Redis: {}", exc)
