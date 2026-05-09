from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Optional

from langgraph.checkpoint.redis import AsyncRedisSaver
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from gustobot.application.agents.lg_builder import build_agent_graph
from gustobot.application.services.chat_runtime import ChatGraphRuntime
from gustobot.config import settings


@dataclass
class RuntimeResources:
    """统一持有运行时资源。"""

    exit_stack: AsyncExitStack
    saver: AsyncRedisSaver
    graph: CompiledStateGraph
    chat_runtime: ChatGraphRuntime


_runtime_resources: Optional[RuntimeResources] = None
_runtime_lock: Optional[asyncio.Lock] = None


def _get_runtime_lock() -> asyncio.Lock:
    global _runtime_lock
    if _runtime_lock is None:
        _runtime_lock = asyncio.Lock()
    return _runtime_lock


def _require_runtime() -> RuntimeResources:
    if _runtime_resources is None:
        raise RuntimeError("运行时尚未初始化，请先调用 initialize_runtime()。")
    return _runtime_resources


async def initialize_runtime() -> RuntimeResources:
    """初始化 Redis Stack checkpointer、graph 和 chat runtime。"""
    global _runtime_resources

    async with _get_runtime_lock():
        if _runtime_resources is not None:
            return _runtime_resources

        exit_stack = AsyncExitStack()
        try:
            saver = await exit_stack.enter_async_context(
                AsyncRedisSaver.from_conn_string(
                    settings.REDIS_URL,
                    checkpoint_prefix=settings.CHECKPOINT_PREFIX,
                    checkpoint_write_prefix=settings.CHECKPOINT_WRITE_PREFIX,
                )
            )
            graph = build_agent_graph(saver)
            chat_runtime = ChatGraphRuntime(graph=graph)
            _runtime_resources = RuntimeResources(
                exit_stack=exit_stack,
                saver=saver,
                graph=graph,
                chat_runtime=chat_runtime,
            )
            logger.info("Redis Stack checkpoint runtime 初始化完成。")
            return _runtime_resources
        except Exception as exc:
            await exit_stack.aclose()
            logger.error("Redis Stack checkpoint runtime 初始化失败: {}", exc)
            raise


def get_graph() -> CompiledStateGraph:
    """获取已初始化的主聊天图。"""
    return _require_runtime().graph


def get_chat_runtime() -> ChatGraphRuntime:
    """获取已初始化的聊天运行时。"""
    return _require_runtime().chat_runtime


async def shutdown_runtime() -> None:
    """关闭运行时资源并断开 Redis 连接。"""
    global _runtime_resources

    async with _get_runtime_lock():
        if _runtime_resources is None:
            return

        await _runtime_resources.exit_stack.aclose()
        _runtime_resources = None
        logger.info("Redis Stack checkpoint runtime 已关闭。")
