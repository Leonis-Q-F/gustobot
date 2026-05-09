from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Sequence
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, messages_from_dict, messages_to_dict
from langgraph.graph.state import CompiledStateGraph
from langchain_openai import ChatOpenAI
from loguru import logger

from gustobot.application.prompt_budget import PromptBudgetManager, RedisPromptMemoryStore, TokenCounter
from gustobot.application.prompt_budget.policy import resolve_route_budget
from gustobot.application.prompt_budget.sliding_window import compact_window
from gustobot.application.prompt_budget.types import PromptMemoryState
from gustobot.config import settings


TEXT_ROUTES = {
    "general-query",
    "additional-query",
    "kb-query",
    "graphrag-query",
    "text2sql-query",
}


@dataclass
class PreparedGraphTurn:
    input_state: Dict[str, Any]
    prompt_memory: PromptMemoryState
    user_message: HumanMessage


def _extract_configurable(config: Dict[str, Any]) -> Dict[str, Any]:
    value = config.get("configurable", {})
    return value if isinstance(value, dict) else {}


def _message_id(message: BaseMessage) -> str:
    return str(getattr(message, "id", "") or "")


def _serialise_messages(messages: Sequence[BaseMessage]) -> List[Dict[str, Any]]:
    return list(messages_to_dict(list(messages)))


def _deserialise_messages(payload: Sequence[Dict[str, Any]]) -> List[BaseMessage]:
    return list(messages_from_dict(list(payload)))


def _bootstrap_memory_state(messages: Sequence[BaseMessage]) -> PromptMemoryState:
    return {
        "schema_version": 1,
        "recent_messages": _serialise_messages(messages),
        "rolling_summary": {},
        "fact_memory": {},
        "pinned_message_ids": [],
        "last_compaction_seq": 0,
        "summary_version": 1,
    }


class ChatGraphRuntime:
    """Unified runtime wrapper for LangGraph chat execution with prompt memory."""

    def __init__(
        self,
        *,
        graph: CompiledStateGraph,
        manager: Optional[PromptBudgetManager] = None,
        store: Optional[RedisPromptMemoryStore] = None,
    ) -> None:
        self.graph = graph
        self.store = store or RedisPromptMemoryStore()
        self.manager = manager or PromptBudgetManager(
            counter=TokenCounter(
                model_name=settings.OPENAI_MODEL,
                fallback_encoding=settings.PROMPT_BUDGET_TOKENIZER,
            ),
            summary_llm=self._build_summary_llm(),
        )

    def _build_summary_llm(self) -> Optional[ChatOpenAI]:
        if not settings.OPENAI_API_KEY:
            return None
        try:
            return ChatOpenAI(
                openai_api_key=settings.OPENAI_API_KEY,
                model_name=settings.OPENAI_MODEL,
                openai_api_base=settings.OPENAI_API_BASE,
                temperature=0.0,
                tags=["prompt_budget_summary"],
            )
        except Exception as exc:
            logger.warning("Failed to initialise prompt-budget summary model: {}", exc)
            return None

    async def _aget_snapshot(self, config: Dict[str, Any]) -> Any:
        if hasattr(self.graph, "aget_state"):
            return await self.graph.aget_state(config)
        return self.graph.get_state(config)

    async def _aupdate_state(self, config: Dict[str, Any], values: Dict[str, Any]) -> None:
        if hasattr(self.graph, "aupdate_state"):
            await self.graph.aupdate_state(config, values)
            return
        self.graph.update_state(config, values)

    async def invoke(self, *, query: str, config: Dict[str, Any]) -> Dict[str, Any]:
        if not settings.PROMPT_BUDGET_ENABLED:
            return await self.graph.ainvoke(
                {"messages": [HumanMessage(content=query, id=str(uuid4()))]},
                config=config,
            )

        prepared = await self.prepare_turn(query=query, config=config)
        result = await self.graph.ainvoke(prepared.input_state, config=config)
        await self.finalize_turn(
            config=config,
            result=result,
            prepared=prepared,
        )
        return result

    async def prepare_turn(self, *, query: str, config: Dict[str, Any]) -> PreparedGraphTurn:
        configurable = _extract_configurable(config) 
        session_id = str(configurable.get("thread_id") or "")

        snapshot = await self._aget_snapshot(config) # 尝试从图状态中获取当前对话快照
        values = snapshot.values if snapshot is not None else {} 
        thread_messages = list(values.get("messages", [])) # 从快照中提取当前线程的消息列表

        prompt_memory: PromptMemoryState = {}
        if isinstance(values.get("prompt_memory"), dict) and values.get("prompt_memory"):
            prompt_memory = dict(values.get("prompt_memory"))
        elif session_id:
            prompt_memory = await self.store.load(session_id) or {} # 如果图状态中没有有效的记忆状态，则尝试从持久化存储中加载

        if not prompt_memory and thread_messages:
            prompt_memory = _bootstrap_memory_state(thread_messages) # 如果仍然没有记忆状态，但有线程消息，则使用这些消息引导一个新的记忆状态
            prompt_memory = await self.manager.append_turn( # 将引导消息追加到记忆中，以便生成初始的滚动摘要和事实记忆
                route="general-query",
                memory_state=prompt_memory,
                new_messages=[],
                force_hard_compaction=True,
            )
        elif prompt_memory and not thread_messages and prompt_memory.get("recent_messages"):
            thread_messages = _deserialise_messages(prompt_memory.get("recent_messages", []))

        removals: List[RemoveMessage] = []
        if thread_messages:
            raw_serialized = _serialise_messages(thread_messages)
            policy = resolve_route_budget("general-query")
            result = compact_window(
                raw_serialized,
                counter=self.manager.counter,
                trigger_limit=policy.recent_hard_limit,
                target_limit=policy.recent_target_limit,
                pinned_ids=(prompt_memory or {}).get("pinned_message_ids", []),
            )
            if result.evicted_messages:
                removals = [
                    RemoveMessage(id=message["data"]["id"])
                    for message in result.evicted_messages
                    if isinstance(message.get("data"), dict) and message["data"].get("id")
                ]
                prompt_memory = prompt_memory or _bootstrap_memory_state(thread_messages)
                prompt_memory["recent_messages"] = list(result.kept_messages)

        user_message = HumanMessage(content=query, id=str(uuid4()))
        if thread_messages:
            input_messages: List[BaseMessage] = [*removals, user_message]
        else:
            restored_messages = _deserialise_messages(prompt_memory.get("recent_messages", []))
            input_messages = [*restored_messages, user_message]

        prompt_memory.setdefault("schema_version", 1)
        prompt_memory.setdefault("rolling_summary", {})
        prompt_memory.setdefault("fact_memory", {})
        prompt_memory.setdefault("recent_messages", prompt_memory.get("recent_messages", []))
        prompt_memory.setdefault("pinned_message_ids", [])
        prompt_memory.setdefault("last_compaction_seq", 0)
        prompt_memory.setdefault("summary_version", 1)

        return PreparedGraphTurn(
            input_state={"messages": input_messages, "prompt_memory": prompt_memory},
            prompt_memory=prompt_memory,
            user_message=user_message,
        )

    async def finalize_turn(
        self,
        *,
        config: Dict[str, Any],
        result: Dict[str, Any],
        prepared: PreparedGraphTurn,
    ) -> None:
        if not settings.PROMPT_BUDGET_ENABLED:
            return

        snapshot = await self._aget_snapshot(config)
        values = snapshot.values if snapshot is not None else {}
        router_info = values.get("router", {}) if isinstance(values.get("router"), dict) else {}
        route = str(router_info.get("type") or result.get("router", {}).get("type") or "general-query")
        if route not in TEXT_ROUTES:
            return

        current_memory = values.get("prompt_memory") if isinstance(values.get("prompt_memory"), dict) else None
        memory_state = dict(current_memory or prepared.prompt_memory or {})
        all_messages = list(values.get("messages", []))
        assistant_message = self._pick_assistant_message(all_messages, fallback=result.get("messages", []))
        if assistant_message is None:
            return

        new_messages = _serialise_messages([prepared.user_message, assistant_message])
        updated_memory = await self.manager.append_turn(
            route=route,
            memory_state=memory_state,
            new_messages=new_messages,
            force_hard_compaction=True,
        )

        try:
            await self._aupdate_state(config, {"prompt_memory": updated_memory})
        except Exception as exc:
            logger.warning("Failed to mirror prompt memory into LangGraph thread state: {}", exc)

        session_id = str(_extract_configurable(config).get("thread_id") or "")
        if session_id:
            await self.store.save(session_id, updated_memory)

    async def astream_messages(
        self,
        *,
        query: str,
        config: Dict[str, Any],
    ) -> AsyncGenerator[tuple[Any, Dict[str, Any]], None]:
        if not settings.PROMPT_BUDGET_ENABLED:
            input_state = {"messages": [HumanMessage(content=query, id=str(uuid4()))]}
            async for chunk, metadata in self.graph.astream(
                input=input_state,
                stream_mode="messages",
                config=config,
            ):
                yield chunk, metadata
            return

        prepared = await self.prepare_turn(query=query, config=config)
        async for chunk, metadata in self.graph.astream(
            input=prepared.input_state,
            stream_mode="messages",
            config=config,
        ):
            yield chunk, metadata

        snapshot = await self._aget_snapshot(config)
        values = snapshot.values if snapshot is not None else {}
        result = {"messages": list(values.get("messages", [])), "router": values.get("router", {})}
        await self.finalize_turn(config=config, result=result, prepared=prepared)

    def _pick_assistant_message(
        self,
        messages: Sequence[BaseMessage],
        *,
        fallback: Sequence[Any],
    ) -> Optional[AIMessage]:
        for message in reversed(messages):
            if getattr(message, "type", None) == "ai":
                if getattr(message, "id", None):
                    return message if isinstance(message, AIMessage) else AIMessage(
                        content=getattr(message, "content", ""),
                        id=_message_id(message),
                    )
                return AIMessage(content=getattr(message, "content", ""), id=str(uuid4()))

        for message in reversed(fallback):
            if getattr(message, "type", None) == "ai":
                return message if isinstance(message, AIMessage) else AIMessage(
                    content=getattr(message, "content", ""),
                    id=str(uuid4()),
                )
        return None
