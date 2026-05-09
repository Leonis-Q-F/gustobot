from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Sequence

from langchain_core.language_models import BaseChatModel

from gustobot.config import settings

from .assembler import build_budgeted_messages, pack_retrieval_segments
from .dynamic_summary import DynamicSummaryService
from .policy import resolve_route_budget
from .sliding_window import compact_window
from .token_counter import TokenCounter
from .types import BudgetReport, PreparedPrompt, PromptMemoryState, PromptSegment


class PromptBudgetManager:
    """统一协调 prompt 预算、滑窗压缩、摘要更新与检索打包。

    运作流程：
    1. 根据路由解析 RouteBudgetPolicy。
    2. 归一化 prompt_memory，拿到 recent window、rolling summary、fact memory。
    3. 对 recent window 先做滑动压缩，把旧消息沉淀到摘要层。
    4. 对检索片段执行去重、排序和 token-aware 装箱。
    5. 按固定层次组装最终 prompt。
    6. 若仍超预算，按顺序降级：先丢检索，再收紧 recent window。
    7. 返回最终消息、更新后的记忆状态和预算报告。
    """

    def __init__(
        self,
        *,
        counter: Optional[TokenCounter] = None,
        summary_llm: Optional[BaseChatModel] = None,
    ) -> None:
        """初始化预算管理器及其依赖组件。"""
        self.counter = counter or TokenCounter(
            model_name=settings.OPENAI_MODEL,
            fallback_encoding=settings.PROMPT_BUDGET_TOKENIZER,
        )
        self.summary_llm = summary_llm

    async def prepare(
        self,
        *,
        route: str,
        session_id: str,
        system_prompt: str,
        current_user_message: str,
        recent_messages: list[dict],
        rolling_summary: dict | None,
        fact_memory: dict | None,
        retrieval_segments: list[dict] | None = None,
        memory_state: dict | None = None,
    ) -> PreparedPrompt:
        """为一次文本模型调用准备预算受控的 prompt。"""
        policy = resolve_route_budget(route)
        state = self._coerce_memory_state(
            recent_messages=recent_messages,
            rolling_summary=rolling_summary,
            fact_memory=fact_memory,
            memory_state=memory_state,
        )
        report = BudgetReport(
            route=route,
            model=self.counter.model_name or settings.OPENAI_MODEL,
            raw_tokens=0,
            final_tokens=0,
            reserved_output=policy.output_reserve,
            available_input_tokens=policy.available_input_tokens,
        )

        recent_window = list(state.get("recent_messages", []))
        raw_segments = self._coerce_retrieval_segments(retrieval_segments or [])
        raw_messages = build_budgeted_messages(
            system_prompt=system_prompt,
            current_user_message=current_user_message,
            recent_messages=recent_window,
            fact_memory=state.get("fact_memory"),
            rolling_summary=state.get("rolling_summary"),
            retrieval_context=self._render_packed_segments(raw_segments),
        )
        report.raw_tokens = self.counter.count_messages(raw_messages)

        recent_window, state = await self._compact_recent_window(
            route=route,
            state=state,
            recent_window=recent_window,
            target_limit=min(policy.recent_target_limit, policy.history_budget),
            trigger_limit=policy.recent_soft_limit,
            report=report,
        )

        packed_segments, dropped_segments = pack_retrieval_segments(
            raw_segments,
            counter=self.counter,
            budget_tokens=policy.retrieval_budget,
        )
        if dropped_segments:
            report.dropped_segments.extend(dropped_segments)

        messages = build_budgeted_messages(
            system_prompt=system_prompt,
            current_user_message=current_user_message,
            recent_messages=recent_window,
            fact_memory=state.get("fact_memory"),
            rolling_summary=state.get("rolling_summary"),
            retrieval_context=self._render_packed_segments(packed_segments),
        )

        final_tokens = self.counter.count_messages(messages)
        if final_tokens > policy.available_input_tokens and packed_segments:
            report.compressed_segments.append("retrieval_segments")
            messages = build_budgeted_messages(
                system_prompt=system_prompt,
                current_user_message=current_user_message,
                recent_messages=recent_window,
                fact_memory=state.get("fact_memory"),
                rolling_summary=state.get("rolling_summary"),
                retrieval_context="",
            )
            final_tokens = self.counter.count_messages(messages)

        if final_tokens > policy.available_input_tokens and recent_window:
            tighter_target = max(min(policy.history_budget, policy.recent_target_limit // 2), 1024)
            recent_window, state = await self._compact_recent_window(
                route=route,
                state=state,
                recent_window=recent_window,
                target_limit=tighter_target,
                trigger_limit=0,
                report=report,
            )
            messages = build_budgeted_messages(
                system_prompt=system_prompt,
                current_user_message=current_user_message,
                recent_messages=recent_window,
                fact_memory=state.get("fact_memory"),
                rolling_summary=state.get("rolling_summary"),
                retrieval_context="",
            )
            final_tokens = self.counter.count_messages(messages)

        state["recent_messages"] = recent_window
        state.setdefault("schema_version", 1)
        report.final_tokens = final_tokens
        if final_tokens > policy.available_input_tokens:
            report.notes.append("Prompt still above target budget after compression fallback.")

        return PreparedPrompt(
            messages=messages,
            updated_memory_state=state,
            budget_report=report,
        )

    async def append_turn(
        self,
        *,
        route: str,
        memory_state: dict | None,
        new_messages: Sequence[dict[str, Any]],
        force_hard_compaction: bool = True,
    ) -> PromptMemoryState:
        """在一轮问答结束后，把新消息并入 recent window 并按需压缩。"""
        policy = resolve_route_budget(route)
        state = self._coerce_memory_state(
            recent_messages=[],
            rolling_summary=None,
            fact_memory=None,
            memory_state=memory_state,
        )
        recent_window = list(state.get("recent_messages", []))
        recent_window.extend(deepcopy(list(new_messages)))
        state["recent_messages"] = recent_window

        _, state = await self._compact_recent_window(
            route=route,
            state=state,
            recent_window=recent_window,
            target_limit=policy.recent_target_limit,
            trigger_limit=policy.recent_hard_limit if force_hard_compaction else policy.recent_soft_limit,
            report=None,
        )
        state.setdefault("schema_version", 1)
        return state

    def _coerce_memory_state(
        self,
        *,
        recent_messages: list[dict],
        rolling_summary: dict | None,
        fact_memory: dict | None,
        memory_state: dict | None,
    ) -> PromptMemoryState:
        """把零散输入归一化为完整的 PromptMemoryState。"""
        state: PromptMemoryState = deepcopy(memory_state or {})
        state.setdefault("schema_version", 1)
        state.setdefault("recent_messages", deepcopy(recent_messages or state.get("recent_messages", [])))
        if rolling_summary is not None:
            state["rolling_summary"] = deepcopy(rolling_summary)
        else:
            state.setdefault("rolling_summary", {})
        if fact_memory is not None:
            state["fact_memory"] = deepcopy(fact_memory)
        else:
            state.setdefault("fact_memory", {})
        state.setdefault("pinned_message_ids", [])
        state.setdefault("last_compaction_seq", 0)
        state.setdefault("summary_version", 1)
        return state

    async def _compact_recent_window(
        self,
        *,
        route: str,
        state: PromptMemoryState,
        recent_window: Sequence[dict[str, Any]],
        target_limit: int,
        trigger_limit: int,
        report: BudgetReport | None,
    ) -> tuple[list[dict[str, Any]], PromptMemoryState]:
        """压缩 recent window，并把被驱逐消息沉淀到摘要层。"""
        if not recent_window:
            return list(recent_window), state

        result = compact_window(
            list(recent_window),
            counter=self.counter,
            trigger_limit=trigger_limit,
            target_limit=target_limit,
            pinned_ids=self._collect_pinned_message_ids(state),
        )
        if not result.evicted_messages:
            return list(result.kept_messages), state

        policy = resolve_route_budget(route)
        summary_service = DynamicSummaryService(
            llm=self.summary_llm,
            counter=self.counter,
            summary_max_tokens=policy.summary_max_tokens,
            fact_max_tokens=policy.fact_max_tokens,
        )
        previous_summary = deepcopy(state.get("rolling_summary", {}))
        full_rewrite = ((int(state.get("last_compaction_seq", 0)) + 1) % 8) == 0
        new_summary, new_fact_memory, used_llm = await summary_service.rewrite(
            previous_summary,
            result.evicted_messages,
            full_rewrite=full_rewrite,
        )
        state["rolling_summary"] = new_summary
        state["fact_memory"] = new_fact_memory
        state["recent_messages"] = list(result.kept_messages)
        state["last_compaction_seq"] = int(state.get("last_compaction_seq", 0)) + 1
        state["summary_version"] = int(state.get("summary_version", 1)) + 1

        if report is not None:
            report.compressed_segments.append("recent_window")
            report.notes.append(
                f"Compacted {len(result.evicted_messages)} messages into rolling summary "
                f"({'llm' if used_llm else 'fallback'})."
            )
        return list(result.kept_messages), state

    def _collect_pinned_message_ids(self, state: PromptMemoryState) -> List[str]:
        """收集当前压缩阶段不允许驱逐的消息 id。"""
        pinned = [str(item) for item in state.get("pinned_message_ids", []) if item]
        recent_messages = list(state.get("recent_messages", []))
        for message in recent_messages[-2:]:
            data = message.get("data") if isinstance(message.get("data"), dict) else message
            message_id = data.get("id")
            if message_id:
                pinned.append(str(message_id))
        return list(dict.fromkeys(pinned))

    def _coerce_retrieval_segments(self, retrieval_segments: Iterable[dict[str, Any]]) -> List[PromptSegment]:
        """把外部检索结果统一转换为 PromptSegment。"""
        segments: List[PromptSegment] = []
        for index, segment in enumerate(retrieval_segments):
            content = str(segment.get("content", "")).strip()
            if not content:
                continue
            segments.append(
                PromptSegment(
                    name=str(segment.get("name") or segment.get("label") or f"segment_{index + 1}"),
                    source=segment.get("source"),
                    content=content,
                    priority=int(segment.get("priority", 50)),
                    kind=str(segment.get("kind", "retrieval")),
                    metadata=dict(segment.get("metadata") or {}),
                )
            )
        return segments

    def _render_packed_segments(self, segments: Sequence[PromptSegment]) -> str:
        """把已打包的检索片段渲染成最终 prompt 上下文。"""
        if not segments:
            return ""
        from .assembler import render_retrieval_segments

        return render_retrieval_segments(segments)
