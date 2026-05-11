from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .token_counter import TokenCounter
from .types import FactMemory, MemorySummary


class MemorySummaryOutput(BaseModel):
    """摘要模型的结构化输出格式。"""

    conversation_summary: str = Field(default="")
    user_preferences: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    open_loops: List[str] = Field(default_factory=list)
    important_entities: List[str] = Field(default_factory=list)
    confirmed_facts: List[str] = Field(default_factory=list)


SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "你负责维护对话长记忆。请把旧摘要与被滑出窗口的历史对话融合为新的结构化摘要。"
                "不要复述寒暄，不要编造事实，只保留长期有效的信息。"
                "输出字段必须完整，使用简体中文。"
            ),
        ),
        (
            "human",
            (
                "旧摘要(JSON):\n{previous_summary}\n\n"
                "被压缩的历史消息:\n{evicted_messages}\n\n"
                "请生成新的结构化摘要。"
            ),
        ),
    ]
)


class DynamicSummaryService:
    """把被滑出窗口的消息沉淀为长期摘要与事实记忆。

    运作流程：
    1. 接收旧摘要和本次被驱逐的消息。
    2. 优先调用 LLM 生成新的结构化摘要。
    3. LLM 不可用或失败时，降级为确定性 fallback 摘要。
    4. 对摘要字段执行归一化与 token 截断。
    5. 再从摘要中提炼出更短、更稳定的 fact memory。
    """

    def __init__(
        self,
        *,
        llm: Optional[BaseChatModel],
        counter: TokenCounter,
        summary_max_tokens: int,
        fact_max_tokens: int,
    ) -> None:
        """初始化摘要器及其 token 约束。"""
        self.llm = llm
        self.counter = counter
        self.summary_max_tokens = summary_max_tokens
        self.fact_max_tokens = fact_max_tokens

    async def rewrite(
        self,
        previous_summary: MemorySummary | None,
        evicted_messages: Sequence[dict[str, Any]],
        *,
        full_rewrite: bool = False,
    ) -> Tuple[MemorySummary, FactMemory, bool]:
        """把旧摘要与被驱逐消息重写为新的摘要与事实记忆。"""
        summary = previous_summary or {}

        if not evicted_messages:
            normalized = self._normalize_summary(summary)
            return normalized, self._build_fact_memory(normalized), False

        if self.llm is not None:
            try:
                chain = SUMMARY_PROMPT | self.llm.with_structured_output(MemorySummaryOutput)
                result: MemorySummaryOutput = await chain.ainvoke(
                    {
                        "previous_summary": json.dumps(summary, ensure_ascii=False),
                        "evicted_messages": self._render_evicted_messages(evicted_messages, full_rewrite=full_rewrite),
                    }
                )
                normalized = self._normalize_summary(result.model_dump())
                return normalized, self._build_fact_memory(normalized), True
            except Exception:
                pass

        fallback = self._fallback_summary(summary, evicted_messages)
        normalized = self._normalize_summary(fallback)
        return normalized, self._build_fact_memory(normalized), False

    def _render_evicted_messages(
        self,
        evicted_messages: Sequence[dict[str, Any]],
        *,
        full_rewrite: bool,
    ) -> str:
        """把被驱逐消息渲染成摘要模型可消费的文本。"""
        lines: List[str] = []
        if full_rewrite:
            lines.append("模式: 全量归一化重写")
        for message in evicted_messages:
            data = message.get("data") if isinstance(message.get("data"), dict) else message
            role = data.get("type") or message.get("type") or data.get("role") or "user"
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            content = data.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _fallback_summary(
        self,
        previous_summary: MemorySummary,
        evicted_messages: Sequence[dict[str, Any]],
    ) -> MemorySummary:
        """在没有 LLM 时执行保守的摘要降级逻辑。"""
        conversation_summary = previous_summary.get("conversation_summary", "")
        new_lines: List[str] = []
        for message in evicted_messages[-6:]:
            data = message.get("data") if isinstance(message.get("data"), dict) else message
            role = data.get("type") or message.get("type") or data.get("role") or "user"
            content = str(data.get("content", "")).strip()
            if role == "ai":
                role = "assistant"
            elif role == "human":
                role = "user"
            if content:
                new_lines.append(f"{role}: {content}")

        merged_summary = "\n".join(item for item in [conversation_summary, *new_lines] if item).strip()
        return {
            "conversation_summary": self.counter.truncate_text(merged_summary, self.summary_max_tokens),
            "user_preferences": list(previous_summary.get("user_preferences", []))[:8],
            "constraints": list(previous_summary.get("constraints", []))[:8],
            "open_loops": list(previous_summary.get("open_loops", []))[:8],
            "important_entities": list(previous_summary.get("important_entities", []))[:8],
            "confirmed_facts": list(previous_summary.get("confirmed_facts", []))[:8],
        }

    def _normalize_summary(self, payload: Dict[str, Any]) -> MemorySummary:
        """统一裁剪摘要字段，保证结构和长度稳定。"""
        summary: MemorySummary = {
            "conversation_summary": self.counter.truncate_text(
                str(payload.get("conversation_summary", "")).strip(),
                self.summary_max_tokens,
            ),
            "user_preferences": self._trim_list(payload.get("user_preferences", [])),
            "constraints": self._trim_list(payload.get("constraints", [])),
            "open_loops": self._trim_list(payload.get("open_loops", [])),
            "important_entities": self._trim_list(payload.get("important_entities", [])),
            "confirmed_facts": self._trim_list(payload.get("confirmed_facts", [])),
        }
        return summary

    def _trim_list(self, values: Sequence[Any], *, max_items: int = 8, max_item_tokens: int = 64) -> List[str]:
        """裁剪摘要列表字段的数量和单项长度。"""
        items: List[str] = []
        for value in values:
            text = str(value).strip()
            if not text:
                continue
            items.append(self.counter.truncate_text(text, max_item_tokens))
            if len(items) >= max_items:
                break
        return items

    def _build_fact_memory(self, summary: MemorySummary) -> FactMemory:
        """从长期摘要中提炼更短、更稳定的事实记忆。"""
        fact_memory: FactMemory = {
            "preferences": list(summary.get("user_preferences", []))[:6],
            "hard_constraints": list(summary.get("constraints", []))[:6],
            "active_topics": list(summary.get("open_loops", []))[:4] + list(summary.get("important_entities", []))[:4],
        }

        serialized = json.dumps(fact_memory, ensure_ascii=False)
        if self.counter.count_text(serialized) <= self.fact_max_tokens:
            return fact_memory

        while self.counter.count_text(json.dumps(fact_memory, ensure_ascii=False)) > self.fact_max_tokens:
            for key in ("active_topics", "hard_constraints", "preferences"):
                values = fact_memory.get(key, [])
                if values:
                    values.pop()
                    break
            else:
                break
        return fact_memory
