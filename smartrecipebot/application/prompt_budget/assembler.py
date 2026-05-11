from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from langchain_core.messages import BaseMessage

from .token_counter import TokenCounter
from .types import FactMemory, MemorySummary, PromptSegment


def message_to_openai_dict(message: BaseMessage | dict[str, Any]) -> Dict[str, Any]:
    """将内部消息对象标准化为 OpenAI 风格的 role/content 结构。"""
    if isinstance(message, BaseMessage):
        role = getattr(message, "type", "") or "user"
        content = getattr(message, "content", "")
    else:
        role = message.get("role") or message.get("type") or "user"
        content = message.get("content", "")

    if role == "human":
        role = "user"
    elif role == "ai":
        role = "assistant"
    return {"role": role, "content": content}


def render_fact_memory(fact_memory: FactMemory | None) -> str:
    """把 fact memory 渲染成可直接拼入 prompt 的文本块。"""
    if not fact_memory:
        return ""
    parts: List[str] = []
    preferences = fact_memory.get("preferences") or []
    constraints = fact_memory.get("hard_constraints") or []
    active_topics = fact_memory.get("active_topics") or []
    if preferences:
        parts.append("用户偏好: " + "；".join(preferences))
    if constraints:
        parts.append("硬约束: " + "；".join(constraints))
    if active_topics:
        parts.append("活跃主题: " + "；".join(active_topics))
    return "\n".join(parts).strip()


def render_rolling_summary(summary: MemorySummary | None) -> str:
    """把 rolling summary 渲染成可直接拼入 prompt 的文本块。"""
    if not summary:
        return ""
    parts: List[str] = []
    if summary.get("conversation_summary"):
        parts.append("长期摘要: " + str(summary["conversation_summary"]).strip())
    if summary.get("open_loops"):
        parts.append("未完成问题: " + "；".join(summary["open_loops"]))
    if summary.get("confirmed_facts"):
        parts.append("已确认事实: " + "；".join(summary["confirmed_facts"]))
    if summary.get("important_entities"):
        parts.append("关键实体: " + "；".join(summary["important_entities"]))
    return "\n".join(parts).strip()


def pack_retrieval_segments(
    segments: Sequence[PromptSegment],
    *,
    counter: TokenCounter,
    budget_tokens: int,
) -> Tuple[List[PromptSegment], List[str]]:
    """按预算打包检索片段，并尽量去除重复内容。

    运作流程：
    1. 先按优先级排序。
    2. 用来源与正文做去重键。
    3. 逐条尝试装入预算桶。
    4. 超预算或重复的片段进入 dropped 列表。
    """
    if budget_tokens <= 0:
        return [], [segment.name for segment in segments]

    packed: List[PromptSegment] = []
    dropped: List[str] = []
    total_tokens = 0

    ordered = sorted(segments, key=lambda item: (item.priority, item.name))
    seen_payloads: set[str] = set()
    for segment in ordered:
        payload = json.dumps(
            {
                "name": segment.name,
                "source": segment.source,
                "content": segment.content.strip(),
            },
            ensure_ascii=False,
        )
        dedupe_key = json.dumps(
            {
                "source": segment.source,
                "content": segment.content.strip(),
            },
            ensure_ascii=False,
        )
        if dedupe_key in seen_payloads:
            dropped.append(segment.name)
            continue

        tokens = counter.count_text(payload)
        if total_tokens + tokens > budget_tokens:
            dropped.append(segment.name)
            continue
        seen_payloads.add(dedupe_key)
        packed.append(segment)
        total_tokens += tokens

    return packed, dropped


def render_retrieval_segments(segments: Iterable[PromptSegment]) -> str:
    """将检索片段渲染为供模型阅读的上下文文本。"""
    lines: List[str] = []
    for segment in segments:
        header = segment.name
        if segment.source:
            header = f"{header} | 来源: {segment.source}"
        lines.append(f"[{header}]\n{segment.content.strip()}")
    return "\n\n".join(lines).strip()


def build_budgeted_messages(
    *,
    system_prompt: str,
    current_user_message: str,
    recent_messages: Sequence[BaseMessage | dict[str, Any]],
    fact_memory: FactMemory | None,
    rolling_summary: MemorySummary | None,
    retrieval_context: str = "",
) -> List[Dict[str, Any]]:
    """按固定层次组装最终发送给模型的消息列表。

    组装顺序：
    1. system prompt
    2. fact memory 与 rolling summary
    3. retrieval context
    4. recent window
    5. 当前用户问题
    """
    memory_sections: List[str] = []
    fact_text = render_fact_memory(fact_memory)
    if fact_text:
        memory_sections.append("会话事实记忆:\n" + fact_text)
    summary_text = render_rolling_summary(rolling_summary)
    if summary_text:
        memory_sections.append("会话长期摘要:\n" + summary_text)

    merged_system_prompt = system_prompt.strip()
    if memory_sections:
        merged_system_prompt = (
            merged_system_prompt + "\n\n" + "\n\n".join(memory_sections)
        ).strip()

    messages: List[Dict[str, Any]] = [{"role": "system", "content": merged_system_prompt}]
    if retrieval_context:
        messages.append(
            {
                "role": "system",
                "content": "以下是供回答参考的检索/工具上下文，请只在相关时使用:\n" + retrieval_context,
            }
        )

    for message in recent_messages:
        openai_message = message_to_openai_dict(message)
        if openai_message["role"] in {"user", "assistant"} and openai_message.get("content"):
            messages.append(openai_message)

    messages.append({"role": "user", "content": current_user_message})
    return messages


def serialize_recent_history(
    memory_state: dict[str, Any],
    *,
    recent_turn_limit: int = 4,
) -> List[Dict[str, str]]:
    """把 prompt_memory 序列化成轻量历史记录。

    这个函数主要供 KB/GraphRAG 子工作流使用：
    先注入摘要态记忆，再附加最近几轮原始对话。
    """
    recent_messages = list(memory_state.get("recent_messages", []))
    rendered: List[Dict[str, str]] = []

    summary_text = render_rolling_summary(memory_state.get("rolling_summary"))
    fact_text = render_fact_memory(memory_state.get("fact_memory"))
    synthetic_parts = [part for part in [summary_text, fact_text] if part]
    if synthetic_parts:
        rendered.append({"role": "system", "content": "\n".join(synthetic_parts)})

    for message in recent_messages[-recent_turn_limit:]:
        data = message.get("data") if isinstance(message.get("data"), dict) else message
        role = data.get("type") or data.get("role") or "user"
        if role == "human":
            role = "user"
        elif role == "ai":
            role = "assistant"
        content = str(data.get("content", "")).strip()
        if content:
            rendered.append({"role": role, "content": content})
    return rendered
