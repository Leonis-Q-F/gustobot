from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence, Set

from .token_counter import TokenCounter


@dataclass
class SlidingWindowResult:
    """记录一次滑动窗口压缩后的保留结果与驱逐结果。"""

    kept_messages: List[dict[str, Any]]
    evicted_messages: List[dict[str, Any]]
    tokens_before: int
    tokens_after: int


def _message_id(message: dict[str, Any]) -> str:
    """提取消息 id，兼容序列化后的消息结构。"""
    data = message.get("data") if isinstance(message.get("data"), dict) else message
    return str(data.get("id") or "")


def compact_window(
    messages: Sequence[dict[str, Any]],
    *,
    counter: TokenCounter,
    trigger_limit: int,
    target_limit: int,
    pinned_ids: Iterable[str] | None = None,
) -> SlidingWindowResult:
    """执行 recent window 的滑动压缩。

    运作流程：
    1. 先统计当前窗口的总 token。
    2. 未超过触发阈值则原样返回。
    3. 超限后从最旧消息开始驱逐。
    4. 被 pin 的消息跳过，不参与驱逐。
    5. 直到窗口压回目标阈值以下。
    """
    pinned: Set[str] = {str(item) for item in (pinned_ids or []) if item}
    kept = [dict(message) for message in messages]
    evicted: List[dict[str, Any]] = []

    tokens_before = counter.count_messages([_as_openai_message(message) for message in kept])
    current_tokens = tokens_before
    if current_tokens <= trigger_limit:
        return SlidingWindowResult(
            kept_messages=kept,
            evicted_messages=evicted,
            tokens_before=tokens_before,
            tokens_after=current_tokens,
        )

    idx = 0
    while idx < len(kept) and current_tokens > target_limit:
        candidate = kept[idx]
        candidate_id = _message_id(candidate)
        if candidate_id and candidate_id in pinned:
            idx += 1
            continue

        evicted.append(candidate)
        del kept[idx]
        current_tokens = counter.count_messages([_as_openai_message(message) for message in kept])

    return SlidingWindowResult(
        kept_messages=kept,
        evicted_messages=evicted,
        tokens_before=tokens_before,
        tokens_after=current_tokens,
    )


def _as_openai_message(message: dict[str, Any]) -> dict[str, Any]:
    """将内部消息结构标准化为 OpenAI 风格消息字典。"""
    data = message.get("data") if isinstance(message.get("data"), dict) else message
    role = data.get("role") or data.get("type") or message.get("type") or "user"
    if role == "human":
        role = "user"
    elif role == "ai":
        role = "assistant"
    return {"role": role, "content": data.get("content", "")}
