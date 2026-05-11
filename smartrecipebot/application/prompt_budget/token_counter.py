from __future__ import annotations

from typing import Any, Iterable, List, Sequence

import tiktoken
from langchain_core.messages import BaseMessage


class TokenCounter:
    """统一的 token 计数与截断工具。

    运作流程：
    1. 初始化时优先按模型名解析 tokenizer。
    2. 若模型名无法识别，则回退到通用 tokenizer。
    3. 对文本、消息、内容片段提供统一计数接口。
    4. 为滑动窗口、摘要裁剪、检索打包提供长度基准。
    """

    def __init__(self, *, model_name: str | None = None, fallback_encoding: str = "cl100k_base") -> None:
        """初始化 token 计数器并绑定底层 tokenizer。"""
        self.model_name = model_name or ""
        self.fallback_encoding = fallback_encoding
        self._encoding = self._resolve_encoding(self.model_name)

    def _resolve_encoding(self, model_name: str):
        """解析模型对应的 tokenizer，不可用时回退到默认编码。"""
        if model_name:
            try:
                return tiktoken.encoding_for_model(model_name)
            except Exception:
                pass
        return tiktoken.get_encoding(self.fallback_encoding)

    @property
    def encoding_name(self) -> str:
        """返回当前使用的 tokenizer 名称。"""
        return getattr(self._encoding, "name", self.fallback_encoding)

    def encode(self, text: str) -> List[int]:
        """将文本编码为 token id 列表。"""
        if not text:
            return []
        return list(self._encoding.encode(text))

    def count_text(self, text: str) -> int:
        """统计一段文本的 token 数量。"""
        return len(self.encode(text))

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """按 token 上限截断文本，并尽量保留可读结果。"""
        if max_tokens <= 0 or not text:
            return ""
        encoded = self.encode(text)
        if len(encoded) <= max_tokens:
            return text
        return self._encoding.decode(encoded[:max_tokens]).strip()

    def count_message(self, message: BaseMessage | dict[str, Any]) -> int:
        """统计单条消息的 token 数量，包含近似的 chat 开销。"""
        role = ""
        content: Any = ""
        if isinstance(message, BaseMessage):
            role = getattr(message, "type", "") or ""
            content = getattr(message, "content", "")
        else:
            role = str(message.get("role") or message.get("type") or "")
            content = message.get("content", "")

        overhead = 4
        return overhead + self.count_text(role) + self.count_content(content)

    def count_content(self, content: Any) -> int:
        """统计消息 content 字段的 token 数量。"""
        if content is None:
            return 0
        if isinstance(content, str):
            return self.count_text(content)
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(str(item.get("text", item.get("content", ""))))
                else:
                    parts.append(str(item))
            return self.count_text("".join(parts))
        if isinstance(content, dict):
            return self.count_text(str(content))
        return self.count_text(str(content))

    def count_messages(self, messages: Sequence[BaseMessage | dict[str, Any]]) -> int:
        """统计整组消息的 token 数量。"""
        total = 2
        for message in messages:
            total += self.count_message(message)
        return total

    def count_segments(self, segments: Iterable[str]) -> int:
        """统计多个文本片段的总 token 数量。"""
        return sum(self.count_text(segment) for segment in segments)
