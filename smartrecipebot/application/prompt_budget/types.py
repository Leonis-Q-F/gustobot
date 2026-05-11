from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict


class MemorySummary(TypedDict, total=False):
    """滚动摘要的结构化表示。"""

    conversation_summary: str # 对话的整体摘要，捕捉主要事件、决策和主题。
    user_preferences: List[str] # 用户表达的偏好、兴趣和个性化信息。
    constraints: List[str] # 任何硬性约束，如预算限制、时间限制或特定要求。
    open_loops: List[str] # 仍未解决或需要进一步信息的悬而未决的问题或主题。
    important_entities: List[str] # 对话中提到的关键实体，如人、地点、组织或概念。
    confirmed_facts: List[str] # 已经确认并且在对话中反复提到的稳定事实。


class FactMemory(TypedDict, total=False):
    """从滚动摘要中提炼出的稳定事实记忆。"""

    preferences: List[str] # 用户的稳定偏好和兴趣，已经在对话中多次确认。
    hard_constraints: List[str] # 任何持续存在的硬性约束，如预算限制或特定要求。
    active_topics: List[str] # 当前对话中仍然活跃的主题或悬而未决的问题，可能需要在未来的消息中参考。


class PromptMemoryState(TypedDict, total=False):
    """长会话记忆的统一状态快照。"""

    schema_version: int # 记忆状态的版本号，以便未来的兼容性和迁移。
    recent_messages: List[Dict[str, Any]] # 最近的消息列表，包含角色、内容和时间戳等基本信息。
    rolling_summary: MemorySummary # 从最近消息中提炼出的滚动摘要，捕捉对话的主要内容和动态。
    fact_memory: FactMemory# 从滚动摘要中提炼出的稳定事实记忆，包含用户偏好、约束和活跃主题等关键信息。
    pinned_message_ids: List[str] # 任何被固定以确保保留在预算内的消息ID列表。
    last_compaction_seq: int # 上一次进行记忆压缩的消息序列号，以便跟踪何时需要再次压缩。
    summary_version: int # 当前滚动摘要的版本号，以便跟踪何时需要更新摘要。


@dataclass
class PromptSegment:
    """可被预算器打包、裁剪和排序的上下文片段。"""

    name: str
    content: str
    priority: int
    kind: str = "context"
    compressible: bool = True
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetReport:
    """记录一次预算决策的观测结果。"""

    route: str
    model: str
    raw_tokens: int
    final_tokens: int
    reserved_output: int
    available_input_tokens: int
    dropped_segments: List[str] = field(default_factory=list)
    compressed_segments: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class PreparedPrompt:
    """封装预算后的最终消息与更新后的记忆状态。"""

    messages: List[Dict[str, Any]]
    updated_memory_state: PromptMemoryState
    budget_report: BudgetReport
