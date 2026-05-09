from __future__ import annotations

from dataclasses import dataclass

from gustobot.config import settings


@dataclass(frozen=True)
class RouteBudgetPolicy:
    """描述某一路由下的预算参数集合。

    这个对象把上下文窗口、输出预留、摘要额度、检索额度
    收口成一份只读策略，供预算器在单次请求内复用。
    """

    route: str
    context_window: int
    output_reserve: int
    safety_margin_ratio: float
    recent_soft_limit: int
    recent_target_limit: int
    recent_hard_limit: int
    summary_max_tokens: int
    fact_max_tokens: int
    history_budget: int
    retrieval_budget: int
    sources_budget: int

    @property
    def available_input_tokens(self) -> int:
        """计算当前路由可用于输入 prompt 的 token 上限。"""
        safety_margin = int(self.context_window * self.safety_margin_ratio)
        return max(self.context_window - self.output_reserve - safety_margin, 2048)


def resolve_route_budget(route: str) -> RouteBudgetPolicy:
    """按路由类型解析具体预算策略。"""
    route_name = (route or "general-query").lower()
    common = dict(
        context_window=settings.PROMPT_BUDGET_CONTEXT_WINDOW,
        output_reserve=settings.PROMPT_BUDGET_OUTPUT_RESERVE,
        safety_margin_ratio=settings.PROMPT_BUDGET_SAFETY_MARGIN_RATIO,
        recent_soft_limit=settings.PROMPT_BUDGET_RECENT_SOFT_LIMIT,
        recent_target_limit=settings.PROMPT_BUDGET_RECENT_TARGET_LIMIT,
        recent_hard_limit=settings.PROMPT_BUDGET_RECENT_HARD_LIMIT,
        summary_max_tokens=settings.PROMPT_BUDGET_SUMMARY_MAX_TOKENS,
        fact_max_tokens=settings.PROMPT_BUDGET_FACT_MAX_TOKENS,
    )

    if route_name in {"kb-query"}:
        return RouteBudgetPolicy(
            route=route_name,
            history_budget=4096,
            retrieval_budget=16384,
            sources_budget=2048,
            **common,
        )
    if route_name in {"graphrag-query", "text2sql-query"}:
        return RouteBudgetPolicy(
            route=route_name,
            history_budget=3072,
            retrieval_budget=12288,
            sources_budget=1024,
            **common,
        )
    return RouteBudgetPolicy(
        route=route_name,
        history_budget=common["recent_target_limit"],
        retrieval_budget=0,
        sources_budget=0,
        **common,
    )
