"""用 LangGraph 连接输入校验、逐 JD 分析、结果校验与有限重试。

新任务用法（调用方负责 llm 的生命周期）::

    graph = build_recruitment_graph(llm, max_retries=2)
    result = await graph.ainvoke({"agent_input": agent_input.model_dump(mode="json")})
    execution = read_execution(result)
    # result["status"]: succeeded / partial_failed / failed
    # result["output"]: 全部成功时为 AgentAnalysisOutput 的 JSON 字典，否则为 None。
    # 部分成功的报告位于 execution.jobs[jd_id].report。

max_retries=2 表示每个 JD 最多首次调用 + 2 次重试；一岗耗尽预算后继续其他岗位。
输入覆盖范围 analysis_scope=partial 与执行结果 partial_failed 含义不同。

checkpoint 可选，由调用方创建并传入；本模块不创建数据库连接或表。
启用后，每个新任务使用新的 configurable.thread_id；恢复未完成任务时使用同一个
thread_id 调用 await graph.ainvoke(None, config)，不要重新提交初始输入。
恢复时沿用同一版本的图、提示词和配置，且同一 thread 不要并发执行。

已完成节点可从 checkpoint 继续。模型请求完成但节点结果尚未保存时中断，
恢复可能重放该请求；本图不保证跨进程崩溃场景的恰好一次调用/计费。
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.services.recruitment_agent.nodes import (
    JobAnalyzer, RecruitmentGraphState, RecruitmentNodes, read_execution,
)


def build_recruitment_graph(
    llm: JobAnalyzer, *, max_retries: int = 2,
    retry_base_delay: float = 1.0, retry_max_delay: float = 30.0,
    checkpointer: BaseCheckpointSaver | None = None,
    recursion_limit: int = 1000,
) -> CompiledStateGraph:
    """构建可复用的异步执行图；用 ainvoke/astream 调用。

    不配置 LangGraph RetryPolicy，避免和显式重试分支叠加。
    recursion_limit 是节点步数保护，不是重试次数。当前串行拓扑的保守步数预算为
    4 * JD数量 * (max_retries + 1) + 3；大量 JD 时可提高 recursion_limit。
    超过图步数上限会抛出 GraphRecursionError，不会伪装成正常业务结果。
    """
    if type(recursion_limit) is not int or recursion_limit < 1:
        raise ValueError("recursion_limit 必须为正整数")
    nodes = RecruitmentNodes(
        llm, max_retries=max_retries,
        retry_base_delay=retry_base_delay, retry_max_delay=retry_max_delay,
    )
    builder = StateGraph(RecruitmentGraphState)
    builder.add_node("validate_input", nodes.validate_input)
    builder.add_node("select_job", nodes.select_job)
    builder.add_node("analyze_job", nodes.analyze_job)
    builder.add_node("validate_result", nodes.validate_result)
    builder.add_node("wait_retry", nodes.wait_retry)
    builder.add_node("finalize", nodes.finalize)

    builder.add_edge(START, "validate_input")
    builder.add_edge("validate_input", "select_job")
    builder.add_conditional_edges("select_job", nodes.route_selected, {
        "analyze_job": "analyze_job", "finalize": "finalize",
    })
    builder.add_conditional_edges("analyze_job", nodes.route_analyzed, {
        "validate_result": "validate_result", "wait_retry": "wait_retry", "select_job": "select_job",
    })
    builder.add_conditional_edges("validate_result", nodes.route_result, {
        "wait_retry": "wait_retry", "select_job": "select_job",
    })
    builder.add_edge("wait_retry", "select_job")
    builder.add_edge("finalize", END)

    compiled = builder.compile(checkpointer=checkpointer, name="recruitment_analysis")
    return compiled.with_config({"recursion_limit": recursion_limit})


__all__ = ["build_recruitment_graph", "read_execution"]
