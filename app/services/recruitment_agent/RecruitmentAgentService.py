"""招聘 Agent 的统一服务入口：analyze(agent_input)。

服务对象构造时编译一次图，可用于多次独立分析。每次调用创建新的 thread_id，
不复用上一份简历的执行记录，也不在服务层叠加整图重试。

用法::

    async with RecruitmentLLMClient(api_key=api_key) as llm:
        service = RecruitmentAgentService(llm, max_retries=2)
        result = await service.analyze(agent_input)
        if result.status == "succeeded":
            output = result.output  # AgentAnalysisOutput
        else:
            # 部分失败或全部失败仍是正常的业务返回。
            for jd_id, job in result.execution.jobs.items():
                print(jd_id, job.status, job.report, job.last_error)

llm 与可选 checkpointer 的创建、关闭由调用方负责。本服务不创建数据库表，
不修改候选人招聘状态。启用 checkpoint 时需使用支持异步调用的 checkpointer。
本入口始终启动新任务；恢复已有任务应使用 graph 的 checkpoint 恢复入口，
不能再次调用 analyze 来代替恢复。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver

from app.services.middle_layer.models import AgentAnalysisInput
from app.services.recruitment_agent.agent.graph import build_recruitment_graph, read_execution
from app.services.recruitment_agent.agent.nodes import JobAnalyzer
from app.services.recruitment_agent.schema.output_schema import AgentAnalysisOutput
from app.services.recruitment_agent.agent.state import RecruitmentAgentState


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecruitmentAgentResult:
    """业务返回：执行记录只保存一份，状态和汇总报告按需从记录中获得。"""

    thread_id: str
    execution: RecruitmentAgentState

    def __post_init__(self) -> None:
        if not self.execution.is_finished:
            raise ValueError("服务结果必须包含全部到达终态的 JD")

    @property
    def status(self) -> Literal["succeeded", "partial_failed", "failed"]:
        succeeded = sum(job.status == "succeeded" for job in self.execution.jobs.values())
        if succeeded == len(self.execution.jobs):
            return "succeeded"
        return "partial_failed" if succeeded else "failed"

    @property
    def output(self) -> AgentAnalysisOutput | None:
        """全部成功时生成标准输出；部分成功报告可从 execution.jobs 读取。"""
        return self.execution.build_output() if self.status == "succeeded" else None


class RecruitmentAgentServiceError(RuntimeError):
    """输入或图执行异常；不把它们伪装成某个 JD 的分析失败。"""

    def __init__(self, message: str, *, code: str, thread_id: str) -> None:
        super().__init__(message)
        self.code = code
        self.thread_id = thread_id


class RecruitmentAgentService:
    """封装图的输入、调用配置和返回值；可复用，实例不保存当前候选人状态。"""

    def __init__(
        self, llm: JobAnalyzer, *, max_retries: int = 2,
        retry_base_delay: float = 1.0, retry_max_delay: float = 30.0,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        # 配置合法性由图构建器统一校验；依赖生命周期仍归调用方管理。
        self._graph = build_recruitment_graph(
            llm, max_retries=max_retries, retry_base_delay=retry_base_delay,
            retry_max_delay=retry_max_delay, checkpointer=checkpointer,
        )
        self._max_retries = max_retries

    async def analyze(self, agent_input: AgentAnalysisInput) -> RecruitmentAgentResult:
        """分析一份标准输入；JD 的可预期失败保留在结果中。

        非法输入抛出 code=invalid_input；图步数异常、代码错误、checkpoint
        存储异常等抛出 code=graph_execution_failed。调用方取消会原样传播。
        不接受原始 PDF、简历模型或自然语言；这些应先转换成 AgentAnalysisInput。
        """
        thread_id = str(uuid4())
        if not isinstance(agent_input, AgentAnalysisInput):
            raise RecruitmentAgentServiceError(
                "analyze 仅接受 AgentAnalysisInput", code="invalid_input", thread_id=thread_id,
            )
        try:
            # 深度快照并重新校验，避免调用方之后修改嵌套字段影响本次分析。
            snapshot = AgentAnalysisInput.model_validate_json(agent_input.model_dump_json())
        except (ValueError, TypeError):
            raise RecruitmentAgentServiceError(
                "标准输入未通过校验", code="invalid_input", thread_id=thread_id,
            ) from None

        jd_count = len({target.selected_jd_id for target in snapshot.target_matches})
        # 和当前 graph.py 拓扑一致的保守步数预算；预算包含校验、等待和汇总节点。
        # 每个 JD 的实际重试次数仍由 RecruitmentAgentState.max_retries 限制。
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 4 * jd_count * (self._max_retries + 1) + 3,
        }
        started = time.perf_counter()
        logger.info("招聘Agent任务开始 thread_id=%s jd_count=%d", thread_id, jd_count)
        try:
            raw_result = await self._graph.ainvoke(
                {"agent_input": snapshot.model_dump(mode="json")}, config=config,
            )
            execution = read_execution(raw_result)
            if execution.agent_input != snapshot or execution.max_retries != self._max_retries:
                raise ValueError("图返回的输入或重试预算与本次任务不一致")
            result = RecruitmentAgentResult(thread_id=thread_id, execution=execution)
            if raw_result.get("status") != result.status:
                raise ValueError("图返回的汇总状态与 JD 执行结果不一致")
        except asyncio.CancelledError:
            logger.info("招聘Agent任务取消 thread_id=%s", thread_id)
            raise
        except Exception as exc:
            # 不记录原始异常正文/响应/输入，防止将简历内容或凭据写入日志。
            logger.error(
                "招聘Agent图执行异常 thread_id=%s error_type=%s",
                thread_id, type(exc).__name__,
            )
            raise RecruitmentAgentServiceError(
                "招聘分析图执行失败，请通过任务 ID 排查",
                code="graph_execution_failed", thread_id=thread_id,
            ) from None

        logger.info(
            "招聘Agent任务结束 thread_id=%s status=%s elapsed_ms=%.0f",
            thread_id, result.status, (time.perf_counter() - started) * 1000,
        )
        return result


__all__ = ["RecruitmentAgentService", "RecruitmentAgentResult", "RecruitmentAgentServiceError"]
