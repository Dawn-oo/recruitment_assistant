"""招聘分析图的节点与路由；每轮顺序处理一个 JD。

复用 RecruitmentAgentState 管理业务状态，不复制其状态转换逻辑。
图的 channel 保存普通 JSON 数据，节点内再恢复严格模型，方便 checkpoint 序列化。
pending_report 是尚未通过图节点最终校验的暂存结果，不属于已完成报告。
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from typing import Any, Literal, Protocol

from typing_extensions import TypedDict

from app.services.middle_layer.models import AgentAnalysisInput
from app.services.recruitment_agent.agent.llm_client import AgentLLMError
from app.services.recruitment_agent.schema.output_schema import JobAnalysis
from app.services.recruitment_agent.agent.prompts import validate_scoring_rules
from app.services.recruitment_agent.agent.state import RecruitmentAgentState


logger = logging.getLogger(__name__)


class RecruitmentGraphState(TypedDict, total=False):
    # 新任务仅传入 agent_input.model_dump(mode="json")。
    # 输入校验后清空此字段，标准输入只保留在execution.agent_input 中。
    agent_input: dict[str, Any] | None
    execution: dict[str, Any]
    current_jd_id: int | None
    pending_report: dict[str, Any] | None
    retry_after_seconds: float | None
    output: dict[str, Any] | None
    status: Literal["running", "succeeded", "partial_failed", "failed"]


class JobAnalyzer(Protocol):
    """RecruitmentLLMClient 已实现此接口；测试可注入不访问API的实现。"""

    async def analyze_job(self, agent_input: AgentAnalysisInput, *, jd_id: int) -> JobAnalysis:
        ...


def read_execution(state: RecruitmentGraphState) -> RecruitmentAgentState:
    """JSON 校验模式可正确恢复严格模型中的枚举、时间与整数 JD 字典键。"""
    return RecruitmentAgentState.model_validate_json(json.dumps(state["execution"], ensure_ascii=False, allow_nan=False))


class RecruitmentNodes:
    """只持有调用依赖和配置；候选人数据全部位于图状态，支持不同任务复用节点。"""

    def __init__(self, llm: JobAnalyzer, *, max_retries: int = 2,retry_base_delay: float = 1.0, retry_max_delay: float = 30.0) -> None:

        if type(max_retries) is not int or max_retries < 0:
            raise ValueError("max_retries必须是非负整数")
        for value in (retry_base_delay, retry_max_delay):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("重试等待时间必须是非负有限数值")
            if not math.isfinite(value) or value < 0:
                raise ValueError("重试等待时间必须是非负有限数值")
        if retry_base_delay > retry_max_delay:
            raise ValueError("retry_base_delay不能超过retry_max_delay")

        self.llm = llm
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay

    def validate_input(self, state: RecruitmentGraphState) -> RecruitmentGraphState:
        """新任务入口；坏输入直接报错，不发起模型请求，也不消耗重试预算。"""
        if state.get("execution") is not None:
            raise ValueError("当前图已有执行记录；新任务请使用新的 thread_id，恢复任务请传入 None")
        raw = state.get("agent_input")
        if not isinstance(raw, dict):
            raise ValueError("agent_input 必须为标准输入的 JSON 字典")
        try:
            agent_input = AgentAnalysisInput.model_validate_json(
                json.dumps(raw, ensure_ascii=False, allow_nan=False)
            )
        except (ValueError, TypeError):
            raise ValueError("agent_input 未通过标准输入校验") from None
        execution = RecruitmentAgentState.from_input(agent_input, max_retries=self.max_retries)
        logger.info("Agent图输入校验完成 jd_count=%d", len(execution.jobs))
        return {
            "agent_input": None, "execution": execution.model_dump(mode="json"),
            "current_jd_id": None, "pending_report": None,
            "retry_after_seconds": None, "output": None, "status": "running",
        }

    def select_job(self, state: RecruitmentGraphState) -> RecruitmentGraphState:
        """选取待执行 JD，并在调用前记录 running 和本次重试计数。"""
        execution = read_execution(state)
        if any(job.status == "running" for job in execution.jobs.values()):
            raise RuntimeError("仍有执行中的 JD，不能重新调度；应从 checkpoint 的下一节点恢复")
        runnable = execution.runnable_jd_ids
        if not runnable:
            return {"current_jd_id": None, "pending_report": None, "retry_after_seconds": None}
        jd_id = runnable[0]
        execution = execution.start_job(jd_id)
        logger.info("Agent图开始分析 jd_id=%s attempt=%d", jd_id, execution.jobs[jd_id].retry_count + 1)
        return {
            "execution": execution.model_dump(mode="json"), "current_jd_id": jd_id,
            "pending_report": None, "retry_after_seconds": None,
        }

    async def analyze_job(self, state: RecruitmentGraphState) -> RecruitmentGraphState:
        """每次调用一次客户端；已知错误写入状态，程序错误和取消异常原样抛出。"""
        execution, jd_id = self._current_job(state, "running")
        try:
            report = await self.llm.analyze_job(execution.agent_input, jd_id=jd_id)
        except AgentLLMError as exc:
            return self._record_failure(
                execution, jd_id, stage=exc.stage, code=exc.code,
                message=str(exc), retryable=exc.retryable,
                retry_after_seconds=exc.retry_after_seconds,
            )
        # 客户端契约被破坏属于代码错误，不把异常返回值伪装成可重试模型错误。
        if not isinstance(report, JobAnalysis):
            raise TypeError("JobAnalyzer.analyze_job 必须返回 JobAnalysis")
        return {"pending_report": report.model_dump(mode="json"), "retry_after_seconds": None}

    def validate_result(self, state: RecruitmentGraphState) -> RecruitmentGraphState:
        """独立校验节点：重新检查结构、评分、岗位绑定和证据后才保存成功报告。

        现有 llm_client 也会校验；这里作为状态写入边界，兼容其他 JobAnalyzer 实现。
        """
        execution, jd_id = self._current_job(state, "running")
        try:
            report = JobAnalysis.model_validate_json(
                json.dumps(state.get("pending_report"), ensure_ascii=False, allow_nan=False)
            )
            validate_scoring_rules(report)
            execution = execution.complete_job(jd_id, report)
        except (ValueError, TypeError):
            return self._record_failure(
                execution, jd_id, stage="output_validate", code="result_validation_failed",
                message="报告未通过结构、评分、岗位绑定或证据校验", retryable=True,
            )
        return {
            "execution": execution.model_dump(mode="json"),
            "pending_report": None, "retry_after_seconds": None,
        }

    async def wait_retry(self, state: RecruitmentGraphState) -> RecruitmentGraphState:
        """等待后重新进入 select_job；重试计数仅在 start_job 时增加。"""
        execution, jd_id = self._current_job(state, "retry_pending")
        # 第一次失败等待 base，第二次等待 base*2，之后受 max_delay 限制。
        delay = self.retry_base_delay
        for _ in range(execution.jobs[jd_id].retry_count):
            delay = min(self.retry_max_delay, delay * 2)
            if delay >= self.retry_max_delay or delay == 0:
                break
        hint = state.get("retry_after_seconds")
        if hint is not None:
            if not isinstance(hint, (int, float)) or not math.isfinite(hint) or hint < 0:
                raise ValueError("retry_after_seconds 必须为非负有限数值")
            # 服务端指定的等待时间可大于本地指数退避上限，不提前重试限流请求。
            delay = max(delay, hint)
        logger.info("Agent图等待重试 jd_id=%s delay_seconds=%.2f", jd_id, delay)
        if delay:
            await asyncio.sleep(delay)
        return {"retry_after_seconds": None}

    def finalize(self, state: RecruitmentGraphState) -> RecruitmentGraphState:
        """所有 JD 终止后返回；部分失败时通过 execution.jobs 读取已成功报告。"""
        execution = read_execution(state)
        if not execution.is_finished:
            raise RuntimeError("仍有未到终态的 JD，不能结束分析")
        succeeded = sum(job.status == "succeeded" for job in execution.jobs.values())
        if succeeded == len(execution.jobs):
            output = execution.build_output().model_dump(mode="json")
            status = "succeeded"
        else:
            output = None
            status = "partial_failed" if succeeded else "failed"
        logger.info("Agent图结束 status=%s succeeded=%d total=%d", status, succeeded, len(execution.jobs))
        return {
            "status": status, "output": output, "current_jd_id": None,
            "pending_report": None, "retry_after_seconds": None,
        }

    @staticmethod
    def route_selected(state: RecruitmentGraphState) -> Literal["analyze_job", "finalize"]:
        return "finalize" if state.get("current_jd_id") is None else "analyze_job"

    @staticmethod
    def route_analyzed(state: RecruitmentGraphState) -> Literal["validate_result", "wait_retry", "select_job"]:
        if state.get("pending_report") is not None:
            return "validate_result"
        return RecruitmentNodes.route_result(state)

    @staticmethod
    def route_result(state: RecruitmentGraphState) -> Literal["wait_retry", "select_job"]:
        execution = read_execution(state)
        jd_id = state["current_jd_id"]
        status = execution.jobs[jd_id].status
        if status == "retry_pending":
            return "wait_retry"
        if status in {"succeeded", "failed"}:
            return "select_job"
        raise RuntimeError("分析节点没有产生报告或有效执行结果")

    @staticmethod
    def _current_job(
        state: RecruitmentGraphState, expected_status: str,
    ) -> tuple[RecruitmentAgentState, int]:
        execution = read_execution(state)
        jd_id = state.get("current_jd_id")
        if type(jd_id) is not int or jd_id not in execution.jobs:
            raise RuntimeError("缺少有效的当前 JD")
        if execution.jobs[jd_id].status != expected_status:
            raise RuntimeError(f"当前 JD 不处于 {expected_status} 状态")
        return execution, jd_id

    @staticmethod
    def _record_failure(
        execution: RecruitmentAgentState, jd_id: int, *, stage: str, code: str,
        message: str, retryable: bool, retry_after_seconds: float | None = None,
    ) -> RecruitmentGraphState:
        execution = execution.fail_job(
            jd_id, stage=stage, error_type=code, message=message, retryable=retryable,
        )
        logger.warning("Agent图记录错误 jd_id=%s code=%s status=%s", jd_id, code, execution.jobs[jd_id].status)
        return {
            "execution": execution.model_dump(mode="json"), "pending_report": None,
            "retry_after_seconds": retry_after_seconds,
        }


__all__ = ["RecruitmentGraphState", "RecruitmentNodes", "JobAnalyzer", "read_execution"]
