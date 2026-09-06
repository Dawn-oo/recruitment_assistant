"""一份简历的 Agent 执行状态；每个已绑定JD独立分析、失败和重试。

标准输入只保存一次，单岗报告复用output_schema.JobAnalysis。
retry_count 不包含首次调用：max_retries=2 表示每个 JD 最多执行 3 次。
analysis_scope 是申请岗位的覆盖范围，不表示本轮执行是否成功。

状态变更方法返回新的状态，不原地修改传入对象，便于节点返回状态更新。
本模块不调用 LLM、不执行重试，也不负责持久化；调度与 checkpoint 由服务层负责。
使用 model_dump_json()/model_validate_json() 保存、恢复 JSON 快照。

用法::

    state = RecruitmentAgentState.from_input(agent_input, max_retries=2)
    state = state.start_job(jd_id)
    # 服务层调用 LLM，再解析 JobAnalysis；异常时调用 fail_job。
    state = state.complete_job(jd_id, report)
    # 全部 JD 成功后才能生成完整输出；部分成功报告始终保留在 jobs 中。
    output = state.build_output()
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.services.middle_layer.models import AgentAnalysisInput
from app.services.recruitment_agent.schema.output_schema import AgentAnalysisOutput, JobAnalysis


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
JobAnalysisStatus = Literal["pending", "running", "retry_pending", "succeeded", "failed"]


class _StateModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class AgentExecutionError(_StateModel):

    """一次执行的错误记录；message 应由服务层脱敏，避免保存请求密钥等信息。"""

    stage: NonEmptyText = Field(description="失败阶段，例如 llm_call/output_parse/output_validate")
    error_type: NonEmptyText = Field(description="异常类型或稳定的业务错误码")
    message: NonEmptyText
    attempt_number: int = Field(ge=1, description="首次执行为1，随后为 retry_count + 1")
    retryable: bool = Field(description="由服务层判断错误是否值得重试，不代表还有重试预算")
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class JDAnalysisState(_StateModel):
    """单个JD的执行记录；它不是候选人的录用/淘汰状态。"""

    jd_id: int
    status: JobAnalysisStatus = "pending"
    report: JobAnalysis | None = None
    errors: list[AgentExecutionError] = Field(default_factory=list)
    retry_count: int = Field(default=0, ge=0)

    @property
    def last_error(self) -> AgentExecutionError | None:
        return self.errors[-1] if self.errors else None

    @model_validator(mode="after")
    def validate_execution(self) -> JDAnalysisState:

        if (self.status == "succeeded") != (self.report is not None):
            raise ValueError("仅succeeded状态必须且可以持有report")

        if self.report is not None and self.report.jd_id != self.jd_id:
            raise ValueError("report.jd_id必须与执行记录的jd_id一致")

        if self.status == "pending" and (self.retry_count or self.errors):
            raise ValueError("pending状态不能包含已执行的重试或错误")

        expected_errors = self.retry_count + (1 if self.status in {"retry_pending", "failed"} else 0)

        if len(self.errors) != expected_errors:
            raise ValueError("错误数量与状态、retry_count不一致")

        if [error.attempt_number for error in self.errors] != list(range(1, len(self.errors) + 1)):
            raise ValueError("错误记录的attempt_number必须从1连续递增")

        # 只有可重试错误才能进入下一次执行。
        if any(not error.retryable for error in self.errors[:self.retry_count]):
            raise ValueError("不可重试错误之后不能继续执行")

        if self.status == "retry_pending" and not self.errors[-1].retryable:
            raise ValueError("retry_pending的最后一次错误必须可重试")
        return self


class RecruitmentAgentState(_StateModel):
    """一次 Agent 分析的状态快照。

    jobs以整数jd_id为键，多个target_id指向同一JD时只执行一次。
    请通过下方方法变更状态；frozen仅禁止字段重新赋值，不会冻结嵌套容器。
    并行节点不能直接覆盖整份jobs；并行合并策略应在graph层另行定义。
    """

    schema_version: Literal["recruitment_agent_state_v1"] = "recruitment_agent_state_v1"
    agent_input: AgentAnalysisInput
    jobs: dict[int, JDAnalysisState] = Field(min_length=1)
    max_retries: int = Field(default=2, ge=0)

    @classmethod
    def from_input(cls, agent_input: AgentAnalysisInput, *, max_retries: int = 3) -> RecruitmentAgentState:
        """按target_matches的首次出现顺序初始化，不分析未被申请岗位引用的JD。"""
        snapshot = AgentAnalysisInput.model_validate_json(agent_input.model_dump_json())
        jd_ids = dict.fromkeys(target.selected_jd_id for target in snapshot.target_matches)
        return cls(
            agent_input=snapshot,
            jobs={jd_id: JDAnalysisState(jd_id=jd_id) for jd_id in jd_ids},
            max_retries=max_retries,
        )

    @model_validator(mode="after")
    def validate_jobs(self) -> RecruitmentAgentState:
        expected_ids = {target.selected_jd_id for target in self.agent_input.target_matches}
        if set(self.jobs) != expected_ids:
            raise ValueError("jobs必须精确覆盖标准输入中已绑定的JD")
        for jd_id, job in self.jobs.items():
            if jd_id != job.jd_id:
                raise ValueError("jobs的键必须与job.jd_id一致")
            if job.retry_count > self.max_retries:
                raise ValueError("retry_count不能超过max_retries")
            if job.status == "retry_pending" and job.retry_count >= self.max_retries:
                raise ValueError("重试预算耗尽后必须进入failed")
            if (
                job.status == "failed"
                and job.last_error is not None
                and job.last_error.retryable
                and job.retry_count < self.max_retries
            ):
                raise ValueError("可重试且仍有预算的错误应进入 retry_pending")
            if job.report is not None:
                self._validate_report(jd_id, job.report)
        return self

    @property
    def runnable_jd_ids(self) -> list[int]:
        """服务层可调度的JD；running和终态不会被重复调度。"""
        return [
            jd_id for jd_id, job in self.jobs.items()
            if job.status in {"pending", "retry_pending"}
        ]

    @property
    def is_finished(self) -> bool:
        """所有JD均到达终态，包括失败；不等于全部分析成功。"""
        return all(job.status in {"succeeded", "failed"} for job in self.jobs.values())

    def start_job(self, jd_id: int) -> RecruitmentAgentState:
        """在发起一次调用前执行；仅重新执行时消耗一次重试预算。"""
        job = self._get_job(jd_id)
        if job.status not in {"pending", "retry_pending"}:
            raise ValueError(f"JD {jd_id} 处于 {job.status}，不能开始执行")
        return self._replace_job(
            jd_id,
            status="running",
            retry_count=job.retry_count + (1 if job.status == "retry_pending" else 0),
        )

    def complete_job(self, jd_id: int, report: JobAnalysis) -> RecruitmentAgentState:
        """保存通过结构、岗位绑定和证据校验的报告，保留历史错误。

        校验异常直接抛给服务层，原状态保持running；服务层可据此调用
        fail_job(stage="output_validate", ...) 并决定是否重试。
        """
        self._require_running(jd_id)
        checked = JobAnalysis.model_validate_json(report.model_dump_json())
        self._validate_report(jd_id, checked)
        return self._replace_job(jd_id, status="succeeded", report=checked)

    def fail_job(self,jd_id: int,*,stage: str,error_type: str,message: str,retryable: bool) -> RecruitmentAgentState:
        """记录本次失败；存在重试预算时进入retry_pending，否则进入failed。"""
        job = self._require_running(jd_id)
        error = AgentExecutionError(
            stage=stage,
            error_type=error_type,
            message=message,
            attempt_number=job.retry_count + 1,
            retryable=retryable,
        )
        status = "retry_pending" if retryable and job.retry_count < self.max_retries else "failed"
        return self._replace_job(jd_id, status=status, errors=[*job.errors, error])

    def build_output(self) -> AgentAnalysisOutput:
        """全部成功后汇总；不把部分执行成功伪装成完整的 AgentAnalysisOutput。"""
        if any(job.status != "succeeded" for job in self.jobs.values()):
            raise ValueError("尚有未成功的JD；可从jobs读取已完成报告和失败原因")
        output = AgentAnalysisOutput(
            analysis_scope=self.agent_input.analysis_scope,
            job_analyses=[job.report for job in self.jobs.values() if job.report is not None],
        )
        output = AgentAnalysisOutput.model_validate_json(output.model_dump_json())
        return output.validate_against_input(self.agent_input)

    def _get_job(self, jd_id: int) -> JDAnalysisState:
        if type(jd_id) is not int or jd_id not in self.jobs:
            raise ValueError(f"标准输入中不存在已绑定的JD: {jd_id!r}")
        return self.jobs[jd_id]

    def _require_running(self, jd_id: int) -> JDAnalysisState:
        job = self._get_job(jd_id)
        if job.status != "running":
            raise ValueError(f"JD{jd_id}处于{job.status}，不能保存执行结果")
        return job

    def _validate_report(self, jd_id: int, report: JobAnalysis) -> None:
        if report.jd_id != jd_id:
            raise ValueError("报告必须属于当前分析的JD")
        # 复用输出契约的校验，只截取当前JD的输入视图，不修改标准输入。
        job_input = self.agent_input.model_copy(update={
            "matched_jds": [job for job in self.agent_input.matched_jds if job.jd_id == jd_id],
            "target_matches": [
                target for target in self.agent_input.target_matches
                if target.selected_jd_id == jd_id
            ],
        })
        AgentAnalysisOutput(
            analysis_scope=self.agent_input.analysis_scope,
            job_analyses=[report],
        ).validate_against_input(job_input)

    def _replace_job(self, jd_id: int, **changes: object) -> RecruitmentAgentState:
        payload = self.model_dump(mode="python")
        payload["jobs"][jd_id].update(changes)
        # model_copy(update=...) 不会验证更新内容，状态变更必须重新走模型校验。
        return type(self).model_validate(payload)


