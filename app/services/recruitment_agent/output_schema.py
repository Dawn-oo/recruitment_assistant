"""招聘 Agent 对一份简历、多个已确定 JD 的结构化分析结果。

与 middle_layer.models.AgentAnalysisInput 配套：每个JD只分析一次，
通过target_ids关联申请岗位。只表达分析与面试建议，不修改招聘状态。

服务层用法::

    output = AgentAnalysisOutput.model_validate_json(llm_response_text)
    output.validate_against_input(agent_input)
    payload = output.model_dump(mode="json")

第一步校验结构与结果内部约束；第二步必须由服务层调用，检查岗位绑定、
分析范围与证据引用。引用仅指向标准输入中的字符串，不代表PDF原文溯源；
引用存在也不能证明模型推理正确，分析结论仍需人工复核。
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.services.middle_layer.models import AgentAnalysisInput, AgentAnalysisScope


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
MatchStatus = Literal["matched", "partially_matched", "mismatched", "unknown", "not_applicable"]
AssessmentDimension = Literal["education", "major", "experience", "skills", "responsibilities"]


class _AgentOutputModel(BaseModel):
    """拒绝未知字段和隐式类型转换。"""

    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)


class AnalysisEvidence(_AgentOutputModel):
    """一条标准输入引用；JD 引用始终属于所在 JobAnalysis.jd_id。"""

    source: Literal["resume", "jd"]
    path: NonEmptyText = Field(
        description=(
            "相对 resume 或当前 JD 对象的 JSON Pointer，数组从 0 开始；"
            "例如 /skills/0、/work_experiences/0/achievements/0、"
            "/qualification/competencies/0。必须指向非空字符串字段。"
        )
    )
    quote: NonEmptyText = Field(
        description="指定字段中逐字存在的文本片段，不允许改写或虚构"
    )


class DimensionAssessment(_AgentOutputModel):
    """unknown 表示缺少判断依据，不能作为 mismatched 或零分处理。"""

    status: MatchStatus
    score: float | None = Field(
        ge=0, le=100,
        description=(
            "该维度的岗位适配分；unknown 和 not_applicable 必须为 null。"
            "这是分析评分，不是向量相似度或 rerank 分数。"
        ),
    )
    rationale: NonEmptyText = Field(description="基于 JD 要求和简历事实解释判断")
    evidence: list[AnalysisEvidence] = Field(
        description="引用当前 JD 要求及相关简历事实；无依据时可为空"
    )
    missing_information: list[NonEmptyText] = Field(
        description="需要候选人或面试官补充的信息；无缺失时返回空列表"
    )

    @model_validator(mode="after")
    def validate_assessment(self) -> DimensionAssessment:
        if self.status in {"unknown", "not_applicable"}:
            if self.score is not None:
                raise ValueError("unknown/not_applicable 维度的 score 必须为 None")
        else:
            if self.score is None:
                raise ValueError("已作出匹配判断的维度必须提供 score")
            if {item.source for item in self.evidence} != {"resume", "jd"}:
                raise ValueError("匹配判断必须同时引用简历事实和当前 JD 要求")
        if self.status == "unknown" and not self.missing_information:
            raise ValueError("unknown 维度必须说明缺少哪些判断信息")
        return self


class AssessmentDimensions(_AgentOutputModel):
    """固定五个维度，避免 LLM 漏项、重复输出或临时更改维度名称。"""

    education: DimensionAssessment
    major: DimensionAssessment
    experience: DimensionAssessment
    skills: DimensionAssessment
    responsibilities: DimensionAssessment


class AnalysisFinding(_AgentOutputModel):
    """有依据的优势或差距；缺失信息放入 missing_information。"""

    dimension: AssessmentDimension
    description: NonEmptyText
    evidence: list[AnalysisEvidence] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_evidence_sources(self) -> AnalysisFinding:
        if {item.source for item in self.evidence} != {"resume", "jd"}:
            raise ValueError("优势或差距必须同时引用简历事实和当前 JD 要求")
        return self


class InterviewQuestion(_AgentOutputModel):
    dimension: AssessmentDimension
    question: NonEmptyText = Field(description="围绕岗位要求、经历或待核实信息提出的问题")
    purpose: NonEmptyText = Field(description="该问题需要验证的能力或事实")
    evidence: list[AnalysisEvidence] = Field(
        min_length=1, description="问题依据，可引用岗位要求或简历相关经历"
    )
    evaluation_points: list[NonEmptyText] = Field(
        min_length=1, description="面试官应关注的回答要点，不是候选人已具备能力的断言"
    )


class JobAnalysis(_AgentOutputModel):
    """当前候选人对一个已选定 JD 的分析，不包含录用/淘汰状态写入指令。"""

    jd_id: int
    target_ids: list[NonEmptyText] = Field(
        min_length=1, description="输入中 selected_jd_id 指向本 JD 的全部 target_id"
    )
    summary: NonEmptyText = Field(description="岗位适配概述，应区分明确事实与待核实判断")
    overall_score: float | None = Field(
        ge=0, le=100,
        description=(
            "基于同一评分 rubric 的整体适配评分，不是各维度默认等权平均，"
            "也不是录用概率。依据不足以给出整体评分时为 null。"
            "跨候选人比较须由服务层固定 JD 与评分 rubric。"
        ),
    )
    overall_score_reason: NonEmptyText = Field(
        description="解释整体评分及不确定性；不评分时说明原因"
    )
    dimensions: AssessmentDimensions
    strengths: list[AnalysisFinding]
    gaps: list[AnalysisFinding] = Field(
        description="有证据支持的不匹配项；简历未提及的信息不能直接写成能力不足"
    )
    interview_questions: list[InterviewQuestion] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_job_analysis(self) -> JobAnalysis:
        if len(self.target_ids) != len(set(self.target_ids)):
            raise ValueError("同一 JD 的 target_ids 不能重复")
        assessments = [
            getattr(self.dimensions, name)
            for name in AssessmentDimensions.model_fields
        ]
        if self.overall_score is not None and all(
            item.score is None for item in assessments
        ):
            raise ValueError("所有维度均无法评分时，overall_score 必须为 None")
        return self


class AgentAnalysisOutput(_AgentOutputModel):
    """与 AgentAnalysisInput 对应的一份简历分析结果，不承担多候选人排名。"""

    schema_version: Literal["agent_analysis_output_v1"] = "agent_analysis_output_v1"
    analysis_scope: AgentAnalysisScope = Field(description="必须与输入的 analysis_scope 一致")
    job_analyses: list[JobAnalysis] = Field(
        min_length=1, description="按 JD 去重，覆盖输入 target_matches 中的全部已确定岗位"
    )

    @model_validator(mode="after")
    def validate_uniqueness(self) -> AgentAnalysisOutput:
        jd_ids = [job.jd_id for job in self.job_analyses]
        if len(jd_ids) != len(set(jd_ids)):
            raise ValueError("job_analyses 中的 jd_id 不能重复")
        target_ids = [target_id for job in self.job_analyses for target_id in job.target_ids]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("一个 target_id 只能出现在一份 JD 分析中")
        return self

    def validate_against_input(self, agent_input: AgentAnalysisInput) -> AgentAnalysisOutput:
        """服务层在保存/展示前调用；失败抛出 ValueError，可进入有限次修复重试。

        以输入中 target_matches 的绑定为准，禁止增加待确认岗位、遗漏已确定岗位、
        改绑 JD 或把 partial 分析标记为 complete。证据校验只验证引用存在性。
        """
        if self.analysis_scope != agent_input.analysis_scope:
            raise ValueError("输出 analysis_scope 必须与输入一致")

        expected = {
            target.target_id: target.selected_jd_id
            for target in agent_input.target_matches
        }
        actual = {
            target_id: job.jd_id
            for job in self.job_analyses
            for target_id in job.target_ids
        }
        if actual != expected:
            raise ValueError("输出的 target_id/JD 绑定必须完整且精确地对应输入 target_matches")

        resume = agent_input.resume.model_dump(mode="json")
        jobs = {job.jd_id: job.model_dump(mode="json") for job in agent_input.matched_jds}
        for job in self.job_analyses:
            evidence_items: list[AnalysisEvidence] = []
            for name in AssessmentDimensions.model_fields:
                evidence_items.extend(getattr(job.dimensions, name).evidence)
            for finding in [*job.strengths, *job.gaps]:
                evidence_items.extend(finding.evidence)
            for question in job.interview_questions:
                evidence_items.extend(question.evidence)
            for item in evidence_items:
                document = resume if item.source == "resume" else jobs[job.jd_id]
                value = _resolve_text_pointer(document, item.path)
                if item.quote not in value:
                    raise ValueError(
                        f"JD {job.jd_id}: 证据 quote 不存在于 {item.source}{item.path}"
                    )
        return self


def _resolve_text_pointer(document: dict, pointer: str) -> str:
    """解析相对对象根的 JSON Pointer，只允许引用字符串叶子，不执行表达式。"""
    if not pointer.startswith("/"):
        raise ValueError(f"证据 path 必须以 / 开始: {pointer}")
    value: object = document
    for raw_token in pointer[1:].split("/"):
        # JSON Pointer 只允许 ~0 和 ~1 两种转义，不能静默接受非法路径。
        remainder = raw_token.replace("~0", "").replace("~1", "")
        if "~" in remainder:
            raise ValueError(f"证据 path 包含非法转义: {pointer}")
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict) and token in value:
            value = value[token]
        elif isinstance(value, list) and (
            token == "0" or (token.isascii() and token.isdecimal() and not token.startswith("0"))
        ):
            index = int(token)
            if index >= len(value):
                raise ValueError(f"证据 path 的数组下标越界: {pointer}")
            value = value[index]
        else:
            raise ValueError(f"证据 path 不存在: {pointer}")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"证据 path 必须指向非空字符串: {pointer}")
    return value


__all__ = [
    "AgentAnalysisOutput",
    "JobAnalysis",
    "AssessmentDimensions",
    "DimensionAssessment",
    "AnalysisEvidence",
    "AnalysisFinding",
    "InterviewQuestion",
]
