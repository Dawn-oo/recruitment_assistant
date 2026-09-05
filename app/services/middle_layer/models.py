from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _AgentInputModel(BaseModel):
    """Agent 输入模型的统一严格配置。"""

    model_config = ConfigDict(strict=True, extra="forbid")


class AgentAnalysisScope(str, Enum):
    """本次输入覆盖全部还是部分申请岗位。"""

    COMPLETE = "complete"
    PARTIAL = "partial"


class AgentMatchSource(str, Enum):
    """最终 JD 最初由哪条路径获得。"""

    EXACT = "exact"
    SEMANTIC = "semantic"
    MANUAL_SEARCH = "manual_search"


class AgentEducation(_AgentInputModel):
    school: str | None = None
    degree: str | None = None
    major: str | None = None
    start_date_raw: str | None = None
    end_date_raw: str | None = None
    description: list[str] = Field(default_factory=list)


class AgentWorkExperience(_AgentInputModel):
    company: str | None = None
    job_title: str | None = None
    department: str | None = None
    start_date_raw: str | None = None
    end_date_raw: str | None = None
    responsibilities: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)


class AgentProjectExperience(_AgentInputModel):
    project_name: str | None = None
    start_date_raw: str | None = None
    end_date_raw: str | None = None
    description: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)


class AgentLanguage(_AgentInputModel):
    language: str = Field(min_length=1)
    level: str | None = None


class AgentResumeContent(_AgentInputModel):
    """从ResumeModel提取的简历业务内容，不包含原文证据和联系方式。"""

    candidate_name: str | None = None
    location: str | None = None
    declared_target_job_title: str | None = None

    educations: list[AgentEducation] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    work_experiences: list[AgentWorkExperience] = Field(default_factory=list)
    project_experiences: list[AgentProjectExperience] = Field(default_factory=list)

    certificates: list[str] = Field(default_factory=list)
    awards: list[str] = Field(default_factory=list)
    languages: list[AgentLanguage] = Field(default_factory=list)
    publications: list[str] = Field(default_factory=list)
    self_evaluation: list[str] = Field(default_factory=list)
    other_information: list[str] = Field(default_factory=list)


class AgentJobResponsibility(_AgentInputModel):
    """一组 JD 职责；与 job_descriptions.responsibilities 的结构一致。"""

    sequence: int | None = None
    description: str = Field(min_length=1)
    tasks: list[str] = Field(default_factory=list)
    time_percentage: int | float | str | None = None


class AgentJobQualification(_AgentInputModel):
    minimum_education: str | None = None
    education_background: str | None = None
    work_experience_raw: str | None = None
    competencies: list[str] = Field(default_factory=list)


class AgentJobDescription(_AgentInputModel):
    """从数据库标准 JD 提取的分析字段。"""

    jd_id: int
    job_title: str = Field(min_length=1)
    department: str | None = None
    responsibilities: list[AgentJobResponsibility] = Field(default_factory=list)
    qualification: AgentJobQualification


class AgentTargetMatch(_AgentInputModel):
    """一个申请岗位与最终选定 JD 的轻量绑定关系。"""

    target_id: str = Field(min_length=1)
    requested_job_title: str = Field(min_length=1)
    selected_jd_id: int
    source: AgentMatchSource
    human_confirmed: bool


class AgentAnalysisInput(_AgentInputModel):
    """Agent 唯一接收的标准输入契约。"""

    schema_version: str = "agent_analysis_input_v1"
    analysis_scope: AgentAnalysisScope

    # 必须包含结构化简历内容，而且整份输入只保存一次。
    resume: AgentResumeContent

    # 只包含已经唯一确定或人工确认的JD，并按jd_id去重。
    matched_jds: list[AgentJobDescription] = Field(min_length=1)

    # 多个申请岗位可以引用同一份matched_jds，避免复制完整 JD。
    target_matches: list[AgentTargetMatch] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_references_and_uniqueness(self) -> "AgentAnalysisInput":
        jd_ids = [job.jd_id for job in self.matched_jds]
        if len(jd_ids) != len(set(jd_ids)):
            raise ValueError("matched_jds 中的 jd_id 不能重复")

        target_ids = [target.target_id for target in self.target_matches]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("target_matches 中的 target_id 不能重复")

        known_jd_ids = set(jd_ids)
        missing_references = {
            target.selected_jd_id
            for target in self.target_matches
            if target.selected_jd_id not in known_jd_ids
        }
        if missing_references:
            raise ValueError(
                "target_matches 引用了 matched_jds 中不存在的 JD: "
                f"{sorted(missing_references)}"
            )
        return self