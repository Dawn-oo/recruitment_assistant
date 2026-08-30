from pydantic import BaseModel, ConfigDict, Field



class SourceEvidenceModel(BaseModel):
    """
    用于记录该条结构化信息来自简历的什么位置。
    第一版如果 document_parser 暂时拿不到 page，
    page_number 可以为 None。
    """

    model_config = ConfigDict(
        strict=True,
        extra="ignore",
    )

    page_number: int | None = Field(
        default=None,
        description="信息所在PDF页码"
    )

    source_text: str = Field(
        description="该信息对应的简历原始文本，不允许改写"
    )

# ============================================================
# 一、基础信息
# ============================================================

class BasicInfoModel(BaseModel):

    model_config = ConfigDict(
        strict=True,
        extra="ignore",
    )

    name: str | None = Field(
        default=None,
        description="姓名，保持简历原文"
    )

    phone: str | None = Field(
        default=None,
        description="手机号或联系电话，保持原文"
    )

    email: str | None = Field(
        default=None,
        description="邮箱，保持原文"
    )

    location: str | None = Field(
        default=None,
        description="简历明确填写的所在地或求职地点，保持原文"
    )

    target_job_title: str | None = Field(
        default=None,
        description="简历中明确填写的求职岗位或目标职位，不允许推断"
    )

    personal_website: str | None = Field(
        default=None,
        description="个人网站、博客等链接"
    )

    github: str | None = Field(
        default=None,
        description="GitHub地址"
    )

    other_links: list[str] = Field(
        default_factory=list,
        description="其他明确出现的个人主页或作品链接"
    )

# ============================================================
# 二、教育背景
# ============================================================

class EducationExperienceModel(BaseModel):

    model_config = ConfigDict(
        strict=True,
        extra="ignore",
    )

    school: str | None = Field(
        default=None,
        description="学校名称，保持原文"
    )

    degree: str | None = Field(
        default=None,
        description="学历或学位，如本科、硕士等，保持原文"
    )

    major: str | None = Field(
        default=None,
        description="专业名称，保持原文"
    )

    start_date_raw: str | None = Field(
        default=None,
        description="教育开始时间原文，不做日期格式转换"
    )

    end_date_raw: str | None = Field(
        default=None,
        description="教育结束时间原文，不做日期格式转换"
    )

    description: list[str] = Field(
        default_factory=list,
        description="教育经历下明确出现的其他描述，保持原文"
    )

    evidence: list[SourceEvidenceModel] = Field(
        default_factory=list,
        description="该教育经历对应的原始文本证据"
    )


class EducationBackgroundModel(BaseModel):

    model_config = ConfigDict(
        strict=True,
        extra="ignore",
    )

    educations: list[EducationExperienceModel] = Field(
        default_factory=list,
        description="教育经历，按照简历中的顺序保存"
    )

# ============================================================
# 三、知识技能与经验
# ============================================================

# ------------------------------------------------------------
# 技能
# ------------------------------------------------------------

class SkillModel(BaseModel):

    model_config = ConfigDict(
        strict=True,
        extra="ignore",
    )

    skill: str = Field(
        description="简历中明确出现的技能或能力描述，保持原文"
    )

    evidence: list[SourceEvidenceModel] = Field(
        default_factory=list,
        description="技能对应的原始文本证据"
    )

# ------------------------------------------------------------
# 工作经历
# ------------------------------------------------------------

class WorkExperienceModel(BaseModel):

    model_config = ConfigDict(
        strict=True,
        extra="ignore",
    )

    company: str | None = Field(
        default=None,
        description="公司或组织名称，保持原文"
    )

    job_title: str | None = Field(
        default=None,
        description="岗位名称，保持原文，不做岗位标准化"
    )

    department: str | None = Field(
        default=None,
        description="简历明确出现的部门名称；没有则为None"
    )

    start_date_raw: str | None = Field(
        default=None,
        description="工作开始时间原文"
    )

    end_date_raw: str | None = Field(
        default=None,
        description="工作结束时间原文，例如'至今'"
    )

    responsibilities: list[str] = Field(
        default_factory=list,
        description="该岗位下明确描述的工作职责、工作内容，保持原文，不允许总结"
    )

    achievements: list[str] = Field(
        default_factory=list,
        description="该岗位下明确描述的工作成果、业绩或量化结果，保持原文，不允许自行总结"
    )

    evidence: list[SourceEvidenceModel] = Field(
        default_factory=list,
        description="该工作经历对应的原始文本证据"
    )

# ------------------------------------------------------------
# 项目经历
# ------------------------------------------------------------

class ProjectExperienceModel(BaseModel):

    model_config = ConfigDict(
        strict=True,
        extra="ignore",
    )

    project_name: str | None = Field(
        default=None,
        description="项目名称，保持原文"
    )

    start_date_raw: str | None = Field(
        default=None,
        description="项目开始时间原文"
    )

    end_date_raw: str | None = Field(
        default=None,
        description="项目结束时间原文"
    )

    description: list[str] = Field(
        default_factory=list,
        description="项目背景或项目描述，保持原文"
    )

    responsibilities: list[str] = Field(
        default_factory=list,
        description="个人在项目中的具体职责，保持原文"
    )

    technologies: list[str] = Field(
        default_factory=list,
        description="项目中明确出现的技术、框架、工具，不允许根据常识补充"
    )

    achievements: list[str] = Field(
        default_factory=list,
        description="项目中明确描述的成果或指标，不允许自行总结"
    )

    evidence: list[SourceEvidenceModel] = Field(
        default_factory=list,
        description="该项目经历对应的原始文本证据"
    )

# ------------------------------------------------------------
# 知识技能与经验总结构
# ------------------------------------------------------------

class KnowledgeAndExperienceModel(BaseModel):

    model_config = ConfigDict(
        strict=True,
        extra="ignore",
    )

    skills: list[SkillModel] = Field(
        default_factory=list,
        description="简历明确列出的知识、技能或能力"
    )

    work_experiences: list[WorkExperienceModel] = Field(
        default_factory=list,
        description="工作经历，按照简历顺序保存"
    )

    project_experiences: list[ProjectExperienceModel] = Field(
        default_factory=list,
        description="简历中明确列出的项目经历，按照简历中的顺序保存，包括工作、学校、科研、个人等项目，按照简历顺序保存"
    )

# ============================================================
# 四、其他信息
# ============================================================

class CertificateModel(BaseModel):

    model_config = ConfigDict(
        strict=True,
        extra="ignore",
    )

    name: str = Field(
        description="证书名称，保持原文"
    )

    date_raw: str | None = Field(
        default=None,
        description="证书获得时间原文"
    )

    evidence: list[SourceEvidenceModel] = Field(
        default_factory=list
    )

class AwardModel(BaseModel):

    model_config = ConfigDict(
        strict=True,
        extra="ignore",
    )

    name: str = Field(
        description="奖项或荣誉名称，保持原文"
    )

    date_raw: str | None = Field(
        default=None,
        description="获得时间原文"
    )

    evidence: list[SourceEvidenceModel] = Field(
        default_factory=list
    )

class LanguageModel(BaseModel):

    model_config = ConfigDict(
        strict=True,
        extra="ignore",
    )

    language: str = Field(
        description="语言名称，保持原文"
    )

    level: str | None = Field(
        default=None,
        description="只有简历明确描述语言水平时才填写"
    )

    evidence: list[SourceEvidenceModel] = Field(
        default_factory=list
    )

class OtherInfoModel(BaseModel):

    model_config = ConfigDict(
        strict=True,
        extra="ignore",
    )

    certificates: list[CertificateModel] = Field(
        default_factory=list,
        description="证书信息"
    )

    awards: list[AwardModel] = Field(
        default_factory=list,
        description="奖项、荣誉信息"
    )

    languages: list[LanguageModel] = Field(
        default_factory=list,
        description="语言能力"
    )

    publications: list[str] = Field(
        default_factory=list,
        description="论文、专利、出版物等，保持原文"
    )

    self_evaluation: list[str] = Field(
        default_factory=list,
        description="自我评价或个人总结，保持原文，不做总结改写"
    )

    other: list[str] = Field(
        default_factory=list,
        description="无法归入以上字段但可能与岗位相关的原始信息"
    )

# ============================================================
# Resume总Schema
# ============================================================

class ResumeModel(BaseModel):

    model_config = ConfigDict(
        strict=True,
        extra="ignore",
    )

    schema_version: int = Field(
        default=1,
        description="Resume Schema版本"
    )

    basic_info: BasicInfoModel = Field(
        default_factory=BasicInfoModel,
        description="基础信息"
    )

    education_background: EducationBackgroundModel = Field(
        default_factory=EducationBackgroundModel,
        description="教育背景"
    )

    knowledge_and_experience: KnowledgeAndExperienceModel = Field(
        default_factory=KnowledgeAndExperienceModel,
        description="知识技能与经验"
    )

    other_info: OtherInfoModel = Field(
        default_factory=OtherInfoModel,
        description="其他信息"
    )















