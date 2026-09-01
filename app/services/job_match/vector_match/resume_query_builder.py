from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from app.services.resume_handle.resume_extractor.resume_schema import ResumeModel


class ResumeQueryType(str, Enum):
    WORK_EXPERIENCE = "work_experience"
    PROJECT_EXPERIENCE = "project_experience"
    SKILLS = "skills"


class ResumeQueryUnit(BaseModel):
    """
    一条可以独立做 embedding 的 Resume 查询单元。
    """

    model_config = ConfigDict(strict=True,extra="ignore",)

    query_id: str = Field(description="当前 Resume 内唯一的 QueryUnit 标识")

    query_type: ResumeQueryType = Field(description="QueryUnit 类型")

    text: str = Field(min_length=1,description="真正发送给 embedding 模型的文本")

    source_index: int | None = Field(default=None,
        description=(
            "该 QueryUnit 对应 Resume 列表中的原始索引；"
            "skills 这种聚合 QueryUnit 为 None"
        ),
    )

    weight: float = Field(default=1.0,gt=0,description="后续候选 JD 聚合时可使用的权重")


class ResumeQueryBuildResult(BaseModel):
    """
    QueryBuilder 的完整输出。

    query_units 为空时，上层 Workflow 可以触发降级策略，
    例如改用 MinerU Markdown 构造检索 Query。
    """

    model_config = ConfigDict(strict=True,extra="ignore")

    query_units: list[ResumeQueryUnit] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)

    @property
    def has_query(self) -> bool:
        return bool(self.query_units)


class ResumeQueryBuilder:
    """
    将结构化 ResumeModel 转换成多个独立的语义检索 QueryUnit。
    第一版：
        - 每条工作经历 -> 1 个向量
        - 每条项目经历 -> 1 个向量
        - 所有技能 -> 1 个向量
    只拼接 Resume 中已经抽取出的事实，不推断、不改写。
    """

    def __init__(self,*,max_chars_per_query: int = 1500,min_chars_per_query: int = 2) -> None:
        if max_chars_per_query <= 0:
            raise ValueError("max_chars_per_query 必须大于 0")

        if min_chars_per_query <= 0:
            raise ValueError("min_chars_per_query 必须大于 0")

        if min_chars_per_query > max_chars_per_query:
            raise ValueError("min_chars_per_query 不能大于 max_chars_per_query")

        self._max_chars_per_query = max_chars_per_query
        self._min_chars_per_query = min_chars_per_query

    def build(self,resume: ResumeModel) -> ResumeQueryBuildResult:

        query_units: list[ResumeQueryUnit] = []
        warnings: list[str] = []

        work_units = self._build_work_units(resume)
        query_units.extend(work_units)

        if not work_units:
            warnings.append("未构造出工作经历检索单元")

        project_units = self._build_project_units(resume)
        query_units.extend(project_units)

        if not project_units:
            warnings.append("未构造出项目经历检索单元")

        skills_unit = self._build_skills_unit(resume)

        if skills_unit is not None:
            query_units.append(skills_unit)
        else:
            warnings.append("未构造出技能检索单元")

        query_units = self._deduplicate_units(query_units)

        if not query_units:
            warnings.append("结构化 Resume 中没有足够的语义检索信息")

        return ResumeQueryBuildResult(query_units=query_units,warnings=warnings)

    def _build_work_units(self,resume: ResumeModel) -> list[ResumeQueryUnit]:

        units: list[ResumeQueryUnit] = []

        works = (
            resume
            .knowledge_and_experience
            .work_experiences
        )

        for index, work in enumerate(works):
            parts: list[str] = []

            if work.job_title:
                parts.append(f"岗位：{self._clean_text(work.job_title)}")

            if work.department:
                parts.append(f"部门：{self._clean_text(work.department)}")

            responsibilities = getattr(work,"responsibilities",[])

            cleaned_responsibilities = self._clean_list(responsibilities)

            if cleaned_responsibilities:
                parts.append("工作职责："+ "；".join(cleaned_responsibilities))

            cleaned_achievements = self._clean_list(work.achievements)

            if cleaned_achievements:
                parts.append( "工作成果："+ "；".join(cleaned_achievements))

            text = self._join_parts(parts)

            if not self._is_valid_query(text):
                continue

            units.append(
                ResumeQueryUnit(
                    query_id=f"work_{index}",
                    query_type=ResumeQueryType.WORK_EXPERIENCE,
                    text=self._truncate(text),
                    source_index=index,
                    weight=1.0,
                )
            )

        return units

    def _build_project_units(self,resume: ResumeModel) -> list[ResumeQueryUnit]:

        units: list[ResumeQueryUnit] = []

        projects = (
            resume
            .knowledge_and_experience
            .project_experiences
        )

        for index, project in enumerate(projects):
            parts: list[str] = []

            if project.project_name:
                parts.append("项目名称："+ self._clean_text(project.project_name))

            description = self._clean_list(project.description)

            if description:
                parts.append("项目描述："+ "；".join(description))

            responsibilities = self._clean_list(project.responsibilities)
            if responsibilities:
                parts.append("项目职责："+ "；".join(responsibilities))

            technologies = self._clean_list(project.technologies)
            if technologies:
                parts.append("项目技术："+ "；".join(technologies))

            achievements = self._clean_list(project.achievements)
            if achievements:
                parts.append("项目成果："+ "；".join(achievements))

            text = self._join_parts(parts)

            if not self._is_valid_query(text):
                continue

            units.append(
                ResumeQueryUnit(
                    query_id=f"project_{index}",
                    query_type=(
                        ResumeQueryType
                        .PROJECT_EXPERIENCE
                    ),
                    text=self._truncate(text),
                    source_index=index,
                    weight=1.0,
                )
            )

        return units

    def _build_skills_unit(self,resume: ResumeModel) -> ResumeQueryUnit | None:
        """
        技能不逐条 embedding。
        对短技能词做聚合，避免单词/短词 embedding 过于不稳定。
        """
        skills = (
            resume
            .knowledge_and_experience
            .skills
        )

        skill_texts = self._deduplicate_strings(
            [
                self._clean_text(skill.skill)
                for skill in skills
                if skill.skill
            ]
        )

        if not skill_texts:
            return None

        text = ("技能与能力："+ "；".join(skill_texts))

        if not self._is_valid_query(text):
            return None

        return ResumeQueryUnit(
            query_id="skills",
            query_type=ResumeQueryType.SKILLS,
            text=self._truncate(text),
            source_index=None,
            weight=1.0,
        )

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        只清理空白，不改变原始语义。
        """
        return " ".join(text.strip().split())

    @classmethod
    def _clean_list(cls,values: list[str]) -> list[str]:

        cleaned = [cls._clean_text(value) for value in values if value and value.strip()]

        return cls._deduplicate_strings(cleaned)

    @staticmethod
    def _deduplicate_strings(values: list[str]) -> list[str]:
        """
        对字符串列表进行去重，并过滤掉空字符串，同时保持剩余元素首次出现的相对顺序不变。
        """

        result: list[str] = []
        seen: set[str] = set()

        for value in values:
            if not value:
                continue

            if value in seen:
                continue

            seen.add(value)
            result.append(value)

        return result

    @staticmethod
    def _join_parts(parts: list[str]) -> str:
        return "；".join(part for part in parts if part)

    def _is_valid_query(self,text: str) -> bool:
        return len(text.strip()) >= self._min_chars_per_query

    def _truncate(self,text: str) -> str:
        """
        截断文本，确保不超过最大字符数。
        """

        text = text.strip()

        if len(text) <= self._max_chars_per_query:
            return text

        return text[: self._max_chars_per_query].rstrip()

    @staticmethod
    def _deduplicate_units(units: list[ResumeQueryUnit]) -> list[ResumeQueryUnit]:

        result: list[ResumeQueryUnit] = []
        seen_texts: set[str] = set()

        for unit in units:
            if unit.text in seen_texts:
                continue

            seen_texts.add(unit.text)
            result.append(unit)

        return result
