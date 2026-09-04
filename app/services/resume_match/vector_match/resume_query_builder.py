from __future__ import annotations

from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.services.resume_handle.resume_extractor.resume_schema import ResumeModel
from app.services.resume_match.vector_match.base import ResumeQueryType


class ResumeQueryUnit(BaseModel):
    """一条面向某个申请岗位、可以独立嵌入的查询单元。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    target_id: str = Field(min_length=1,description="本次申请岗位在当前请求中的唯一标识")

    requested_job_title: str = Field(min_length=1,description="申请人原始申请岗位名称")

    query_id: str = Field(min_length=1,description="当前 target_id 内唯一的 Query 标识")

    query_type: ResumeQueryType

    text: str = Field(min_length=1,description="真正发送给嵌入模型的目标岗位感知文本")

    resume_evidence_text: str | None = Field(default=None,description="不含目标岗位前缀的简历原始事实；纯岗位标题 Query 为 None")

    source_index: int | None = Field(default=None,description="对应工作或项目经历在简历列表中的索引")
    weight: float = Field(default=1.0,gt=0,description="同一 QueryType 内部的相对权重")


class ResumeQueryBuildResult(BaseModel):
    """针对一个申请岗位构造出的完整多 Query 结果。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    target_id: str = Field(min_length=1)
    requested_job_title: str = Field(min_length=1)
    query_units: list[ResumeQueryUnit] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def has_query(self) -> bool:
        """是否至少构造出了纯岗位标题 Query。"""
        return bool(self.query_units)


class ResumeQueryBuilder:

    """使用申请岗位名称和简历事实构造目标岗位感知的多 Query。"""

    def __init__(self,*,max_chars_per_query: int = 1500,min_chars_per_query: int = 2) -> None:

        if max_chars_per_query <= 0:
            raise ValueError("max_chars_per_query 必须大于0")
        if min_chars_per_query <= 0:
            raise ValueError("min_chars_per_query 必须大于0")
        if min_chars_per_query > max_chars_per_query:
            raise ValueError("min_chars_per_query 不能大于 max_chars_per_query")

        self._max_chars_per_query = max_chars_per_query
        self._min_chars_per_query = min_chars_per_query

    def build(self,resume: ResumeModel,*,requested_job_title: str,target_id: str) -> ResumeQueryBuildResult:
        """为一个未精确命中的申请岗位构造标题及标题+简历 Query。"""
        title = self._clean_required_text(requested_job_title,field_name="requested_job_title")

        normalized_target_id = self._clean_required_text(target_id,field_name="target_id")

        query_units: list[ResumeQueryUnit] = [
            self._build_title_unit(
                target_id=normalized_target_id,
                requested_job_title=title,
            )
        ]
        warnings: list[str] = []

        work_units = self._build_work_units(
            resume,
            target_id=normalized_target_id,
            requested_job_title=title,
        )
        query_units.extend(work_units)
        if not work_units:
            warnings.append("未构造出工作经历检索单元")

        project_units = self._build_project_units(
            resume,
            target_id=normalized_target_id,
            requested_job_title=title,
        )
        query_units.extend(project_units)
        if not project_units:
            warnings.append("未构造出项目经历检索单元")

        skills_unit = self._build_skills_unit(
            resume,
            target_id=normalized_target_id,
            requested_job_title=title,
        )
        if skills_unit is not None:
            query_units.append(skills_unit)
        else:
            warnings.append("未构造出技能检索单元")

        query_units = self._deduplicate_units(query_units)

        if len(query_units) == 1:
            warnings.append("简历缺少可用经历或技能，本次仅使用申请岗位名称召回")

        return ResumeQueryBuildResult(
            target_id=normalized_target_id,
            requested_job_title=title,
            query_units=query_units,
            warnings=warnings,
        )

    def _build_title_unit(self,*,target_id: str,requested_job_title: str) -> ResumeQueryUnit:

        """构造独立岗位标题 Query，防止简历长文本淹没标题语义。"""

        return ResumeQueryUnit(
            target_id=target_id,
            requested_job_title=requested_job_title,
            query_id=f"{target_id}:title",
            query_type=ResumeQueryType.TARGET_JOB_TITLE,
            text=f"申请岗位：{requested_job_title}",
            resume_evidence_text=None,
            source_index=None,
            weight=1.0,
        )

    def _build_work_units(self,resume: ResumeModel,*,target_id: str,requested_job_title: str) -> list[ResumeQueryUnit]:

        """每条有效工作经历生成一个“岗位标题+工作经历”Query。"""

        units: list[ResumeQueryUnit] = []

        for index, work in enumerate(resume.knowledge_and_experience.work_experiences):
            parts: list[str] = []
            if work.job_title:
                parts.append(f"岗位：{self._clean_text(work.job_title)}")
            if work.department:
                parts.append(f"部门：{self._clean_text(work.department)}")

            responsibilities = self._clean_list(work.responsibilities)
            if responsibilities:
                parts.append("工作职责：" + "；".join(responsibilities))

            achievements = self._clean_list(work.achievements)
            if achievements:
                parts.append("工作成果：" + "；".join(achievements))

            evidence_text = self._join_parts(parts)
            unit = self._build_evidence_unit(
                target_id=target_id,
                requested_job_title=requested_job_title,
                query_id=f"{target_id}:work_{index}",
                query_type=ResumeQueryType.WORK_EXPERIENCE,
                evidence_label="候选人工作经历",
                evidence_text=evidence_text,
                source_index=index,
            )
            if unit is not None:
                units.append(unit)

        return units

    def _build_project_units(self,resume: ResumeModel,*,target_id: str,requested_job_title: str) -> list[ResumeQueryUnit]:
        """每条有效项目经历生成一个“岗位标题+项目经历”Query。"""
        units: list[ResumeQueryUnit] = []

        for index, project in enumerate(resume.knowledge_and_experience.project_experiences):
            parts: list[str] = []
            if project.project_name:
                parts.append("项目名称：" + self._clean_text(project.project_name))

            description = self._clean_list(project.description)
            if description:
                parts.append("项目描述：" + "；".join(description))

            responsibilities = self._clean_list(project.responsibilities)
            if responsibilities:
                parts.append("项目职责：" + "；".join(responsibilities))

            technologies = self._clean_list(project.technologies)
            if technologies:
                parts.append("项目技术：" + "；".join(technologies))

            achievements = self._clean_list(project.achievements)
            if achievements:
                parts.append("项目成果：" + "；".join(achievements))

            evidence_text = self._join_parts(parts)
            unit = self._build_evidence_unit(
                target_id=target_id,
                requested_job_title=requested_job_title,
                query_id=f"{target_id}:project_{index}",
                query_type=ResumeQueryType.PROJECT_EXPERIENCE,
                evidence_label="候选人项目经历",
                evidence_text=evidence_text,
                source_index=index,
            )
            if unit is not None:
                units.append(unit)

        return units

    def _build_skills_unit(self,resume: ResumeModel,*,target_id: str,requested_job_title: str) -> ResumeQueryUnit | None:
        """聚合短技能词，生成一个“岗位标题+技能”Query。"""
        skill_texts = self._deduplicate_strings(
            self._clean_text(skill.skill)
            for skill in resume.knowledge_and_experience.skills
            if skill.skill
        )
        evidence_text = "技能与能力：" + "；".join(skill_texts) if skill_texts else ""

        return self._build_evidence_unit(
            target_id=target_id,
            requested_job_title=requested_job_title,
            query_id=f"{target_id}:skills",
            query_type=ResumeQueryType.SKILLS,
            evidence_label="候选人技能",
            evidence_text=evidence_text,
            source_index=None,
        )

    def _build_evidence_unit(self,*,target_id: str,requested_job_title: str,query_id: str,query_type: ResumeQueryType,
        evidence_label: str,evidence_text: str,source_index: int | None) -> ResumeQueryUnit | None:
        """把一段简历事实与申请岗位组合成最终嵌入文本。"""
        cleaned_evidence = evidence_text.strip()
        if not self._is_valid_query(cleaned_evidence):
            return None

        query_text = self._truncate(
            f"申请岗位：{requested_job_title}\n{evidence_label}：{cleaned_evidence}"
        )

        return ResumeQueryUnit(
            target_id=target_id,
            requested_job_title=requested_job_title,
            query_id=query_id,
            query_type=query_type,
            text=query_text,
            resume_evidence_text=self._truncate(cleaned_evidence),
            source_index=source_index,
            weight=1.0,
        )

    @staticmethod
    def _clean_text(text: str) -> str:
        """折叠连续空白，但不改写原始事实。"""
        return " ".join(text.strip().split())

    @classmethod
    def _clean_required_text(cls, text: str, *, field_name: str) -> str:
        """清理并校验必填文本。"""
        if not isinstance(text, str):
            raise TypeError(f"{field_name} 必须是字符串")
        cleaned = cls._clean_text(text)
        if not cleaned:
            raise ValueError(f"{field_name} 不能为空")
        return cleaned

    @classmethod
    def _clean_list(cls, values: Sequence[str]) -> list[str]:
        """清理、过滤并保持原始顺序去重。"""
        return cls._deduplicate_strings(
            cls._clean_text(value)
            for value in values
            if value and value.strip()
        )

    @staticmethod
    def _deduplicate_strings(values: Iterable[str]) -> list[str]:
        """保持首次出现顺序对字符串迭代结果去重。"""
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            if value and value not in seen:
                seen.add(value)
                result.append(value)
        return result

    @staticmethod
    def _join_parts(parts: Sequence[str]) -> str:
        """使用中文分号连接非空事实字段。"""
        return "；".join(part for part in parts if part)

    def _is_valid_query(self, text: str) -> bool:
        """判断简历证据是否达到最小字符数。"""
        return len(text.strip()) >= self._min_chars_per_query

    def _truncate(self, text: str) -> str:
        """按配置限制单个 Query 的最大字符数。"""
        normalized = text.strip()
        if len(normalized) <= self._max_chars_per_query:
            return normalized
        return normalized[: self._max_chars_per_query].rstrip()

    @staticmethod
    def _deduplicate_units(units: Sequence[ResumeQueryUnit]) -> list[ResumeQueryUnit]:
        """按 Query 文本去重，并始终保留首次出现的 Query。"""
        result: list[ResumeQueryUnit] = []
        seen_texts: set[str] = set()
        for unit in units:
            if unit.text in seen_texts:
                continue
            seen_texts.add(unit.text)
            result.append(unit)
        return result
