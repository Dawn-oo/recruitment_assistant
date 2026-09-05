"""把结构化简历和最终岗位匹配结果组装为 Agent 标准输入。"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TYPE_CHECKING

from app.services.middle_layer.models import (
    AgentAnalysisInput,
    AgentAnalysisScope,
    AgentEducation,
    AgentJobDescription,
    AgentJobQualification,
    AgentJobResponsibility,
    AgentLanguage,
    AgentMatchSource,
    AgentProjectExperience,
    AgentResumeContent,
    AgentTargetMatch,
    AgentWorkExperience,
)
if TYPE_CHECKING:
    from app.services.resume_handle.resume_extractor.resume_schema import ResumeModel
    from app.services.resume_match.ResumeMatchService import (
        ResumeMatchResult,
        TargetMatchResult,
    )


logger = logging.getLogger(__name__)


class AgentInputAssemblyError(RuntimeError):
    """当前匹配结果或数据库 JD 无法构造可靠的 Agent 输入。"""


class JDContextRepository(Protocol):
    """Assembler 实际依赖的最小 JD 查询接口。"""

    def find_by_ids(self, jd_ids: Sequence[int]) -> list[dict[str, Any]]:
        ...


class AgentInputAssembler:
    """从过程模型中提取最小、去重且可供 Agent 消费的业务事实。"""

    def __init__(self, repository: JDContextRepository) -> None:
        self._repository = repository

    def build(self,*,resume: ResumeModel,match_result: ResumeMatchResult) -> AgentAnalysisInput:
        """构造 Agent 输入，只消费状态为 RESOLVED 的目标岗位。

        ``NEEDS_CONFIRMATION``、``NEEDS_MANUAL_RESOLUTION``、
        ``RESEARCHING`` 和 ``NO_CANDIDATE`` 均不会进入 Agent。
        """
        resolved_targets = [
            target
            for target in match_result.targets
            if self._is_resolved(target)
        ]
        if not resolved_targets:
            raise AgentInputAssemblyError(
                "当前匹配结果中没有已确定的岗位，不能构造 Agent 输入"
            )

        selected_jd_ids = list(
            dict.fromkeys(
                int(target.selected_jd_id)
                for target in resolved_targets
                if target.selected_jd_id is not None
            )
        )
        rows = self._load_jd_rows(selected_jd_ids)
        matched_jds = [self._build_jd(rows[jd_id]) for jd_id in selected_jd_ids]

        has_pending_target = len(resolved_targets) != len(match_result.targets)
        result = AgentAnalysisInput(
            analysis_scope=(
                AgentAnalysisScope.PARTIAL
                if has_pending_target
                else AgentAnalysisScope.COMPLETE
            ),
            resume=self._build_resume(resume),
            matched_jds=matched_jds,
            target_matches=[
                self._build_target_match(target)
                for target in resolved_targets
            ],
        )
        logger.info(
            "Agent输入组装完成: scope=%s resolved_targets=%d unique_jds=%d",
            result.analysis_scope.value,
            len(result.target_matches),
            len(result.matched_jds),
        )
        return result

    def _load_jd_rows(self, jd_ids: Sequence[int]) -> dict[int, Mapping[str, Any]]:
        try:
            rows = self._repository.find_by_ids(jd_ids)
        except Exception as exc:
            logger.exception("批量加载 Agent 所需 JD 失败: jd_ids=%s", jd_ids)
            raise AgentInputAssemblyError("批量加载最终 JD 失败") from exc

        rows_by_id: dict[int, Mapping[str, Any]] = {}
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise AgentInputAssemblyError(f"JDRepository.find_by_ids() 第 {index} 条结果不是 Mapping")
            try:
                jd_id = int(row["id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise AgentInputAssemblyError(f"第 {index} 条 JD 缺少有效 id") from exc
            if jd_id in rows_by_id:
                raise AgentInputAssemblyError(f"数据库返回了重复 JD: jd_id={jd_id}")
            rows_by_id[jd_id] = row

        missing_jd_ids = [jd_id for jd_id in jd_ids if jd_id not in rows_by_id]
        if missing_jd_ids:
            raise AgentInputAssemblyError(f"最终选定的 JD 不存在或未能加载: jd_ids={missing_jd_ids}")
        return rows_by_id

    def _build_resume(self, resume: ResumeModel) -> AgentResumeContent:
        basic = resume.basic_info
        knowledge = resume.knowledge_and_experience
        other = resume.other_info
        return AgentResumeContent(
            candidate_name=self._clean_optional(basic.name),
            location=self._clean_optional(basic.location),
            declared_target_job_title=self._clean_optional(basic.target_job_title),
            educations=[
                AgentEducation(
                    school=self._clean_optional(item.school),
                    degree=self._clean_optional(item.degree),
                    major=self._clean_optional(item.major),
                    start_date_raw=self._clean_optional(item.start_date_raw),
                    end_date_raw=self._clean_optional(item.end_date_raw),
                    description=self._clean_texts(item.description),
                )
                for item in resume.education_background.educations
            ],
            skills=self._clean_texts(item.skill for item in knowledge.skills),
            work_experiences=[
                AgentWorkExperience(
                    company=self._clean_optional(item.company),
                    job_title=self._clean_optional(item.job_title),
                    department=self._clean_optional(item.department),
                    start_date_raw=self._clean_optional(item.start_date_raw),
                    end_date_raw=self._clean_optional(item.end_date_raw),
                    responsibilities=self._clean_texts(item.responsibilities),
                    achievements=self._clean_texts(item.achievements),
                )
                for item in knowledge.work_experiences
            ],
            project_experiences=[
                AgentProjectExperience(
                    project_name=self._clean_optional(item.project_name),
                    start_date_raw=self._clean_optional(item.start_date_raw),
                    end_date_raw=self._clean_optional(item.end_date_raw),
                    description=self._clean_texts(item.description),
                    responsibilities=self._clean_texts(item.responsibilities),
                    technologies=self._clean_texts(item.technologies),
                    achievements=self._clean_texts(item.achievements),
                )
                for item in knowledge.project_experiences
            ],
            certificates=self._clean_texts(item.name for item in other.certificates),
            awards=self._clean_texts(item.name for item in other.awards),
            languages=[
                AgentLanguage(
                    language=item.language.strip(),
                    level=self._clean_optional(item.level),
                )
                for item in other.languages
                if item.language.strip()
            ],
            publications=self._clean_texts(other.publications),
            self_evaluation=self._clean_texts(other.self_evaluation),
            other_information=self._clean_texts(other.other),
        )

    def _build_jd(self, row: Mapping[str, Any]) -> AgentJobDescription:
        jd_id = int(row["id"])
        job_title = self._required_text(row.get("job_title"), f"JD {jd_id}.job_title")
        responsibilities = row.get("responsibilities") or []
        if isinstance(responsibilities, (str, bytes)) or not isinstance(
            responsibilities, Sequence
        ):
            raise AgentInputAssemblyError(
                f"JD {jd_id}.responsibilities 必须是列表"
            )

        return AgentJobDescription(
            jd_id=jd_id,
            job_title=job_title,
            department=self._clean_optional(row.get("department")),
            responsibilities=[
                self._build_responsibility(jd_id, index, item)
                for index, item in enumerate(responsibilities)
            ],
            qualification=AgentJobQualification(
                minimum_education=self._clean_optional(row.get("minimum_education")),
                education_background=self._clean_optional(row.get("education_background")),
                work_experience_raw=self._clean_optional(row.get("work_experience_raw")),
                competencies=self._clean_texts(row.get("competencies") or []),
            ),
        )

    def _build_responsibility(
        self,
        jd_id: int,
        index: int,
        value: Any,
    ) -> AgentJobResponsibility:
        if not isinstance(value, Mapping):
            raise AgentInputAssemblyError(
                f"JD {jd_id}.responsibilities[{index}] 必须是对象"
            )
        return AgentJobResponsibility(
            sequence=self._optional_int(value.get("sequence")),
            description=self._required_text(
                value.get("description"),
                f"JD {jd_id}.responsibilities[{index}].description",
            ),
            tasks=self._clean_texts(value.get("tasks") or []),
            time_percentage=value.get("time_percentage"),
        )

    def _build_target_match(self, target: TargetMatchResult) -> AgentTargetMatch:
        if target.selected_jd_id is None:
            raise AgentInputAssemblyError(
                f"已确定岗位缺少 selected_jd_id: target_id={target.target_id}"
            )
        source = self._map_source(target.source, target.target_id)
        return AgentTargetMatch(
            target_id=target.target_id,
            requested_job_title=target.requested_job_title,
            selected_jd_id=int(target.selected_jd_id),
            source=source,
            human_confirmed=(
                source != AgentMatchSource.EXACT
                or len(target.candidates) > 1
            ),
        )

    @staticmethod
    def _is_resolved(target: TargetMatchResult) -> bool:
        return (
            getattr(target.status, "value", target.status) == "resolved"
            and target.selected_jd_id is not None
        )

    @staticmethod
    def _map_source(value: Any, target_id: str) -> AgentMatchSource:
        source_name = getattr(value, "name", None)
        mapping = {
            "EXACT": AgentMatchSource.EXACT,
            "SEMANTIC": AgentMatchSource.SEMANTIC,
            "MANUAL_SEARCH": AgentMatchSource.MANUAL_SEARCH,
        }
        source = mapping.get(source_name)
        if source is None:
            raise AgentInputAssemblyError(
                f"已确定岗位缺少有效来源: target_id={target_id}, source={value!r}"
            )
        return source

    @staticmethod
    def _clean_optional(value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @classmethod
    def _required_text(cls, value: Any, field_name: str) -> str:
        normalized = cls._clean_optional(value)
        if normalized is None:
            raise AgentInputAssemblyError(f"{field_name} 不能为空")
        return normalized

    @staticmethod
    def _clean_texts(values: Any) -> list[str]:
        if isinstance(values, (str, bytes)):
            values = [values]
        try:
            iterator = iter(values)
        except TypeError as exc:
            raise AgentInputAssemblyError("文本集合必须是可迭代对象") from exc
        return list(
            dict.fromkeys(
                normalized
                for value in iterator
                if value is not None and (normalized := str(value).strip())
            )
        )

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise AgentInputAssemblyError(
                f"职责 sequence 必须是整数或 None，实际为 {value!r}"
            ) from exc
