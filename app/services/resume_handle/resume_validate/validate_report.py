from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from app.services.resume_handle.resume_extractor.resume_schema import ResumeModel


class IssueSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class IssueType(str, Enum):
    SCHEMA_ERROR = "schema_error"
    MISSING_FIELD = "missing_field"
    EMPTY_FIELD = "empty_field"
    INVALID_VALUE = "invalid_value"
    LOW_QUALITY = "low_quality"
    MISSING_EVIDENCE = "missing_evidence"


class ResumeQualityStatus(str, Enum):
    AUTO_APPROVED = "auto_approved"
    REVIEW_RECOMMENDED = "review_recommended"
    REVIEW_REQUIRED = "review_required"


class ResumeValidationIssue(BaseModel):

    model_config = ConfigDict(
        strict=True,
        extra="ignore",
    )

    field_path: str

    issue_type: IssueType

    severity: IssueSeverity

    message: str

    current_value: Any | None = None

    blocks_analysis: bool = False


class ResumeValidationReport(BaseModel):

    model_config = ConfigDict(
        strict=True,
        extra="ignore",
    )

    status: ResumeQualityStatus

    schema_valid: bool

    can_analyze: bool

    issues: list[ResumeValidationIssue] = Field(
        default_factory=list
    )


class ResumeValidator:

    def validate(
        self,
        raw_data: dict,
    ) -> tuple[
        ResumeModel | None,
        ResumeValidationReport,
    ]:

        issues: list[ResumeValidationIssue] = []

        # =====================================================
        # 1. Schema 校验
        # =====================================================

        try:
            resume = ResumeModel.model_validate(
                raw_data
            )

        except ValidationError as exc:

            issues.extend(self._convert_schema_errors(exc))

            report = ResumeValidationReport(
                status=ResumeQualityStatus.REVIEW_REQUIRED,
                schema_valid=False,
                can_analyze=False,
                issues=issues)

            return (None,report)

        # =====================================================
        # 2. 业务质量校验
        # =====================================================

        # 技能经验内容校验
        issues.extend(
            self._validate_resume_content(resume)
        )

        #
        issues.extend(
            self._validate_basic_info(resume)
        )

        issues.extend(
            self._validate_education(resume)
        )

        issues.extend(
            self._validate_work_experience(resume)
        )

        issues.extend(
            self._validate_projects(resume)
        )

        issues.extend(
            self._validate_evidence(resume)
        )

        # =====================================================
        # 3. 计算 Review 状态
        # =====================================================

        status, can_analyze = (
            self._resolve_status(issues)
        )


        return (
            resume,
            ResumeValidationReport(
                status=status,
                schema_valid=True,
                can_analyze=can_analyze,
                issues=issues,
            ),
        )

    # =========================================================
    # Schema Error
    # =========================================================

   # 把 Pydantic 错误转换成前端可理解的格式
    def _convert_schema_errors(
        self,
        exc: ValidationError,
    ) -> list[ResumeValidationIssue]:

        issues = []

        for error in exc.errors():

            issues.append(
                ResumeValidationIssue(
                    field_path=self._loc_to_path(
                        error["loc"]
                    ),
                    issue_type=IssueType.SCHEMA_ERROR,
                    severity=IssueSeverity.ERROR,
                    message=error["msg"],
                    current_value=error.get("input"),
                    blocks_analysis=True,
                )
            )

        return issues

    # 用于定位控件
    @staticmethod
    def _loc_to_path(loc: tuple) -> str:

        result = ""

        for item in loc:

            if isinstance(item, int):
                result += f"[{item}]"

            else:
                if result:
                    result += "."

                result += str(item)

        return result

    # =========================================================
    # Resume整体质量
    # =========================================================

    def _validate_resume_content(
        self,
        resume: ResumeModel,
    ) -> list[ResumeValidationIssue]:

        issues = []

        knowledge = (
            resume.knowledge_and_experience
        )

        has_content = any([
            resume.education_background.educations,
            knowledge.skills,
            knowledge.work_experiences,
            knowledge.project_experiences,
        ])

        if not has_content:
            issues.append(
                ResumeValidationIssue(
                    field_path="$",
                    issue_type=IssueType.LOW_QUALITY,
                    severity=IssueSeverity.ERROR,
                    message=(
                        "未提取到教育、技能、工作或项目经历，"
                        "可能存在简历解析或LLM抽取失败"
                    ),
                    blocks_analysis=True,
                )
            )

        return issues

    # =========================================================
    # BasicInfo
    # =========================================================

    def _validate_basic_info(
        self,
        resume: ResumeModel,
    ) -> list[ResumeValidationIssue]:

        issues = []

        basic = resume.basic_info

        if not basic.name:
            issues.append(
                ResumeValidationIssue(
                    field_path="basic_info.name",
                    issue_type=IssueType.MISSING_FIELD,
                    severity=IssueSeverity.WARNING,
                    message="未提取到姓名",
                    blocks_analysis=False,
                )
            )

        if not basic.phone and not basic.email:
            issues.append(
                ResumeValidationIssue(
                    field_path="basic_info",
                    issue_type=IssueType.MISSING_FIELD,
                    severity=IssueSeverity.WARNING,
                    message="手机号和邮箱均为空",
                    blocks_analysis=False,
                )
            )

        if not basic.target_job_title:
            issues.append(
                ResumeValidationIssue(
                    field_path="basic_info.target_job_title",
                    issue_type=IssueType.MISSING_FIELD,
                    severity=IssueSeverity.WARNING,
                    message="未提取到目标岗位标题",
                    blocks_analysis=False,
                )
            )

        return issues

    # =========================================================
    # Education
    # =========================================================

    def _validate_education(
        self,
        resume: ResumeModel,
    ) -> list[ResumeValidationIssue]:

        issues = []

        educations = (
            resume.education_background.educations
        )

        if not educations:
            issues.append(
                ResumeValidationIssue(
                    field_path=(
                        "education_background.educations"
                    ),
                    issue_type=IssueType.EMPTY_FIELD,
                    severity=IssueSeverity.WARNING,
                    message=(
                        "未提取到教育经历，"
                        "可能影响学历条件判断"
                    ),
                    blocks_analysis=False,
                )
            )

            return issues

        for index, education in enumerate(
            educations
        ):

            prefix = (
                "education_background."
                f"educations[{index}]"
            )

            if not education.school:
                issues.append(
                    ResumeValidationIssue(
                        field_path=f"{prefix}.school",
                        issue_type=IssueType.MISSING_FIELD,
                        severity=IssueSeverity.WARNING,
                        message="缺少学校名称",
                        blocks_analysis=False,
                    )
                )

            if not education.degree:
                issues.append(
                    ResumeValidationIssue(
                        field_path=f"{prefix}.degree",
                        issue_type=IssueType.MISSING_FIELD,
                        severity=IssueSeverity.WARNING,
                        message=(
                            "缺少学历或学位信息，"
                            "可能影响JD硬条件判断"
                        ),
                        blocks_analysis=False,
                    )
                )

        return issues

    # =========================================================
    # Work
    # =========================================================

    def _validate_work_experience(
        self,
        resume: ResumeModel,
    ) -> list[ResumeValidationIssue]:

        issues = []

        works = (
            resume
            .knowledge_and_experience
            .work_experiences
        )

        for index, work in enumerate(works):

            prefix = (
                "knowledge_and_experience."
                f"work_experiences[{index}]"
            )

            if not work.job_title:
                issues.append(
                    ResumeValidationIssue(
                        field_path=(
                            f"{prefix}.job_title"
                        ),
                        issue_type=(
                            IssueType.MISSING_FIELD
                        ),
                        severity=(
                            IssueSeverity.WARNING
                        ),
                        message="未提取到岗位名称",
                        blocks_analysis=False,
                    )
                )

            if (
                not work.start_date_raw
                or not work.end_date_raw
            ):
                issues.append(
                    ResumeValidationIssue(
                        field_path=prefix,
                        issue_type=(
                            IssueType.MISSING_FIELD
                        ),
                        severity=(
                            IssueSeverity.WARNING
                        ),
                        message=(
                            "工作时间信息不完整，"
                            "可能影响工作年限计算"
                        ),
                        blocks_analysis=False,
                    )
                )

        return issues

    # =========================================================
    # Project
    # =========================================================

    def _validate_projects(
        self,
        resume: ResumeModel,
    ) -> list[ResumeValidationIssue]:

        issues = []

        projects = (
            resume
            .knowledge_and_experience
            .project_experiences
        )

        for index, project in enumerate(
            projects
        ):

            has_content = any([
                project.project_name,
                project.description,
                project.responsibilities,
                project.technologies,
                project.achievements,
            ])

            if not has_content:

                issues.append(
                    ResumeValidationIssue(
                        field_path=(
                            "knowledge_and_experience."
                            f"project_experiences[{index}]"
                        ),
                        issue_type=(
                            IssueType.EMPTY_FIELD
                        ),
                        severity=(
                            IssueSeverity.WARNING
                        ),
                        message="项目经历为空",
                        blocks_analysis=False,
                    )
                )

        return issues

    # =========================================================
    # Evidence
    # =========================================================

    def _validate_evidence(
        self,
        resume: ResumeModel,
    ) -> list[ResumeValidationIssue]:

        issues = []

        for index, education in enumerate(
            resume.education_background.educations
        ):

            if not education.evidence:

                issues.append(
                    ResumeValidationIssue(
                        field_path=(
                            "education_background."
                            f"educations[{index}].evidence"
                        ),
                        issue_type=(
                            IssueType.MISSING_EVIDENCE
                        ),
                        severity=(
                            IssueSeverity.WARNING
                        ),
                        message="教育经历缺少原始文本证据",
                        blocks_analysis=False,
                    )
                )

        return issues

    # =========================================================
    # Status
    # =========================================================

    @staticmethod
    def _resolve_status(
        issues: list[ResumeValidationIssue],
    ) -> tuple[ResumeQualityStatus, bool]:

        if any(
            issue.blocks_analysis
            for issue in issues
        ):
            return (
                ResumeQualityStatus.REVIEW_REQUIRED,
                False,
            )

        if issues:
            return (
                ResumeQualityStatus.REVIEW_RECOMMENDED,
                True,
            )

        return (
            ResumeQualityStatus.AUTO_APPROVED,
            True,
        )