from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from app.services.job_match.jd_repository import JDRepository
from app.services.job_match.exact_match.job_alias import JOB_TITLE_ALIASES
from app.services.job_match.exact_match.job_intent_norm import JobIntentNormalizeResult


class ExactMatchType(str, Enum):
    """
    精确匹配命中方式。
    """
    DIRECT = "direct"
    ALIAS = "alias"

class ExactMatchStatus(str, Enum):
    """
    精确匹配命中状态。
    """
    MATCHED = "matched"
    UNRESOLVED = "unresolved"

class ExactMatchedJD(BaseModel):
    """
    具体匹配到了哪条公司 JD
    """
    jd_id: int
    job_title: str
    department: str | None = None
    jd: dict

class ExactIntentMatchResult(BaseModel):
    """
    这个申请岗位匹配得怎么样
    """
    raw_title: str
    normalized_title: str | None = None
    match_type: ExactMatchType | None = None
    status: ExactMatchStatus

    matched_jds: list[ExactMatchedJD] = Field(default_factory=list)

class ExactJobMatchResult(BaseModel):
    """
    匹配的结果合集，因为可能有多个岗位匹配结果
    """
    model_config = ConfigDict(strict=True,extra="ignore",)

    intent_results: list[ExactIntentMatchResult] = Field(default_factory=list)

class ExactJobMatcher:

    def __init__(self,repository: JDRepository) -> None:
        self._repository = repository

    def match(self,intent_result: JobIntentNormalizeResult) -> ExactJobMatchResult:

        results: list[ExactIntentMatchResult] = []

        for job_title in intent_result.job_titles:
            result = self._match_single(job_title)

            results.append(result)

        return ExactJobMatchResult(intent_results=results)

    def _match_single(self,title: str) -> ExactIntentMatchResult:

        # =====================================================
        # 1. 直接精确匹配
        # =====================================================
        direct_jds = self._repository.find_all_by_job_title(title)

        if direct_jds:
            return ExactIntentMatchResult(
                raw_title=title,
                normalized_title=title,
                match_type=ExactMatchType.DIRECT,
                status=ExactMatchStatus.MATCHED,
                matched_jds=self._build_matches(jd_result=direct_jds)
            )
        # =====================================================
        # 2. Alias 精确匹配
        # =====================================================
        alias_title = JOB_TITLE_ALIASES.get(title)

        if alias_title:
            alias_jds = self._repository.find_all_by_job_title(alias_title)

            if alias_jds:
                return ExactIntentMatchResult(
                    raw_title=title,
                    normalized_title=alias_title,
                    match_type=ExactMatchType.ALIAS,
                    status=ExactMatchStatus.MATCHED,
                    matched_jds=self._build_matches(jd_result=alias_jds)
                )

        # =====================================================
        # 3. Exact 分支未命中
        # =====================================================

        return ExactIntentMatchResult(
            raw_title=title,
            normalized_title=None,
            match_type=None,
            status=ExactMatchStatus.UNRESOLVED,
            matched_jds=[],
        )

    @staticmethod
    def _build_matches(*,jd_result: Sequence[dict[str, Any]]) -> list[ExactMatchedJD]:
        """
        Repository row -> ExactMatchedJD。
        """
        matches: list[ExactMatchedJD] = []

        for row in jd_result:
            matches.append(
                ExactMatchedJD(
                    jd_id=int(row["id"]),
                    job_title=str(row["job_title"]),
                    department=row.get("department"),
                    jd=dict(row)
                )
            )

        return matches
