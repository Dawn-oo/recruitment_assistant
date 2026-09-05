from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.services.resume_match.vector_match.recall.base import ResumeQueryType
from app.services.resume_match.vector_match.recall.job_candidate_aggregator import (
    SemanticJobMatchResult,
)


class TargetJobRerankStatus(str, Enum):
    """语义岗位精排后的处理状态。"""

    NEEDS_CONFIRMATION = "needs_confirmation"
    AUTO_RESOLVED = "auto_resolved"
    NO_ELIGIBLE_CANDIDATE = "no_eligible_candidate"
    NO_CANDIDATE = "no_candidate"


class ResumeSupportEvidence(BaseModel):
    """一条简历 Query 在某个 JD 中找到的最佳重排证据。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    query_id: str = Field(min_length=1)
    query_type: ResumeQueryType
    source_index: int | None = None
    resume_evidence_text: str = Field(min_length=1)
    jd_section: str = Field(min_length=1)
    jd_passage: str = Field(min_length=1)
    rerank_score: float = Field(ge=0.0, le=1.0)


class ResumeSupportTypeScore(BaseModel):
    """某类简历事实对 JD 的重排聚合结果。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    query_type: ResumeQueryType
    query_count: int = Field(ge=1)
    normalized_weight: float = Field(gt=0.0, le=1.0)
    mean_best_score: float = Field(ge=0.0, le=1.0)
    weighted_contribution: float = Field(ge=0.0, le=1.0)


class TargetJobRerankCandidate(BaseModel):
    """一个粗召回 JD 经过 BGE-M3 精排后的三分数结果。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    rerank_rank: int = Field(ge=1)
    recall_rank: int = Field(ge=1)
    jd_id: int
    job_title: str = Field(min_length=1)
    department: str | None = None

    # CandidateAggregator.aggregate_score；限制到 [0, 1] 后参与最终组合。
    recall_score: float = Field(ge=0.0, le=1.0)
    title_similarity: float = Field(ge=0.0, le=1.0)
    resume_support_score: float = Field(ge=0.0, le=1.0)
    resolution_score: float = Field(ge=0.0, le=1.0)
    passes_title_threshold: bool

    support_type_scores: list[ResumeSupportTypeScore] = Field(
        default_factory=list
    )
    support_evidence: list[ResumeSupportEvidence] = Field(default_factory=list)
    semantic_candidate: SemanticJobMatchResult


class TargetJobRerankResult(BaseModel):
    """一个未精确命中申请岗位的最终精排结果。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: str = "target_job_rerank_v1"
    target_id: str = Field(min_length=1)
    requested_job_title: str = Field(min_length=1)
    reranker_model: str = Field(min_length=1)
    status: TargetJobRerankStatus

    recall_candidate_count: int = Field(ge=0)
    reranked_candidate_count: int = Field(ge=0)
    recommended_jd_id: int | None = None
    selected_jd_id: int | None = None
    needs_confirmation: bool
    top_margin: float | None = Field(default=None, ge=0.0, le=1.0)

    candidates: list[TargetJobRerankCandidate] = Field(default_factory=list)
    missing_jd_ids: list[int] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TargetJobRerankConfig(BaseModel):
    """目标岗位精排配置；阈值均应通过项目样本校准。"""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    rerank_top_n: int = Field(default=5, gt=0)
    title_weight: float = Field(default=0.60, ge=0.0, le=1.0)
    resume_support_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    recall_weight: float = Field(default=0.10, ge=0.0, le=1.0)

    # BGE-M3 sigmoid 分数不是余弦相似度，未完成业务样本校准前不启用绝对阈值。
    minimum_title_similarity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    auto_confirm_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    minimum_top_margin: float = Field(default=0.08, ge=0.0, le=1.0)
    require_manual_confirmation: bool = True

    max_jd_passages: int = Field(default=12, gt=0)
    max_support_evidence: int = Field(default=10, gt=0)
    resume_type_weights: dict[ResumeQueryType, float] = Field(
        default_factory=lambda: {
            ResumeQueryType.WORK_EXPERIENCE: 0.50,
            ResumeQueryType.PROJECT_EXPERIENCE: 0.30,
            ResumeQueryType.SKILLS: 0.20,
        }
    )

    @model_validator(mode="after")
    def validate_weights(self) -> "TargetJobRerankConfig":
        final_weight_sum = (
            self.title_weight
            + self.resume_support_weight
            + self.recall_weight
        )
        if not math.isclose(final_weight_sum, 1.0, abs_tol=1e-9):
            raise ValueError("title/resume_support/recall 权重之和必须为1")

        required_types = {
            ResumeQueryType.WORK_EXPERIENCE,
            ResumeQueryType.PROJECT_EXPERIENCE,
            ResumeQueryType.SKILLS,
        }
        if set(self.resume_type_weights) != required_types:
            raise ValueError("resume_type_weights 必须且只能配置工作、项目和技能")
        if any(
            not math.isfinite(weight) or weight <= 0.0
            for weight in self.resume_type_weights.values()
        ):
            raise ValueError("resume_type_weights 必须全部为正有限数")
        return self
