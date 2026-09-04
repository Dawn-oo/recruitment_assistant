from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from app.services.resume_match.vector_match.recall.base import ResumeQueryType
from app.services.resume_match.vector_match.recall.vector_retriever import (
    QueryRetrievalResult,
    RetrievedChunk,
    VectorRetrievalResult,
)


logger = logging.getLogger(__name__)


class CandidateAggregationError(ValueError):
    """候选岗位聚合阶段的输入或配置错误。"""


class CandidateQueryEvidence(BaseModel):
    """一个 Query 对某个 JD 的最佳 chunk 命中证据。"""

    model_config = ConfigDict(strict=True, extra="ignore")

    query_id: str
    query_type: ResumeQueryType
    raw_query_weight: float = Field(gt=0)
    effective_query_weight: float = Field(gt=0, le=1)
    best_chunk: RetrievedChunk
    weighted_similarity: float


class QueryTypeAggregation(BaseModel):
    """某种 QueryType 对一个 JD 提供的聚合统计。"""

    model_config = ConfigDict(strict=True, extra="ignore")

    query_type: ResumeQueryType
    normalized_type_weight: float = Field(gt=0, le=1)
    effective_type_weight: float = Field(gt=0, le=1)
    total_query_count: int = Field(ge=1)
    matched_query_count: int = Field(ge=0)
    query_coverage: float = Field(ge=0, le=1)
    best_similarity: float | None = None
    weighted_contribution: float


class SemanticJobMatchResult(BaseModel):
    """由目标岗位感知的多 Query、多 chunk 聚合得到的粗召回 JD。"""

    model_config = ConfigDict(strict=True, extra="ignore")

    jd_id: int
    title_query_similarity: float | None = None
    best_similarity: float
    overall_evidence_score: float
    aggregate_score: float
    total_query_count: int = Field(ge=1)
    matched_query_count: int = Field(ge=1)
    weighted_query_coverage: float = Field(ge=0, le=1)
    chunk_hit_count: int = Field(ge=1)
    type_aggregations: list[QueryTypeAggregation] = Field(default_factory=list)
    evidence: list[CandidateQueryEvidence] = Field(default_factory=list)


class CandidateAggregationResult(BaseModel):
    """针对一个申请岗位生成的 JD 粗召回排序结果。"""

    model_config = ConfigDict(strict=True, extra="ignore")

    aggregation_version: str = "target_aware_max_evidence_v2"
    target_id: str = Field(min_length=1)
    requested_job_title: str = Field(min_length=1)
    total_query_count: int = Field(ge=1)
    total_retrieved_chunk_count: int = Field(ge=0)
    total_usable_chunk_count: int = Field(ge=0)
    candidate_count_before_limit: int = Field(ge=0)
    top_n: int = Field(gt=0)
    candidates: list[SemanticJobMatchResult] = Field(default_factory=list)


@dataclass(slots=True, frozen=True)
class _QueryContext:
    """聚合过程中使用的单个 Query 权重上下文。"""

    query_type: ResumeQueryType
    raw_weight: float
    effective_weight: float


@dataclass(slots=True)
class _CandidateBucket:
    """按 jd_id 暂存 Query-JD 最佳 chunk 和命中次数。"""

    jd_id: int
    chunk_hit_count: int = 0
    best_chunk_by_query: dict[str, RetrievedChunk] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class _TypeWeightPlan:
    """本次目标岗位 Query 的类型标准化权重和动态惩罚。"""

    normalized_type_weights: dict[ResumeQueryType, float]
    effective_type_weights: dict[ResumeQueryType, float]
    penalty_factor: float


class CandidateAggregator:
    """把单个申请岗位的多 Query chunk 结果聚合成 JD 粗召回列表。

    ``aggregate_score`` 只用于重排模型之前的候选截断，不表示最终岗位确认置信度，也不表示候选人与 JD 的最终匹配分。
    """

    # 目标岗位占 40%；简历侧沿用原先 5:3:2 的相对比例并共享其余 60%。
    DEFAULT_QUERY_TYPE_WEIGHTS: Mapping[ResumeQueryType, float] = {
        ResumeQueryType.TARGET_JOB_TITLE: 0.40,
        ResumeQueryType.WORK_EXPERIENCE: 0.30,
        ResumeQueryType.PROJECT_EXPERIENCE: 0.18,
        ResumeQueryType.SKILLS: 0.12,
    }

    # 当前 Query 类型越单薄，证据部分的预算越低。最佳相似度项不受此影响。
    DEFAULT_ACTIVE_TYPE_PENALTIES: Mapping[int, float] = {
        1: 0.30,
        2: 0.70,
        3: 0.90,
        4: 1.00,
    }

    def __init__(
        self,
        *,
        query_type_weights: Mapping[ResumeQueryType, float] | None = None,
        best_score_weight: float = 0.75,
        minimum_similarity: float | None = None,
        evidence_limit: int | None = 10,
        active_type_penalties: Mapping[int, float] | None = None,
    ) -> None:
        """校验并保存粗召回聚合参数。"""
        self._query_type_weights = dict(
            query_type_weights or self.DEFAULT_QUERY_TYPE_WEIGHTS
        )
        self._active_type_penalties = dict(
            active_type_penalties or self.DEFAULT_ACTIVE_TYPE_PENALTIES
        )

        if not self._query_type_weights:
            raise CandidateAggregationError("query_type_weights 不能为空")

        for query_type, weight in self._query_type_weights.items():
            if not math.isfinite(weight) or weight <= 0:
                raise CandidateAggregationError(
                    "QueryType 权重必须是大于0的有限数值: "
                    f"query_type={query_type}, weight={weight}"
                )

        for active_count, penalty in self._active_type_penalties.items():
            if active_count <= 0:
                raise CandidateAggregationError("活跃 QueryType 数量必须大于0")
            if not math.isfinite(penalty) or not 0 < penalty <= 1:
                raise CandidateAggregationError(
                    "QueryType 惩罚系数必须位于 (0, 1]: "
                    f"active_count={active_count}, penalty={penalty}"
                )

        if not math.isfinite(best_score_weight) or not 0 <= best_score_weight <= 1:
            raise CandidateAggregationError("best_score_weight 必须位于 [0, 1]")
        if minimum_similarity is not None and not math.isfinite(minimum_similarity):
            raise CandidateAggregationError("minimum_similarity 必须是有限数值或 None")
        if evidence_limit is not None and evidence_limit <= 0:
            raise CandidateAggregationError("evidence_limit 必须大于0或为 None")

        self._best_score_weight = best_score_weight
        self._evidence_score_weight = 1.0 - best_score_weight
        self._minimum_similarity = minimum_similarity
        self._evidence_limit = evidence_limit

    def aggregate(self,retrieval_results: VectorRetrievalResult,*,top_n: int = 10) -> CandidateAggregationResult:
        """完成输入校验、Query-JD 去重、计分、排序和 TopN 截断。"""
        self._validate_retrieval_results(
            retrieval_results=retrieval_results,
            top_n=top_n)

        type_weight_plan = self._build_type_weight_plan(retrieval_results)
        query_contexts = self._build_query_contexts(
            retrieval_results=retrieval_results,
            type_weight_plan=type_weight_plan)

        buckets, retrieved_count, usable_count = self._collect_candidate_buckets(retrieval_results)

        candidates = [
            self._build_candidate(
                bucket=bucket,
                retrieval_results=retrieval_results,
                query_contexts=query_contexts,
                type_weight_plan=type_weight_plan,
            )
            for bucket in buckets.values()
        ]

        candidates.sort(
            key=lambda candidate: (
                candidate.aggregate_score,
                candidate.title_query_similarity
                if candidate.title_query_similarity is not None
                else -math.inf,
                candidate.weighted_query_coverage,
                candidate.best_similarity,
            ),
            reverse=True,
        )

        result = CandidateAggregationResult(
            target_id=retrieval_results.target_id,
            requested_job_title=retrieval_results.requested_job_title,
            total_query_count=len(retrieval_results.query_results),
            total_retrieved_chunk_count=retrieved_count,
            total_usable_chunk_count=usable_count,
            candidate_count_before_limit=len(candidates),
            top_n=top_n,
            candidates=candidates[:top_n],
        )
        logger.info(
            "目标岗位JD粗召回聚合完成: target_id=%s query_count=%d "
            "candidate_count=%d returned=%d",
            result.target_id,
            result.total_query_count,
            result.candidate_count_before_limit,
            len(result.candidates),
        )
        return result

    def _build_type_weight_plan(self,retrieval_results: VectorRetrievalResult) -> _TypeWeightPlan:
        """在活跃 QueryType 间标准化预算，并施加输入完整度惩罚。"""
        active_types = {
            result.query_type for result in retrieval_results.query_results
        }
        active_weight_sum = sum(
            self._query_type_weights[query_type] for query_type in active_types
        )
        if active_weight_sum <= 0:
            raise CandidateAggregationError("活跃 QueryType 总权重必须大于0")

        normalized = {
            query_type: self._query_type_weights[query_type] / active_weight_sum
            for query_type in active_types
        }
        active_count = len(active_types)
        try:
            penalty = self._active_type_penalties[active_count]
        except KeyError as exc:
            raise CandidateAggregationError(
                "没有配置对应的活跃 QueryType 惩罚系数: "
                f"active_type_count={active_count}"
            ) from exc

        effective = {
            query_type: normalized_weight * penalty
            for query_type, normalized_weight in normalized.items()
        }
        return _TypeWeightPlan(
            normalized_type_weights=normalized,
            effective_type_weights=effective,
            penalty_factor=penalty,
        )

    @staticmethod
    def _build_query_contexts(*,retrieval_results: VectorRetrievalResult,type_weight_plan: _TypeWeightPlan) -> dict[str, _QueryContext]:
        """让同类型多个 Query 共享该类型预算，消除经历数量偏置。"""
        raw_weight_sum_by_type: dict[ResumeQueryType, float] = defaultdict(float)
        for result in retrieval_results.query_results:
            raw_weight_sum_by_type[result.query_type] += result.weight

        contexts: dict[str, _QueryContext] = {}
        for result in retrieval_results.query_results:
            effective_weight = (
                type_weight_plan.effective_type_weights[result.query_type]
                * result.weight
                / raw_weight_sum_by_type[result.query_type]
            )
            contexts[result.query_id] = _QueryContext(
                query_type=result.query_type,
                raw_weight=result.weight,
                effective_weight=effective_weight,
            )
        return contexts

    def _collect_candidate_buckets(self,retrieval_results: VectorRetrievalResult) -> tuple[dict[int, _CandidateBucket], int, int]:
        """按 jd_id 汇总，并让每个 Query 对每个 JD 仅保留最高分 chunk。"""
        buckets: dict[int, _CandidateBucket] = {}
        retrieved_count = 0
        usable_count = 0

        for result in retrieval_results.query_results:
            retrieved_count += len(result.chunks)
            for chunk in result.chunks:
                if not self._is_usable_chunk(chunk):
                    continue
                usable_count += 1
                bucket = buckets.setdefault(
                    chunk.jd_id,
                    _CandidateBucket(jd_id=chunk.jd_id),
                )
                bucket.chunk_hit_count += 1
                current_best = bucket.best_chunk_by_query.get(result.query_id)
                if current_best is None or chunk.similarity > current_best.similarity:
                    bucket.best_chunk_by_query[result.query_id] = chunk

        return buckets, retrieved_count, usable_count

    def _build_candidate(self,*,bucket: _CandidateBucket,retrieval_results: VectorRetrievalResult,
        query_contexts: Mapping[str, _QueryContext],type_weight_plan: _TypeWeightPlan) -> SemanticJobMatchResult:
        """从一个 JD 聚合桶构造可解释的粗召回分和证据。"""
        all_evidence: list[CandidateQueryEvidence] = []
        for query_id, best_chunk in bucket.best_chunk_by_query.items():
            try:
                context = query_contexts[query_id]
            except KeyError as exc:
                raise CandidateAggregationError(
                    f"找不到 Query 权重上下文: query_id={query_id}"
                ) from exc

            all_evidence.append(
                CandidateQueryEvidence(
                    query_id=query_id,
                    query_type=context.query_type,
                    raw_query_weight=context.raw_weight,
                    effective_query_weight=context.effective_weight,
                    best_chunk=best_chunk,
                    weighted_similarity=best_chunk.similarity * context.effective_weight,
                )
            )

        if not all_evidence:
            raise CandidateAggregationError(
                f"候选 JD 不包含可用证据: jd_id={bucket.jd_id}"
            )

        all_evidence.sort(
            key=lambda evidence: (
                evidence.weighted_similarity,
                evidence.best_chunk.similarity,
            ),
            reverse=True,
        )

        best_similarity = max(evidence.best_chunk.similarity for evidence in all_evidence)

        evidence_score = sum(evidence.weighted_similarity for evidence in all_evidence)

        matched_effective_weight = sum(evidence.effective_query_weight for evidence in all_evidence)

        weighted_query_coverage = self._clamp_unit_interval(matched_effective_weight / type_weight_plan.penalty_factor)

        aggregate_score = (best_similarity * self._best_score_weight+ evidence_score * self._evidence_score_weight)

        title_evidence = [evidence for evidence in all_evidence
            if evidence.query_type is ResumeQueryType.TARGET_JOB_TITLE]

        title_query_similarity = (
            max(evidence.best_chunk.similarity for evidence in title_evidence)
            if title_evidence
            else None
        )
        type_aggregations = self._build_type_aggregations(
            retrieval_results=retrieval_results,
            all_evidence=all_evidence,
            type_weight_plan=type_weight_plan,
        )

        type_contribution_sum = sum(
            item.weighted_contribution for item in type_aggregations
        )
        if not math.isclose(
            evidence_score,
            type_contribution_sum,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise CandidateAggregationError(
                "岗位证据分数与类型贡献之和不一致: "
                f"jd_id={bucket.jd_id}"
            )

        visible_evidence = (
            all_evidence
            if self._evidence_limit is None
            else all_evidence[: self._evidence_limit]
        )
        return SemanticJobMatchResult(
            jd_id=bucket.jd_id,
            title_query_similarity=title_query_similarity,
            best_similarity=best_similarity,
            overall_evidence_score=evidence_score,
            aggregate_score=aggregate_score,
            total_query_count=len(retrieval_results.query_results),
            matched_query_count=len(all_evidence),
            weighted_query_coverage=weighted_query_coverage,
            chunk_hit_count=bucket.chunk_hit_count,
            type_aggregations=type_aggregations,
            evidence=visible_evidence,
        )

    @staticmethod
    def _build_type_aggregations(*,retrieval_results: VectorRetrievalResult,all_evidence: Sequence[CandidateQueryEvidence],type_weight_plan: _TypeWeightPlan) -> list[QueryTypeAggregation]:
        """按 QueryType 汇总已计算好的原子证据，用于解释和覆盖率展示。"""
        results_by_type: dict[ResumeQueryType, list[QueryRetrievalResult]] = (
            defaultdict(list)
        )
        evidence_by_type: dict[ResumeQueryType, list[CandidateQueryEvidence]] = (
            defaultdict(list)
        )
        for result in retrieval_results.query_results:
            results_by_type[result.query_type].append(result)
        for evidence in all_evidence:
            evidence_by_type[evidence.query_type].append(evidence)

        aggregations: list[QueryTypeAggregation] = []
        for query_type, type_results in results_by_type.items():
            type_evidence = evidence_by_type.get(query_type, [])
            effective_type_weight = type_weight_plan.effective_type_weights[query_type]
            matched_effective_weight = sum(
                evidence.effective_query_weight for evidence in type_evidence
            )
            coverage = CandidateAggregator._clamp_unit_interval(
                matched_effective_weight / effective_type_weight
            )
            best_similarity = (
                max(item.best_chunk.similarity for item in type_evidence)
                if type_evidence
                else None
            )
            aggregations.append(
                QueryTypeAggregation(
                    query_type=query_type,
                    normalized_type_weight=(
                        type_weight_plan.normalized_type_weights[query_type]
                    ),
                    effective_type_weight=effective_type_weight,
                    total_query_count=len(type_results),
                    matched_query_count=len(type_evidence),
                    query_coverage=coverage,
                    best_similarity=best_similarity,
                    weighted_contribution=sum(
                        item.weighted_similarity for item in type_evidence
                    ),
                )
            )

        aggregations.sort(
            key=lambda item: item.effective_type_weight,
            reverse=True,
        )
        return aggregations

    def _validate_retrieval_results(self,*,retrieval_results: VectorRetrievalResult,top_n: int) -> None:
        """校验单目标、唯一 Query、权重配置和相似度有限性。"""
        if not retrieval_results.query_results:
            raise CandidateAggregationError("query_results 不能为空")
        if top_n <= 0:
            raise CandidateAggregationError("top_n 必须大于0")

        seen_query_ids: set[str] = set()
        title_query_count = 0
        for result in retrieval_results.query_results:
            if result.target_id != retrieval_results.target_id:
                raise CandidateAggregationError("query_result.target_id 不一致")
            if result.requested_job_title != retrieval_results.requested_job_title:
                raise CandidateAggregationError(
                    "query_result.requested_job_title 不一致"
                )
            if not result.query_id.strip() or result.query_id in seen_query_ids:
                raise CandidateAggregationError(
                    f"query_id 不能为空且不能重复: {result.query_id!r}"
                )
            seen_query_ids.add(result.query_id)

            if result.query_type is ResumeQueryType.TARGET_JOB_TITLE:
                title_query_count += 1
            if result.query_type not in self._query_type_weights:
                raise CandidateAggregationError(
                    f"缺少 QueryType 权重配置: {result.query_type}"
                )
            if not math.isfinite(result.weight) or result.weight <= 0:
                raise CandidateAggregationError(
                    f"Query 权重必须为正有限数: query_id={result.query_id}"
                )
            for chunk in result.chunks:
                if not math.isfinite(chunk.similarity):
                    raise CandidateAggregationError(
                        "chunk similarity 必须是有限数值: "
                        f"query_id={result.query_id}, chunk_id={chunk.chunk_id}"
                    )

        if title_query_count != 1:
            raise CandidateAggregationError(
                "每个目标岗位必须且只能包含一个 TARGET_JOB_TITLE Query"
            )

    def _is_usable_chunk(self, chunk: RetrievedChunk) -> bool:
        """应用可选的粗召回最低相似度阈值。"""
        return (
            self._minimum_similarity is None
            or chunk.similarity >= self._minimum_similarity
        )

    @staticmethod
    def _clamp_unit_interval(value: float) -> float:
        """修正浮点误差并将结果限制在 [0, 1]。"""
        return min(1.0, max(0.0, value))
