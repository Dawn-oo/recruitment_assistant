from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Mapping,Sequence

from pydantic import BaseModel, ConfigDict, Field

from .vector_retriever import QueryRetrievalResult, RetrievedChunk, VectorRetrievalResult
from app.services.resume_match.vector_match.base import ResumeQueryType


logger = logging.getLogger(__name__)

"""
 
jd聚合排序计分规则：
1、类型内部原始权重 
每种类型问题有原始权重，默认设置为1(VectorRetrievalResult.query_results.weight)，这个是针对三种类型内部而言的权重；它的用途是用于计算每种类型内部的权重比例；
以skills为例，如果说有两个，那么每个skills的权重就是为0.5；相当于是将两个skills的query看做同等重要，如果某一个skills更为重要，就可以增大它的权重，
这样skills对外部的整体贡献还是是1，而不会增加，相当于是做了归一化动作；默认为所有类型内部权重一致,相当于先求内部总分数然后求一个均值；

2、类型外部原始权重 (CandidateQueryEvidence.raw_query_weight)
这个是针对与每种类型问题外部的原始权重，计算的是每一种类型对于最后得分的贡献；默认为0.5、0.3、0.2；
同时考虑到如果某一种类型缺失造成的分数降低的问题和某一种类型占全部权重的动态类型；采用带惩罚的动态权重来对原始权重进行调整；
这里先将它归一化，然后再使用惩罚系数(主要是考虑到query全面性对简历匹配的影响)，以只有两种类型0.2,0.3为例，先归一化，0.2对应的就是0.4,0.3对应的就是0.6，然后乘以惩罚系数0.7；最后就是0.28和0.42；
标准和动态化以后就得到QueryTypeAggregation.query_coverage；再乘以类型的内部原始权重,就得到了实际上每个query计算证据分数的权重：CandidateQueryEvidence.effective_query_weight；
它计算出来的是每个类型下的每个问题的证据分数；得到的就是对于某jd所有问题下返回了该jd chunk的类；对于没有返回该证据的query就不会生成该类；

3、聚合得分计算(SemanticJobCandidate.aggregate_score)
从一个query的角度看，每一个jd的chunk就只保留与该query的相似度最高的chunk，以此暂为代表该jd的相似度；因此先从query的角度对chunk按照jd_id进行聚合，得到query与jd_id的对应关系；、
然后对所有的query执行该操作；就得到了一份简历经过内容提取后得到的各个query与它们各自对应的jd_id；
由于各个query下都有不同的jd_id,最后要求返回的就是jd，因此需要对他们进行一定的聚合并按照规则进行排序筛选出最为符合的jd，这就是aggregate_score；
它由两部分组成，按照α * max_similarity + (1-α) * total_evidence_score计算得到，α是超参数可以调整(CandidateAggregator._best_score_weight)，先调试默认为0.75；
max_similarity很好计算，就是对所有query的下同个jd按照jd的相似度进行排序，取最大值(SemanticJobCandidate.best_similarity)；
evidence_score计算则会相对繁琐

"""

class CandidateAggregationError(ValueError):
    """候选岗位聚合阶段的输入或配置错误。"""


class CandidateQueryEvidence(BaseModel):
    """一个 Resume Query 对某个 JD 的最佳 chunk 命中证据。"""

    model_config = ConfigDict(strict=True, extra="ignore")

    query_id: str
    query_type: ResumeQueryType

    # 问题类型外部原始权重。
    raw_query_weight: float = Field(gt=0)

    # 问题类型外部实际权重(注意这里是通过计算内部重要性乘以经过惩罚后的外部权重得到)。
    effective_query_weight: float = Field(gt=0, le=1)

    best_chunk: RetrievedChunk

    # best_chunk.similarity * effective_query_weight，算的是每一个Query-JD最佳命中chunk的相似度，一个类型query下的evidence_score
    weighted_similarity: float


class QueryTypeAggregation(BaseModel):
    """某种 QueryType 对一个 JD 提供的聚合统计。"""

    model_config = ConfigDict(strict=True, extra="ignore")

    query_type: ResumeQueryType

    # 在当前活跃类型中的相对权重。
    normalized_type_weight: float = Field(gt=0, le=1)

    # 乘以动态惩罚后的实际类型预算。
    effective_type_weight: float = Field(gt=0, le=1)

    total_query_count: int = Field(ge=1)
    matched_query_count: int = Field(ge=0)

    # 当前类型中，被该JD支持的Query权重占比。
    query_coverage: float = Field(ge=0, le=1)

    best_similarity: float | None = None

    # 直接对该类型原子证据的 weighted_similarity 求和。
    weighted_contribution: float


class SemanticJobMatchResult(BaseModel):
    """由多Query、多chunk聚合得到的岗位级语义候选。"""

    model_config = ConfigDict(strict=True, extra="ignore")

    jd_id: int

    # 当前JD在所有Query-JD最佳命中中的最高相似度。
    best_similarity: float

    # 全部Query按有效权重累加得到的证据分数。
    overall_evidence_score: float

    # best_similarity与evidence_score组合后的岗位排序分数。
    aggregate_score: float

    total_query_count: int = Field(ge=1)
    matched_query_count: int = Field(ge=1)

    # 被当前 JD 支持的 Query 有效权重之和。
    weighted_query_coverage: float = Field(ge=0, le=1)

    # 当前 JD 在可用召回 chunk 中出现的次数，仅用于诊断。
    chunk_hit_count: int = Field(ge=1)

    type_aggregations: list[QueryTypeAggregation] = Field(default_factory=list)

    # evidence_limit 只限制这里的输出条数，不影响实际分数计算。
    evidence: list[CandidateQueryEvidence] = Field(default_factory=list)


class CandidateAggregationResult(BaseModel):
    """一次语义候选岗位聚合的整体结果。"""

    model_config = ConfigDict(strict=True, extra="ignore")

    aggregation_version: str = "query_type_normalized_max_evidence_v1"

    total_query_count: int = Field(ge=1)
    total_retrieved_chunk_count: int = Field(ge=0)
    total_usable_chunk_count: int = Field(ge=0)

    candidate_count_before_limit: int = Field(ge=0)
    top_n: int = Field(gt=0)

    candidates: list[SemanticJobMatchResult] = Field(default_factory=list)


@dataclass(slots=True, frozen=True)
class _QueryContext:
    """聚合过程中使用的 Query 权重上下文。"""

    query_type: ResumeQueryType
    raw_weight: float
    effective_weight: float


@dataclass(slots=True)
class _CandidateBucket:
    """按jd_id暂存聚合过程中的中间状态。"""

    jd_id: int
    chunk_hit_count: int = 0
    # 它保存的就是每个Query对当前JD的相似度最高的chunk
    best_chunk_by_query: dict[str, RetrievedChunk] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class _TypeWeightPlan:
    """记录本次简历的类型标准化权重和动态惩罚结果。"""

    # 活跃类型标准化权重，总和始终为 1。
    normalized_type_weights: dict[ResumeQueryType, float]

    # normalized_type_weight * penalty_factor。
    effective_type_weights: dict[ResumeQueryType, float]

    # 根据活跃 QueryType 数量确定的输入完整度惩罚。
    penalty_factor: float


class CandidateAggregator:
    """
    将多个 Resume Query 的 chunk 召回结果聚合为岗位级候选。

    聚合规则：
    1. 同一个 Query 对同一个 JD 只保留相似度最高的 chunk；
    2. 按 ResumeQueryType 分配固定权重预算；
    3. 同类型内的多个 Query 共享该类型预算；
    4. 所有 Query 都参与 evidence_score 计算；
    5. 使用 best_similarity + evidence_score 形成最终排序分数；
    6. TopN 只用于最终岗位候选截断。
    """

    DEFAULT_QUERY_TYPE_WEIGHTS: Mapping[ResumeQueryType, float] = {
        ResumeQueryType.WORK_EXPERIENCE: 0.5,
        ResumeQueryType.PROJECT_EXPERIENCE: 0.3,
        ResumeQueryType.SKILLS: 0.20,
    }

    DEFAULT_ACTIVE_TYPE_PENALTIES: Mapping[int, float] = {
        1: 0.30,
        2: 0.70,
        3: 1.00,
    }

    def __init__(self,*,query_type_weights: Mapping[ResumeQueryType, float] | None = None,best_score_weight: float = 0.75,
        minimum_similarity: float | None = None,evidence_limit: int | None = 10,active_type_penalties: Mapping[int, float] | None = None,) -> None:
        """
        初始化候选岗位聚合器。
        整体就围绕下面两个计算公式进行岗位jd得分计算，最后根据最终排序分数进行：
        total_score = α*best_similarity + (1-α)*evidence_score;

        evidence_score = sum(each_query_weight * each_chunk_similarity )
        evidence_score主要就是考量Query在各个层面上的与chunk的关系，即每个Query的得分是其有效权重与chunk最高相似度的乘积，再累加得到的总得分。

        Args:
            query_type_weights:
                各 QueryType 的基础权重预算。某种类型未出现在当前简历中时，
                其预算会在实际存在的类型之间重新归一化。
            best_score_weight:
                最强 Query-JD 命中在最终分数中的占比；剩余权重自动分配给
                evidence_score。
            minimum_similarity:
                可选的最低相似度阈值。低于该值的chunk不参与聚合。
                在没有评估数据前建议保持为 None。
            evidence_limit:
                最终每个候选返回的证据条数上限。None 表示返回全部证据。
                该参数不会限制参与分数计算的 Query 数量。
        """
        # 三种类型分别占多少整体预算的比例，主要用于与相似度相乘计算每个Query的得分
        weights = dict(
            query_type_weights or self.DEFAULT_QUERY_TYPE_WEIGHTS
        )

        self._query_type_weights = dict(
            query_type_weights
            or self.DEFAULT_QUERY_TYPE_WEIGHTS
        )

        self._active_type_penalties = dict(
            active_type_penalties
            or self.DEFAULT_ACTIVE_TYPE_PENALTIES
        )

        for active_count, penalty in self._active_type_penalties.items():
            if active_count <= 0:
                raise CandidateAggregationError(
                    "活跃 QueryType 数量必须大于0"
                )

            if (
                    not math.isfinite(penalty)
                    or not 0 < penalty <= 1
            ):
                raise CandidateAggregationError(
                    "QueryType 惩罚系数必须位于 (0, 1]："
                    f"active_count={active_count}, penalty={penalty}"
                )

        if not weights:
            raise CandidateAggregationError("query_type_weights不能为空")

        for query_type, weight in weights.items():
            if not math.isfinite(weight) or weight <= 0:
                raise CandidateAggregationError(
                    "QueryType 权重必须是大于0的有限数值: "
                    f"query_type={query_type}, weight={weight}")

        if (
            not math.isfinite(best_score_weight)
            or not 0 <= best_score_weight <= 1
        ):
            raise CandidateAggregationError("best_score_weight 必须位于 [0, 1] 之间")

        if (
            minimum_similarity is not None
            and not math.isfinite(minimum_similarity)
        ):
            raise CandidateAggregationError("minimum_similarity 必须是有限数值或 None")

        if evidence_limit is not None and evidence_limit <= 0:
            raise CandidateAggregationError("evidence_limit必须大于0或为 None")

        # 最高相似度在最终分数中的占比；剩余权重自动分配给 evidence_score，就是α
        self._best_score_weight = best_score_weight
        # 证据得分在最终分数中的占比,就是1-α，
        self._evidence_score_weight = 1 - best_score_weight
        # 可选的最低相似度阈值。低于该值的chunk不参与聚合,在没有评估数据前建议保持为 None。
        self._minimum_similarity = minimum_similarity or None
        self._evidence_limit = evidence_limit

    def aggregate(self,retrieval_results: VectorRetrievalResult,*,top_n: int = 3) -> CandidateAggregationResult:
        """
        将多个QueryRetrievalResult聚合为TopN岗位候选。

        该方法是聚合器对外提供的主入口。它负责校验输入、计算 Query
        有效权重、按 jd_id 汇总 chunk、构造岗位级分数并完成最终排序。

        Args:
            retrieval_results:
                每个 ResumeQueryUnit 对应的一组向量召回结果。query_id必须唯一，并且必须保留Query分组，不能提前拍平成chunk列表。
            top_n:
                最终返回的语义候选岗位数量。

        Returns:
            包含聚合统计信息和 TopN SemanticJobCandidate 的结果对象。

        Raises:
            CandidateAggregationError:
                输入为空、Query 重复、权重非法或相似度不是有限数值时抛出。
        """
        # 输入合法性校验
        self._validate_retrieval_results(retrieval_results=retrieval_results,top_n=top_n)

        # 根据实际存在的QueryType，归一化权重预算(比如说一份简历只有skills，没有其他QueryType，那么归一化后再进行惩罚，skills的权重就会是0.6)
        type_weight_plan = self._build_type_weight_plan(retrieval_results)

        # 按照一定规则，为每个Query计算最终参与聚合的有效权重(这里引入了最开始的QueryRetrievalResult的weight，
        # 但是默认就将三种类型看做同等重要，都为1，后续可以根据实际情况进行调整，比如说某一个工作经历更为重要，可以将其上调至1.5)
        query_contexts = self._build_query_contexts(retrieval_results=retrieval_results,type_weight_plan=type_weight_plan)

        # 按 jd_id 汇总召回 chunk，并完成 Query-JD 级别的 max 去重；
        (
            candidate_buckets,
            total_retrieved_chunk_count,
            total_usable_chunk_count,
        ) = self._collect_candidate_buckets(retrieval_results)

        #
        candidates = [
            self._build_candidate(
                bucket=bucket,
                retrieval_results=retrieval_results,
                query_contexts=query_contexts,
                type_weight_plan=type_weight_plan,
            )
            for bucket in candidate_buckets.values()
        ]

        candidates.sort(
            key=lambda candidate: (
                candidate.aggregate_score,
                candidate.weighted_query_coverage,
                candidate.matched_query_count,
                candidate.best_similarity,
            ),
            reverse=True,
        )

        result = CandidateAggregationResult(
            total_query_count=len(retrieval_results.query_results),
            total_retrieved_chunk_count=total_retrieved_chunk_count,
            total_usable_chunk_count=total_usable_chunk_count,
            candidate_count_before_limit=len(candidates),
            top_n=top_n,
            candidates=candidates[:top_n],
        )

        logger.debug(
            "语义候选聚合完成: query_count=%d retrieved_chunks=%d "
            "usable_chunks=%d candidates_before_limit=%d returned=%d",
            result.total_query_count,
            result.total_retrieved_chunk_count,
            result.total_usable_chunk_count,
            result.candidate_count_before_limit,
            len(result.candidates),
        )

        return result

    def _build_type_weight_plan(self,retrieval_results: VectorRetrievalResult,) -> _TypeWeightPlan:
        """
        生成当前简历的 QueryType 权重计划。

        计算过程：
        1. 找出当前简历实际存在的 QueryType；
        2. 在活跃类型之间标准化基础权重，使其总和为1；
        3. 根据活跃类型数量确定动态惩罚系数；
        4. 使用标准化权重乘以惩罚系数，得到实际计分预算。

        例如：
            WORK=0.45，SKILLS=0.20，活跃类型数量为2。

            标准化后：
                WORK   = 0.45 / 0.65 = 0.6923
                SKILLS = 0.20 / 0.65 = 0.3077

            乘以0.7惩罚后：
                WORK   = 0.4846
                SKILLS = 0.2154

            有效类型权重之和为0.7。
        """
        active_types = {
            result.query_type
            for result in retrieval_results.query_results
        }

        if not active_types:
            raise CandidateAggregationError(
                "当前简历不存在活跃 QueryType"
            )

        active_weight_sum = sum(
            self._query_type_weights[query_type]
            for query_type in active_types
        )

        if active_weight_sum <= 0:
            raise CandidateAggregationError(
                "当前简历的活跃 QueryType 总权重必须大于0"
            )

        normalized_type_weights = {
            query_type: (
                    self._query_type_weights[query_type]
                    / active_weight_sum
            )
            for query_type in active_types
        }

        active_type_count = len(active_types)

        try:
            penalty_factor = self._active_type_penalties[
                active_type_count
            ]
        except KeyError as exc:
            raise CandidateAggregationError(
                "没有配置对应的活跃 QueryType 惩罚系数: "
                f"active_type_count={active_type_count}"
            ) from exc

        effective_type_weights = {
            query_type: normalized_weight * penalty_factor
            for query_type, normalized_weight
            in normalized_type_weights.items()
        }

        return _TypeWeightPlan(
            normalized_type_weights=normalized_type_weights,
            effective_type_weights=effective_type_weights,
            penalty_factor=penalty_factor,
        )

    def _build_query_contexts(self,*,retrieval_results: VectorRetrievalResult,type_weight_plan: _TypeWeightPlan) -> dict[str, _QueryContext]:
        """
        为每个 Query 分配实际有效权重。

        同一种 QueryType 下的多个 Query 共同分享该类型的有效预算，
        避免工作经历或项目经历数量越多，总权重就越大的问题。
        """
        raw_weight_sum_by_type: dict[
            ResumeQueryType,
            float,
        ] = defaultdict(float)

        for result in retrieval_results.query_results:
            raw_weight_sum_by_type[
                result.query_type
            ] += result.weight

        query_contexts: dict[str, _QueryContext] = {}

        for result in retrieval_results.query_results:
            type_raw_weight_sum = raw_weight_sum_by_type[
                result.query_type
            ]

            effective_query_weight = (
                    type_weight_plan.effective_type_weights[
                        result.query_type
                    ]
                    * result.weight
                    / type_raw_weight_sum
            )

            query_contexts[result.query_id] = _QueryContext(
                query_type=result.query_type,
                raw_weight=result.weight,
                effective_weight=effective_query_weight,
            )

        return query_contexts

    def _collect_candidate_buckets(self,retrieval_results: VectorRetrievalResult,
        ) -> tuple[dict[int, _CandidateBucket], int, int]:
        """
        按 jd_id 汇总召回 chunk，并完成 Query-JD 级别的 max 去重。

        对于同一个 query_id 和 jd_id，只保留 similarity 最高的 chunk；
        同时保留 chunk_hit_count，便于观察某个 JD 在原始召回结果中出现
        的次数。chunk_hit_count 只用于诊断，不直接参与最终得分。

        Returns:
            candidate_buckets:
                以 jd_id 为键的候选聚合桶。
            total_retrieved_chunk_count:
                Retriever 原始返回的 chunk 总数。
            total_usable_chunk_count:
                通过 minimum_similarity 过滤后实际参与聚合的 chunk 总数。
        """
        buckets: dict[int, _CandidateBucket] = {}
        total_retrieved_chunk_count = 0
        total_usable_chunk_count = 0

        for result in retrieval_results.query_results:
            total_retrieved_chunk_count += len(result.chunks)

            for chunk in result.chunks:
                if not self._is_usable_chunk(chunk):
                    continue

                total_usable_chunk_count += 1

                bucket = buckets.setdefault(
                    chunk.jd_id,
                    _CandidateBucket(jd_id=chunk.jd_id),
                )
                bucket.chunk_hit_count += 1

                current_best = bucket.best_chunk_by_query.get(
                    result.query_id
                )

                # 只保留 每个query下对应的聚合的jd桶中，similarity 最高的 chunk，就是默认当前记录为最佳，然后后续比较是否需要更新
                if (
                    current_best is None
                    or chunk.similarity > current_best.similarity
                ):
                    bucket.best_chunk_by_query[result.query_id] = chunk

        return (
            buckets,
            total_retrieved_chunk_count,
            total_usable_chunk_count,
        )

    def _is_usable_chunk(self, chunk: RetrievedChunk) -> bool:
        """
        判断一个召回 chunk 是否达到当前聚合器的最低相似度要求。

        minimum_similarity 为 None 时，所有 Retriever 返回的 chunk 都参与
        聚合；设置阈值后，低于阈值的 chunk 会被忽略。
        """
        if self._minimum_similarity is None:
            return True

        return chunk.similarity >= self._minimum_similarity

    def _validate_retrieval_results(self,*,retrieval_results: VectorRetrievalResult,top_n: int) -> None:
        """
        校验聚合输入是否满足算法前提。

        检查 Query 列表非空、top_n 合法、query_id 唯一、
        Query 权重为正数，以及 chunk 相似度为有限数值。
        """
        if not retrieval_results.query_results:
            raise CandidateAggregationError(
                "query_results 不能为空；无 Query 时不应调用聚合器"
            )

        if top_n <= 0:
            raise CandidateAggregationError("top_n 必须大于0")

        seen_query_ids: set[str] = set()

        for result in retrieval_results.query_results:
            if not result.query_id.strip():
                raise CandidateAggregationError("query_id 不能为空")

            if result.query_id in seen_query_ids:
                raise CandidateAggregationError(
                    f"query_id 不能重复: {result.query_id}"
                )

            seen_query_ids.add(result.query_id)

            if (
                    not math.isfinite(result.weight)
                    or result.weight <= 0
            ):
                raise CandidateAggregationError(
                    "Query 权重必须是大于0的有限数值: "
                    f"query_id={result.query_id}, "
                    f"weight={result.weight}"
                )

            if result.query_type not in self._query_type_weights:
                raise CandidateAggregationError(
                    "缺少 QueryType 权重配置: "
                    f"query_type={result.query_type}"
                )

            for chunk in result.chunks:
                if not math.isfinite(chunk.similarity):
                    raise CandidateAggregationError(
                        "chunk similarity 必须是有限数值: "
                        f"query_id={result.query_id}, "
                        f"chunk_id={chunk.chunk_id}, "
                        f"similarity={chunk.similarity}"
                    )

    def _build_type_aggregations(self,*,retrieval_results: VectorRetrievalResult,all_evidence: Sequence[CandidateQueryEvidence],
        type_weight_plan: _TypeWeightPlan) -> list[QueryTypeAggregation]:
        """
        将已经计算完成的原子证据按 QueryType 分组。

        本方法只负责统计和解释，不重新计算 Query-JD 相似度。
        每种类型的 weighted_contribution 直接由该类型下所有
        CandidateQueryEvidence.weighted_similarity 相加得到。
        """
        results_by_type: dict[
            ResumeQueryType,
            list[QueryRetrievalResult],
        ] = defaultdict(list)

        evidence_by_type: dict[
            ResumeQueryType,
            list[CandidateQueryEvidence],
        ] = defaultdict(list)

        for result in retrieval_results.query_results:
            results_by_type[result.query_type].append(result)

        for evidence in all_evidence:
            evidence_by_type[evidence.query_type].append(evidence)

        aggregations: list[QueryTypeAggregation] = []

        for query_type, type_results in results_by_type.items():
            type_evidence = evidence_by_type.get(
                query_type,
                [],
            )

            effective_type_weight = (
                type_weight_plan.effective_type_weights[
                    query_type
                ]
            )

            matched_effective_weight = sum(
                evidence.effective_query_weight
                for evidence in type_evidence
            )

            query_coverage = self._clamp_unit_interval(
                matched_effective_weight
                / effective_type_weight
            )

            best_similarity = (
                max(
                    evidence.best_chunk.similarity
                    for evidence in type_evidence
                )
                if type_evidence
                else None
            )

            weighted_contribution = sum(
                evidence.weighted_similarity
                for evidence in type_evidence
            )

            aggregations.append(
                QueryTypeAggregation(
                    query_type=query_type,
                    normalized_type_weight=(
                        type_weight_plan.normalized_type_weights[
                            query_type
                        ]
                    ),
                    effective_type_weight=effective_type_weight,
                    total_query_count=len(type_results),
                    matched_query_count=len(type_evidence),
                    query_coverage=query_coverage,
                    best_similarity=best_similarity,
                    weighted_contribution=weighted_contribution,
                )
            )

        aggregations.sort(
            key=lambda item: item.effective_type_weight,
            reverse=True,
        )

        return aggregations

    def _build_candidate(
            self,
            *,
            bucket: _CandidateBucket,
            retrieval_results: VectorRetrievalResult,
            query_contexts: Mapping[str, _QueryContext],
            type_weight_plan: _TypeWeightPlan,
    ) -> SemanticJobMatchResult:
        """
        根据一个 JD 聚合桶构造岗位级候选。

        CandidateQueryEvidence 是计分的唯一原子来源。
        类型聚合和岗位总证据分数都从原子证据汇总产生。
        """
        all_evidence: list[CandidateQueryEvidence] = []

        for query_id, best_chunk in bucket.best_chunk_by_query.items():
            context = query_contexts.get(query_id)

            if context is None:
                raise CandidateAggregationError(
                    "找不到 Query 权重上下文: "
                    f"query_id={query_id}, jd_id={bucket.jd_id}"
                )

            all_evidence.append(
                CandidateQueryEvidence(
                    query_id=query_id,
                    query_type=context.query_type,
                    raw_query_weight=context.raw_weight,
                    effective_query_weight=context.effective_weight,
                    best_chunk=best_chunk,
                    weighted_similarity=(
                            best_chunk.similarity
                            * context.effective_weight
                    ),
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

        best_similarity = max(
            evidence.best_chunk.similarity
            for evidence in all_evidence
        )

        evidence_score = sum(
            evidence.weighted_similarity
            for evidence in all_evidence
        )

        # 这是惩罚后的已命中有效权重。
        matched_effective_weight = sum(
            evidence.effective_query_weight
            for evidence in all_evidence
        )

        # 除以惩罚系数后，表示活跃 Query 内部的实际覆盖程度。
        weighted_query_coverage = self._clamp_unit_interval(
            matched_effective_weight
            / type_weight_plan.penalty_factor
        )

        aggregate_score = (
                best_similarity * self._best_score_weight
                + evidence_score * self._evidence_score_weight
        )

        type_aggregations = self._build_type_aggregations(
            retrieval_results=retrieval_results,
            all_evidence=all_evidence,
            type_weight_plan=type_weight_plan,
        )

        type_contribution_sum = sum(
            aggregation.weighted_contribution
            for aggregation in type_aggregations
        )

        if not math.isclose(
                evidence_score,
                type_contribution_sum,
                rel_tol=1e-9,
                abs_tol=1e-12,
        ):
            raise CandidateAggregationError(
                "岗位证据分数与类型贡献之和不一致: "
                f"jd_id={bucket.jd_id}, "
                f"evidence_score={evidence_score}, "
                f"type_contribution_sum={type_contribution_sum}"
            )

        visible_evidence = (
            all_evidence
            if self._evidence_limit is None
            else all_evidence[: self._evidence_limit]
        )

        return SemanticJobMatchResult(
            jd_id=bucket.jd_id,
            best_similarity=best_similarity,
            overall_evidence_score=evidence_score,
            aggregate_score=aggregate_score,
            total_query_count=len(
                retrieval_results.query_results
            ),
            matched_query_count=len(all_evidence),
            weighted_query_coverage=weighted_query_coverage,
            chunk_hit_count=bucket.chunk_hit_count,
            type_aggregations=type_aggregations,
            evidence=visible_evidence,
        )

    @staticmethod
    def _clamp_unit_interval(value: float) -> float:
        """
        将浮点计算结果限制在 [0, 1] 范围内。

        Query 权重经过多次除法和累加后，理论上的1.0可能表现为
        1.0000000000000002。该方法只修正这种浮点边界误差，避免 Pydantic
        的范围校验误报，不改变正常的权重计算结果。
        """
        return min(1.0, max(0.0, value))


