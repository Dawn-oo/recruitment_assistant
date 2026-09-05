from __future__ import annotations

import logging
import time

from pydantic import BaseModel, ConfigDict, Field

from app.services.resume_handle.resume_extractor.resume_schema import ResumeModel
from app.services.resume_match.vector_match.rerank.model_schema import TargetJobRerankResult
from app.services.resume_match.vector_match.rerank.target_job_reranker import TargetJobReranker
from app.services.resume_match.vector_match.recall.job_candidate_aggregator import (
    CandidateAggregationResult,
    CandidateAggregator,
)
from app.services.resume_match.vector_match.recall.resume_query_builder import (
    ResumeQueryBuildResult,
    ResumeQueryBuilder,
)
from app.services.resume_match.vector_match.recall.vector_retriever import (
    VectorRetrievalResult,
    VectorRetriever,
)


logger = logging.getLogger(__name__)


class SemanticTargetMatchingResult(BaseModel):
    """单个未精确命中岗位从多 Query 召回到 BGE-M3 精排的完整结果。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: str = "semantic_target_matching_v1"
    query_build_result: ResumeQueryBuildResult
    retrieval_result: VectorRetrievalResult
    aggregation_result: CandidateAggregationResult
    rerank_result: TargetJobRerankResult
    warnings: list[str] = Field(default_factory=list)


class SemanticTargetMatchingService:
    """供主流程调用的单目标岗位语义兜底服务。

    主流程应当只对 ``ExactJobMatcher`` 未命中的岗位逐个调用本服务；已经
    精确命中的岗位不需要执行向量召回和 BGE-M3 精排。
    """

    def __init__(
        self,
        *,
        query_builder: ResumeQueryBuilder,
        vector_retriever: VectorRetriever,
        candidate_aggregator: CandidateAggregator,
        target_job_reranker: TargetJobReranker,
        default_top_k_per_query: int = 30,
        default_recall_top_n: int = 10,
        default_rerank_top_n: int = 5,
    ) -> None:
        for name, value in (
            ("default_top_k_per_query", default_top_k_per_query),
            ("default_recall_top_n", default_recall_top_n),
            ("default_rerank_top_n", default_rerank_top_n),
        ):
            if value <= 0:
                raise ValueError(f"{name} 必须大于0")

        self._query_builder = query_builder
        self._vector_retriever = vector_retriever
        self._candidate_aggregator = candidate_aggregator
        self._target_job_reranker = target_job_reranker
        self._default_top_k_per_query = default_top_k_per_query
        self._default_recall_top_n = default_recall_top_n
        self._default_rerank_top_n = default_rerank_top_n

    def match(
        self,
        *,
        resume: ResumeModel,
        requested_job_title: str,
        target_id: str,
        top_k_per_query: int | None = None,
        recall_top_n: int | None = None,
        rerank_top_n: int | None = None,
    ) -> SemanticTargetMatchingResult:
        """执行多 Query 粗召回、JD 聚合和 BGE-M3 精排。"""
        actual_top_k = self._resolve_positive(
            top_k_per_query,
            self._default_top_k_per_query,
            "top_k_per_query",
        )
        actual_recall_top_n = self._resolve_positive(
            recall_top_n,
            self._default_recall_top_n,
            "recall_top_n",
        )
        actual_rerank_top_n = self._resolve_positive(
            rerank_top_n,
            self._default_rerank_top_n,
            "rerank_top_n",
        )

        started_at = time.perf_counter()
        query_build_result = self._query_builder.build(
            resume,
            requested_job_title=requested_job_title,
            target_id=target_id,
        )
        retrieval_result = self._vector_retriever.retrieve(
            query_build_result.query_units,
            top_k_per_query=actual_top_k,
        )
        aggregation_result = self._candidate_aggregator.aggregate(
            retrieval_result,
            top_n=actual_recall_top_n,
        )
        rerank_result = self._target_job_reranker.rerank(
            retrieval_result=retrieval_result,
            aggregation_result=aggregation_result,
            top_n=actual_rerank_top_n,
        )
        warnings = list(
            dict.fromkeys(
                [
                    *query_build_result.warnings,
                    *rerank_result.warnings,
                ]
            )
        )
        result = SemanticTargetMatchingResult(
            query_build_result=query_build_result,
            retrieval_result=retrieval_result,
            aggregation_result=aggregation_result,
            rerank_result=rerank_result,
            warnings=warnings,
        )
        logger.info(
            "单目标岗位语义兜底完成: target_id=%s requested_title=%s "
            "recall_count=%d rerank_count=%d status=%s elapsed_ms=%.2f",
            target_id,
            requested_job_title,
            len(aggregation_result.candidates),
            len(rerank_result.candidates),
            rerank_result.status.value,
            (time.perf_counter() - started_at) * 1000,
        )
        return result

    @staticmethod
    def _resolve_positive(
        value: int | None,
        default: int,
        name: str,
    ) -> int:
        actual = default if value is None else value
        if actual <= 0:
            raise ValueError(f"{name} 必须大于0")
        return actual
