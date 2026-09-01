from __future__ import annotations

from typing import Sequence
from dataclasses import dataclass, field
from pydantic import BaseModel, ConfigDict, Field

from .vector_retriever import RetrievedChunk,VectorRetrievalResult
from .resume_query_builder import ResumeQueryType



class CandidateQueryEvidence(BaseModel):
    """
    某个 Resume Query 对某个 JD 的最佳向量命中证据。
    同一个 query_id 对同一个 jd_id
    最多只保留一条 CandidateQueryEvidence。
    """
    model_config = ConfigDict(strict=True,extra="ignore",)

    query_id: str
    query_type: ResumeQueryType

    query_weight: float = Field(gt=0)

    best_chunk: RetrievedChunk

    # similarity × query_weight，主要用于聚合计算和调试
    weighted_similarity: float

class SemanticJobCandidate(BaseModel):
    """
    由多个Resume Query和多个JD chunk聚合得到的岗位级候选。

    所有分数都属于语义召回阶段，
    不是候选人与岗位的最终匹配分数。
    """

    model_config = ConfigDict(strict=True,extra="ignore")

    jd_id: int

    # 最高若干Query得分的加权平均
    semantic_score: float

    # 匹配到的Query权重 / 全部Query权重
    query_coverage: float = Field(ge=0,le=1)

    # semantic_score和query_coverage组合后的排序分数
    aggregate_score: float

    # 有多少个不同Query支持这个JD
    matched_query_count: int = Field(ge=1)

    # 原始召回结果中，这个JD一共出现了多少个chunk
    chunk_hit_count: int = Field(ge=1)

    # chunk_hit_count / 全部召回chunk数量
    # 只作为辅助解释，不作为主要排序依据
    chunk_hit_share: float = Field(ge=0,le=1)

    # 每个Query对该JD的最佳命中证据
    evidence: list[CandidateQueryEvidence] = Field(default_factory=list)

@dataclass
class _QueryBestMatch:
    query_id: str
    query_type: ResumeQueryType
    query_weight: float

    chunk: RetrievedChunk


@dataclass
class _CandidateBucket:
    jd_id: int

    job_title: str | None = None
    department: str | None = None

    chunk_hit_count: int = 0

    # 一个Query对同一个JD只保留最高分chunk
    best_match_by_query: dict[str,_QueryBestMatch] = field(default_factory=dict)


class CandidateAggregator:

    def __init__(
        self,
        *,
        semantic_weight: float = 0.85,
        max_query_contributions: int = 3,
        evidence_limit: int = 5,
        minimum_similarity: float | None = None,
    ) -> None:
        if not 0 <= semantic_weight <= 1:
            raise ValueError("semantic_weight必须位于[0, 1]之间")

        if max_query_contributions <= 0:
            raise ValueError("max_query_contributions必须大于0")

        if evidence_limit <= 0:
            raise ValueError("evidence_limit必须大于0")

        self._semantic_weight = semantic_weight
        self._coverage_weight = 1 - semantic_weight

        self._max_query_contributions = max_query_contributions
        self._evidence_limit = evidence_limit
        self._minimum_similarity = minimum_similarity

    def aggregate(self,retrieval_results: VectorRetrievalResult,*,top_n: int = 3) -> list[SemanticJobCandidate]:
        if top_n <= 0:
            raise ValueError("top_n必须大于0")

        if not retrieval_results:
            return []

        total_query_weight = sum(result.weight for result in retrieval_results)

        if total_query_weight <= 0:
            raise ValueError("Query总权重必须大于0")

        total_hit_count = sum(len(result.chunks) for result in retrieval_results)

        buckets = self._build_buckets(retrieval_results)

        candidates = [
            self._build_candidate(
                bucket=bucket,
                total_query_weight=total_query_weight,
                total_hit_count=total_hit_count,
            )
            for bucket in buckets.values()
        ]

        candidates.sort(
            key=lambda candidate: (
                candidate.aggregate_score,
                candidate.semantic_score,
                candidate.matched_query_count,
            ),
            reverse=True,
        )

        return candidates[:top_n]

    def _build_buckets(self,retrieval_results: VectorRetrievalResult) -> dict[int, _CandidateBucket]:

        buckets: dict[int, _CandidateBucket] = {}

        for result in retrieval_results:
            if result.weight <= 0:
                raise ValueError(
                    f"Query权重必须大于0: "
                    f"query_id={result.query_id}"
                )

            for chunk in result.chunks:
                if not self._is_usable_chunk(chunk):
                    continue

                bucket = buckets.setdefault(
                    chunk.jd_id,
                    _CandidateBucket(
                        jd_id=chunk.jd_id,
                        job_title=chunk.job_title,
                        department=chunk.department,
                    ),
                )

                bucket.chunk_hit_count += 1

                current_best = (
                    bucket.best_match_by_query.get(
                        result.query_id
                    )
                )

                if (
                    current_best is None
                    or chunk.similarity
                    > current_best.chunk.similarity
                ):
                    bucket.best_match_by_query[
                        result.query_id
                    ] = _QueryBestMatch(
                        query_id=result.query_id,
                        query_type=result.query_type,
                        query_weight=result.weight,
                        chunk=chunk,
                    )

        return buckets

    def _build_candidate(
        self,
        *,
        bucket: _CandidateBucket,
        total_query_weight: float,
        total_hit_count: int,
    ) -> SemanticJobCandidate:
        query_matches = list(
            bucket.best_match_by_query.values()
        )

        query_matches.sort(
            key=lambda match: (
                match.chunk.similarity
                * match.query_weight
            ),
            reverse=True,
        )

        top_matches = query_matches[
            :self._max_query_contributions
        ]

        semantic_score = self._weighted_average(
            top_matches
        )

        matched_query_weight = sum(
            match.query_weight
            for match in query_matches
        )

        query_coverage = (
            matched_query_weight
            / total_query_weight
        )

        aggregate_score = (
            semantic_score * self._semantic_weight
            + query_coverage * self._coverage_weight
        )

        chunk_hit_share = (
            bucket.chunk_hit_count / total_hit_count
            if total_hit_count > 0
            else 0.0
        )

        evidence = [
            CandidateQueryEvidence(
                query_id=match.query_id,
                query_type=match.query_type,
                chunk_id=match.chunk.chunk_id,
                chunk_type=match.chunk.chunk_type,
                chunk_content=match.chunk.content,
                similarity=match.chunk.similarity,
            )
            for match in query_matches[
                :self._evidence_limit
            ]
        ]

        return SemanticJobCandidate(
            jd_id=bucket.jd_id,
            job_title=bucket.job_title,
            department=bucket.department,
            semantic_score=semantic_score,
            query_coverage=query_coverage,
            aggregate_score=aggregate_score,
            matched_query_count=len(
                bucket.best_match_by_query
            ),
            chunk_hit_count=bucket.chunk_hit_count,
            chunk_hit_share=chunk_hit_share,
            evidence=evidence,
        )

    def _is_usable_chunk(self,chunk: RetrievedChunk) -> bool:

        if self._minimum_similarity is None:
            return True

        return chunk.similarity >= self._minimum_similarity


    @staticmethod
    def _weighted_average(matches: Sequence[_QueryBestMatch]) -> float:

        if not matches:
            return 0.0

        total_weight = sum(match.query_weight for match in matches)

        if total_weight <= 0:
            return 0.0

        return (
            sum(match.chunk.similarity * match.query_weight for match in matches) / total_weight
        )