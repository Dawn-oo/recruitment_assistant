from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.services.resume_match.sql_search.jd_repository import JDRepository
from app.services.resume_match.vector_match.recall.base import (
    EmbeddingProvider,
    ResumeQueryType,
)
from app.services.resume_match.vector_match.recall.resume_query_builder import ResumeQueryUnit


logger = logging.getLogger(__name__)


class RetrievedChunk(BaseModel):
    """一条由 pgvector 召回出的 JD chunk。"""

    model_config = ConfigDict(strict=True, extra="ignore")

    chunk_id: int
    jd_id: int
    chunk_key: str | None = None
    chunk_type: str
    source_sequence: int | None = None
    part_index: int | None = None
    item_index: int | None = None
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding_model: str | None = None
    distance: float
    similarity: float


class QueryRetrievalResult(BaseModel):
    """一个目标岗位感知 Query 对应的一组 JD chunk 召回结果。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    target_id: str = Field(min_length=1)
    requested_job_title: str = Field(min_length=1)
    query_id: str = Field(min_length=1)
    query_type: ResumeQueryType
    query_text: str = Field(min_length=1)
    resume_evidence_text: str | None = None
    source_index: int | None = None
    weight: float = Field(gt=0)
    chunks: list[RetrievedChunk] = Field(default_factory=list)


class VectorRetrievalResult(BaseModel):
    """针对一个申请岗位执行多 Query 向量召回的整体结果。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    target_id: str = Field(min_length=1)
    requested_job_title: str = Field(min_length=1)
    query_results: list[QueryRetrievalResult] = Field(default_factory=list)

    @property
    def has_result(self) -> bool:
        """是否至少有一个 Query 召回了 chunk。"""
        return any(result.chunks for result in self.query_results)

    @property
    def total_chunk_count(self) -> int:
        """所有 Query 原始召回 chunk 数量之和。"""
        return sum(len(result.chunks) for result in self.query_results)


class VectorRetriever:
    """为单个申请岗位批量嵌入多 Query，并分别召回 JD chunks。"""

    def __init__(
        self,
        repository: JDRepository,
        embedder: EmbeddingProvider,
        *,
        top_k_per_query: int = 20,
        chunk_types: Sequence[str] | None = None,
    ) -> None:
        if top_k_per_query <= 0:
            raise ValueError("top_k_per_query 必须大于0")

        self._repository = repository
        self._embedder = embedder
        self._top_k_per_query = top_k_per_query
        self._chunk_types = list(chunk_types) if chunk_types is not None else None

    def retrieve(
        self,
        query_units: Sequence[ResumeQueryUnit],
        *,
        top_k_per_query: int | None = None,
        chunk_types: Sequence[str] | None = None,
    ) -> VectorRetrievalResult:
        """执行一次单目标岗位的多 Query 向量召回。"""
        units = list(query_units)
        if not units:
            raise ValueError("query_units 不能为空")

        target_id, requested_job_title = self._validate_query_units(units)
        actual_top_k = (
            top_k_per_query
            if top_k_per_query is not None
            else self._top_k_per_query
        )
        if actual_top_k <= 0:
            raise ValueError("top_k_per_query 必须大于0")

        actual_chunk_types = (
            list(chunk_types)
            if chunk_types is not None
            else self._chunk_types
        )
        texts = [unit.text.strip() for unit in units]

        embedding_started_at = time.perf_counter()
        logger.info(
            "目标岗位Query批量嵌入开始: target_id=%s query_count=%d",
            target_id,
            len(texts),
        )
        embeddings = list(self._embedder.embed_queries(texts))
        logger.info(
            "目标岗位Query批量嵌入完成: target_id=%s query_count=%d "
            "elapsed_ms=%.2f",
            target_id,
            len(texts),
            (time.perf_counter() - embedding_started_at) * 1000,
        )

        if len(embeddings) != len(units):
            raise ValueError(
                "EmbeddingProvider 返回的向量数量与 QueryUnit 数量不一致: "
                f"{len(embeddings)} != {len(units)}"
            )

        retrieval_started_at = time.perf_counter()
        query_results: list[QueryRetrievalResult] = []

        for unit, embedding in zip(units, embeddings, strict=True):
            self._validate_embedding(embedding, query_id=unit.query_id)
            rows = self._repository.search_similar_chunks(
                embedding,
                top_k=actual_top_k,
                chunk_types=actual_chunk_types,
            )
            chunks = [self._build_retrieved_chunk(row) for row in rows]

            logger.debug(
                "单Query召回完成: target_id=%s query_id=%s query_type=%s "
                "chunk_count=%d",
                target_id,
                unit.query_id,
                unit.query_type.value,
                len(chunks),
            )

            query_results.append(
                QueryRetrievalResult(
                    target_id=target_id,
                    requested_job_title=requested_job_title,
                    query_id=unit.query_id,
                    query_type=unit.query_type,
                    query_text=unit.text,
                    resume_evidence_text=unit.resume_evidence_text,
                    source_index=unit.source_index,
                    weight=unit.weight,
                    chunks=chunks,
                )
            )

        result = VectorRetrievalResult(
            target_id=target_id,
            requested_job_title=requested_job_title,
            query_results=query_results,
        )
        logger.info(
            "目标岗位多Query召回完成: target_id=%s query_count=%d "
            "chunk_count=%d elapsed_ms=%.2f",
            target_id,
            len(query_results),
            result.total_chunk_count,
            (time.perf_counter() - retrieval_started_at) * 1000,
        )
        return result

    @staticmethod
    def _validate_query_units(units: Sequence[ResumeQueryUnit]) -> tuple[str, str]:
        """保证一次 retrieve 只处理同一个申请岗位，且 query_id 唯一。"""
        target_ids = {unit.target_id.strip() for unit in units}
        requested_titles = {unit.requested_job_title.strip() for unit in units}

        if "" in target_ids or len(target_ids) != 1:
            raise ValueError("一次 retrieve 只能接收同一个非空 target_id 的 Query")
        if "" in requested_titles or len(requested_titles) != 1:
            raise ValueError("一次 retrieve 只能接收同一个非空 requested_job_title 的 Query")

        query_ids = [unit.query_id.strip() for unit in units]
        if any(not query_id for query_id in query_ids):
            raise ValueError("query_units 中存在空 query_id")
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("query_units 中存在重复 query_id")
        if any(not unit.text.strip() for unit in units):
            raise ValueError("query_units 中存在空 query text")

        return next(iter(target_ids)), next(iter(requested_titles))

    @staticmethod
    def _validate_embedding(embedding: Sequence[float],*,query_id: str) -> None:
        """在访问数据库前拒绝空向量和非有限向量。"""
        if not embedding:
            raise ValueError(f"QueryUnit 得到空 embedding: query_id={query_id}")
        if any(not math.isfinite(float(value)) for value in embedding):
            raise ValueError(f"QueryUnit embedding 包含非有限数值: query_id={query_id}")

    @staticmethod
    def _build_retrieved_chunk(row: dict[str, Any]) -> RetrievedChunk:
        """将 Repository 行转换为稳定的召回结果模型。"""
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        return RetrievedChunk(
            chunk_id=int(row["chunk_id"]),
            jd_id=int(row["jd_id"]),
            chunk_key=row.get("chunk_key"),
            chunk_type=str(row["chunk_type"]),
            source_sequence=(
                int(row["source_sequence"])
                if row.get("source_sequence") is not None
                else None
            ),
            part_index=(
                int(row["part_index"])
                if row.get("part_index") is not None
                else None
            ),
            item_index=(
                int(row["item_index"])
                if row.get("item_index") is not None
                else None
            ),
            content=str(row["content"]),
            metadata=metadata,
            embedding_model=row.get("embedding_model"),
            distance=float(row["distance"]),
            similarity=float(row["similarity"]),
        )
