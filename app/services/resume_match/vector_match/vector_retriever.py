from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from logging import getLogger

from app.services.resume_match.sql_search.jd_repository import JDRepository
from app.services.resume_match.vector_match.resume_query_builder import ResumeQueryUnit
from app.services.resume_match.vector_match.base import ResumeQueryType

from app.tools import BgeM3EmbeddingProvider

logger = getLogger(__name__)

class RetrievedChunk(BaseModel):
    """
    一条由 pgvector 召回出的 JD chunk。

    similarity 仅表示向量召回相似度，
    不是最终的岗位匹配分数。
    """

    model_config = ConfigDict(strict=True,extra="ignore")

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
    """
    一个 ResumeQueryUnit 对应的一组召回结果。
    """

    model_config = ConfigDict(strict=True,extra="ignore")

    query_id: str
    query_type: ResumeQueryType

    query_text: str
    weight: float

    chunks: list[RetrievedChunk] = Field(default_factory=list)


class VectorRetrievalResult(BaseModel):
    """
    一次Resume多QueryUnit向量召回的整体结果。
    """

    model_config = ConfigDict(strict=True,extra="ignore")

    query_results: list[QueryRetrievalResult] = Field(default_factory=list)

    @property
    def has_result(self) -> bool:
        return any(result.chunks for result in self.query_results)

    @property
    def total_chunk_count(self) -> int:
        return sum(len(result.chunks) for result in self.query_results)


class VectorRetriever:
    """
    多 QueryUnit 的 JD 向量召回器。

    输入：
        Sequence[ResumeQueryUnit]

    流程：
        ResumeQueryUnit[]
            ↓
        batch embedding
            ↓
        每个 query vector 分别检索 JD chunks
            ↓
        QueryRetrievalResult[]
            ↓
        VectorRetrievalResult
    """

    def __init__(self,repository: JDRepository,embedder: BgeM3EmbeddingProvider,*,top_k_per_query: int = 20,
                 chunk_types: Sequence[str] | None = None) -> None:

        if top_k_per_query <= 0:
            raise ValueError("top_k_per_query 必须大于 0")

        self._repository = repository
        self._embedder = embedder

        self._top_k_per_query = top_k_per_query

        self._chunk_types = list(chunk_types) if chunk_types is not None else None

    def retrieve(self,query_units: Sequence[ResumeQueryUnit],*,top_k_per_query: int | None = None,
                 chunk_types: Sequence[str] | None = None) -> VectorRetrievalResult:
        """
        对多个 QueryUnit 做向量召回。

        注意：
            Semantic Retrieval 应独立于 Exact Retrieval。

            因此前期不要默认 exclude exact JD，
            否则会丢失“Exact JD是否也出现在Semantic TopN”
            这个非常重要的 overlap 信号。
        """

        units = list(query_units)

        if not units:
            return VectorRetrievalResult(query_results=[])

        actual_top_k = top_k_per_query if top_k_per_query is not None else self._top_k_per_query

        if actual_top_k <= 0:
            raise ValueError("top_k_per_query 必须大于 0")

        actual_chunk_types = list(chunk_types) if chunk_types is not None else self._chunk_types

        # =====================================================
        # 1. QueryUnit -> text
        # =====================================================

        texts = [unit.text.strip() for unit in units]

        if any(not text for text in texts):
            raise ValueError("query_units 中存在空 query text")

        # =====================================================
        # 2. Batch Embedding
        # =====================================================

        logger.info(f"开始批量嵌入 {len(texts)} 个query text")
        start_time = time.time()

        embeddings = self._embedder.embed_queries(texts)

        end_time = time.time()
        logger.info(f"批量嵌入完成，总计 {len(texts)} 个query text，耗时{end_time - start_time:.4f}秒")

        if len(embeddings) != len(units):
            raise ValueError(
                "EmbeddingProvider 返回的向量数量与 "
                "QueryUnit 数量不一致："
                f"{len(embeddings)} != {len(units)}"
            )

        # =====================================================
        # 3. 每个 Query 独立检索 JD chunks
        # =====================================================

        logger.info(f"开始批量检索 {len(units)} 个query vector")
        start_time = time.time()

        query_results: list[QueryRetrievalResult] = []

        for unit,embedding in zip(units,embeddings,strict=True):

            if len(embedding) == 0:
                logger.error(f"QueryUnit {unit.query_id!r} 得到空embedding")
                raise ValueError

            rows = (self._repository.search_similar_chunks(
                    embedding,
                    top_k=actual_top_k,
                    chunk_types=actual_chunk_types)
                    )

            chunks = [self._build_retrieved_chunk(row) for row in rows]

            query_results.append(
                QueryRetrievalResult(
                    query_id=unit.query_id,
                    query_type=unit.query_type,
                    query_text=unit.text,
                    weight=unit.weight,
                    chunks=chunks,
                )
            )

        end_time = time.time()
        logger.info(f"批量检索完成，总计 {len(units)} 个query vector，耗时{end_time - start_time:.4f}秒")
        return VectorRetrievalResult(query_results=query_results)

    @staticmethod
    def _build_retrieved_chunk(row: dict[str, Any]) -> RetrievedChunk:

        metadata = row.get("metadata")

        if not isinstance(metadata,dict):
            metadata = {}

        return RetrievedChunk(
            chunk_id=int(row["chunk_id"]),
            jd_id=int(row["jd_id"]),
            chunk_key=row.get("chunk_key"),
            chunk_type=str(row["chunk_type"]),
            source_sequence=(int(row["source_sequence"])if row.get("source_sequence")is not None else None),
            part_index=(int(row["part_index"])if row.get("part_index")is not None else None),
            item_index=(int(row["item_index"])if row.get("item_index")is not None else None),
            content=str(row["content"]),
            metadata=metadata,
            embedding_model=row.get("embedding_model"),
            distance=float(row["distance"]),
            similarity=float(row["similarity"])
        )

