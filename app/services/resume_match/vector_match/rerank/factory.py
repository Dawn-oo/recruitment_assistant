from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from app.services.resume_match.vector_match.rerank.model_schema import TargetJobRerankConfig
from app.services.resume_match.vector_match.SemanticMatchingService import (
    SemanticTargetMatchingService,
)
from app.services.resume_match.vector_match.rerank.target_job_reranker import TargetJobReranker
from app.services.resume_match.vector_match.recall.base import EmbeddingProvider
from app.services.resume_match.vector_match.recall.job_candidate_aggregator import (
    CandidateAggregator,
)
from app.services.resume_match.vector_match.recall.resume_query_builder import (
    ResumeQueryBuilder,
)
from app.services.resume_match.vector_match.recall.vector_retriever import VectorRetriever
from app.tools.rerank_tools.BGE_M3_rerank import BgeReranker

if TYPE_CHECKING:
    from app.database_search.jd_repository import JDRepository


def create_semantic_target_matching_service(
    *,
    repository: JDRepository,
    embedder: EmbeddingProvider,
    reranker_model_path: str | Path | None = None,
    reranker_device: str | None = None,
    reranker_batch_size: int | None = None,
    rerank_config: TargetJobRerankConfig | None = None,
) -> SemanticTargetMatchingService:
    """用默认组件构造主流程可直接调用的单岗位语义兜底服务。"""
    bge_reranker = BgeReranker(
        model_path=reranker_model_path,
        device=reranker_device,
        batch_size=reranker_batch_size,
    )
    return SemanticTargetMatchingService(
        query_builder=ResumeQueryBuilder(),
        vector_retriever=VectorRetriever(
            repository=repository,
            embedder=embedder,
        ),
        candidate_aggregator=CandidateAggregator(),
        target_job_reranker=TargetJobReranker(
            repository=repository,
            reranker=bge_reranker,
            config=rerank_config,
        ),
    )
