from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.services.resume_match.vector_match.rerank.base import (
    JDLookupRepository,
    RerankerProvider,
)
from app.services.resume_match.vector_match.rerank.jd_text_builder import (
    JDPassage,
    JDTextBuildError,
    JDTextBuilder,
    RerankJDDocument,
)
from app.services.resume_match.vector_match.rerank.model_schema import (
    ResumeSupportEvidence,
    ResumeSupportTypeScore,
    TargetJobRerankCandidate,
    TargetJobRerankConfig,
    TargetJobRerankResult,
    TargetJobRerankStatus,
)
from app.services.resume_match.vector_match.recall.base import ResumeQueryType
from app.services.resume_match.vector_match.recall.job_candidate_aggregator import (
    CandidateAggregationResult,
    SemanticJobMatchResult,
)
from app.services.resume_match.vector_match.recall.vector_retriever import (
    QueryRetrievalResult,
    VectorRetrievalResult,
)


logger = logging.getLogger(__name__)


class TargetJobRerankError(RuntimeError):
    """目标岗位精排输入或执行失败。"""


@dataclass(slots=True, frozen=True)
class _ScoreRequest:
    kind: str
    jd_id: int
    pair: tuple[str, str]
    query: QueryRetrievalResult | None = None
    passage: JDPassage | None = None


@dataclass(slots=True, frozen=True)
class _ScoredRequest:
    request: _ScoreRequest
    score: float


class TargetJobReranker:
    """对 CandidateAggregator 的粗召回 JD 执行 BGE-M3 精排。

    分数职责：

    * ``recall_score``：现有 ``aggregate_score``，只保留为弱召回信号；
    * ``title_similarity``：申请岗位名称与公司岗位名称的 BGE-M3 相关分；
    * ``resume_support_score``：不含纯标题 Query 的简历事实与 JD 分段相关分；
    * ``resolution_score``：三种信号的加权和，只用于候选岗位排序。

    BGE-M3 reranker 的 sigmoid 分数是相关性分数，不是已经校准的概率。
    """

    def __init__(
        self,
        *,
        repository: JDLookupRepository,
        reranker: RerankerProvider,
        config: TargetJobRerankConfig | None = None,
        jd_text_builder: JDTextBuilder | None = None,
    ) -> None:
        self._repository = repository
        self._reranker = reranker
        self._config = config or TargetJobRerankConfig()
        self._jd_text_builder = jd_text_builder or JDTextBuilder()

    def rerank(
        self,
        *,
        retrieval_result: VectorRetrievalResult,
        aggregation_result: CandidateAggregationResult,
        top_n: int | None = None,
    ) -> TargetJobRerankResult:
        """加载粗召回 JD、批量执行 BGE-M3 pair 打分并生成待确认结果。"""
        actual_top_n = self._config.rerank_top_n if top_n is None else top_n
        if actual_top_n <= 0:
            raise TargetJobRerankError("top_n 必须大于0")
        self._validate_inputs(retrieval_result, aggregation_result)

        recall_candidates = list(aggregation_result.candidates)
        if not recall_candidates:
            return self._empty_result(
                aggregation_result=aggregation_result,
                status=TargetJobRerankStatus.NO_CANDIDATE,
                warnings=["粗召回阶段没有产生候选JD"],
            )

        candidate_ids = [candidate.jd_id for candidate in recall_candidates]
        rows = self._repository.find_by_ids(candidate_ids)
        documents, missing_jd_ids, build_warnings = self._build_documents(
            rows=rows,
            candidate_ids=candidate_ids,
        )
        if not documents:
            return self._empty_result(
                aggregation_result=aggregation_result,
                status=TargetJobRerankStatus.NO_CANDIDATE,
                missing_jd_ids=missing_jd_ids,
                warnings=[*build_warnings, "所有粗召回候选均缺少可用的完整JD"],
            )

        requests = self._build_score_requests(
            requested_job_title=aggregation_result.requested_job_title,
            retrieval_result=retrieval_result,
            recall_candidates=recall_candidates,
            documents=documents,
        )
        started_at = time.perf_counter()
        scores = self._reranker.score_pairs(
            [request.pair for request in requests]
        )
        if len(scores) != len(requests):
            raise TargetJobRerankError(
                "RerankerProvider 返回数量与打分请求数量不一致: "
                f"expected={len(requests)}, actual={len(scores)}"
            )

        scored_requests = [
            _ScoredRequest(request=request, score=self._validate_score(score))
            for request, score in zip(requests, scores, strict=True)
        ]
        title_scores, support_scores = self._index_scores(scored_requests)
        recall_rank_by_id = {
            candidate.jd_id: rank
            for rank, candidate in enumerate(recall_candidates, start=1)
        }

        candidates: list[TargetJobRerankCandidate] = []
        for semantic_candidate in recall_candidates:
            document = documents.get(semantic_candidate.jd_id)
            if document is None:
                continue
            title_score = title_scores.get(semantic_candidate.jd_id)
            if title_score is None:
                raise TargetJobRerankError(
                    f"候选JD缺少标题重排分数: jd_id={semantic_candidate.jd_id}"
                )
            candidates.append(
                self._build_candidate(
                    document=document,
                    semantic_candidate=semantic_candidate,
                    recall_rank=recall_rank_by_id[semantic_candidate.jd_id],
                    title_score=title_score,
                    support_requests=support_scores.get(
                        semantic_candidate.jd_id,
                        [],
                    ),
                )
            )

        candidates.sort(
            key=lambda candidate: (
                candidate.passes_title_threshold,
                candidate.resolution_score,
                candidate.title_similarity,
                candidate.resume_support_score,
                candidate.recall_score,
            ),
            reverse=True,
        )
        candidates = [
            candidate.model_copy(update={"rerank_rank": rank})
            for rank, candidate in enumerate(candidates[:actual_top_n], start=1)
        ]

        result = self._build_decision(
            aggregation_result=aggregation_result,
            candidates=candidates,
            recall_candidate_count=len(recall_candidates),
            missing_jd_ids=missing_jd_ids,
            warnings=build_warnings,
        )
        logger.info(
            "目标岗位BGE精排完成: target_id=%s recall_count=%d returned=%d "
            "status=%s pair_count=%d elapsed_ms=%.2f",
            result.target_id,
            result.recall_candidate_count,
            result.reranked_candidate_count,
            result.status.value,
            len(requests),
            (time.perf_counter() - started_at) * 1000,
        )
        return result

    def _build_score_requests(
        self,
        *,
        requested_job_title: str,
        retrieval_result: VectorRetrievalResult,
        recall_candidates: Sequence[SemanticJobMatchResult],
        documents: Mapping[int, RerankJDDocument],
    ) -> list[_ScoreRequest]:
        requests: list[_ScoreRequest] = []
        support_queries = [
            query
            for query in retrieval_result.query_results
            if query.query_type is not ResumeQueryType.TARGET_JOB_TITLE
            and query.resume_evidence_text
            and query.resume_evidence_text.strip()
        ]

        for candidate in recall_candidates:
            document = documents.get(candidate.jd_id)
            if document is None:
                continue
            requests.append(
                _ScoreRequest(
                    kind="title",
                    jd_id=document.jd_id,
                    pair=(
                        f"申请岗位：{requested_job_title}",
                        document.title_passage,
                    ),
                )
            )

            for query in support_queries:
                passages = self._select_support_passages(
                    document=document,
                    query_type=query.query_type,
                )
                # 支持度查询只保留简历原始事实。Query 类型已经作为结构化
                # 元数据参与聚合，再添加“候选人工作经历”等模板词反而会
                # 稀释 BGE-M3 对事实内容的相关性判断。
                query_text = query.resume_evidence_text
                for passage in passages:
                    requests.append(
                        _ScoreRequest(
                            kind="support",
                            jd_id=document.jd_id,
                            pair=(query_text, passage.text),
                            query=query,
                            passage=passage,
                        )
                    )

        if not requests:
            raise TargetJobRerankError("没有生成任何重排文本对")
        return requests

    def _select_support_passages(
        self,
        *,
        document: RerankJDDocument,
        query_type: ResumeQueryType,
    ) -> list[JDPassage]:
        prefix_priority: Mapping[ResumeQueryType, tuple[str, ...]] = {
            ResumeQueryType.WORK_EXPERIENCE: (
                "qualification",
                "responsibility",
                "competency",
            ),
            ResumeQueryType.PROJECT_EXPERIENCE: (
                "responsibility",
                "competency",
                "qualification",
            ),
            ResumeQueryType.SKILLS: (
                "competency",
                "responsibility",
                "qualification",
            ),
        }
        priorities = prefix_priority.get(query_type, tuple())
        selected: list[JDPassage] = []
        for prefix in priorities:
            selected.extend(
                passage
                for passage in document.support_passages
                if passage.section.startswith(prefix)
                and passage not in selected
            )
        if not selected:
            selected = list(document.support_passages)
        return selected[: self._config.max_jd_passages]

    def _build_candidate(
        self,
        *,
        document: RerankJDDocument,
        semantic_candidate: SemanticJobMatchResult,
        recall_rank: int,
        title_score: float,
        support_requests: Sequence[_ScoredRequest],
    ) -> TargetJobRerankCandidate:
        support_score, type_scores, evidence = self._aggregate_support_scores(
            support_requests
        )
        recall_score = self._clamp_unit_interval(
            semantic_candidate.aggregate_score
        )
        resolution_score = self._clamp_unit_interval(
            title_score * self._config.title_weight
            + support_score * self._config.resume_support_weight
            + recall_score * self._config.recall_weight
        )
        return TargetJobRerankCandidate(
            rerank_rank=1,
            recall_rank=recall_rank,
            jd_id=document.jd_id,
            job_title=document.job_title,
            department=document.department,
            recall_score=recall_score,
            title_similarity=title_score,
            resume_support_score=support_score,
            resolution_score=resolution_score,
            passes_title_threshold=(
                self._config.minimum_title_similarity is None
                or title_score >= self._config.minimum_title_similarity
            ),
            support_type_scores=type_scores,
            support_evidence=evidence[: self._config.max_support_evidence],
            semantic_candidate=semantic_candidate,
        )

    def _aggregate_support_scores(
        self,
        scored_requests: Sequence[_ScoredRequest],
    ) -> tuple[
        float,
        list[ResumeSupportTypeScore],
        list[ResumeSupportEvidence],
    ]:
        # 同一简历 Query 对同一 JD 的多个分段只保留最高 BGE-M3 分数。
        best_by_query: dict[str, _ScoredRequest] = {}
        for item in scored_requests:
            query = item.request.query
            if query is None:
                continue
            current = best_by_query.get(query.query_id)
            if current is None or item.score > current.score:
                best_by_query[query.query_id] = item

        if not best_by_query:
            return 0.0, [], []

        by_type: dict[ResumeQueryType, list[_ScoredRequest]] = defaultdict(list)
        for item in best_by_query.values():
            query = item.request.query
            if query is not None:
                by_type[query.query_type].append(item)

        active_weight_sum = sum(
            self._config.resume_type_weights[query_type]
            for query_type in by_type
        )
        type_scores: list[ResumeSupportTypeScore] = []
        for query_type, items in by_type.items():
            normalized_weight = (
                self._config.resume_type_weights[query_type]
                / active_weight_sum
            )
            mean_best_score = sum(item.score for item in items) / len(items)
            type_scores.append(
                ResumeSupportTypeScore(
                    query_type=query_type,
                    query_count=len(items),
                    normalized_weight=normalized_weight,
                    mean_best_score=mean_best_score,
                    weighted_contribution=mean_best_score * normalized_weight,
                )
            )
        type_scores.sort(key=lambda item: item.normalized_weight, reverse=True)
        support_score = self._clamp_unit_interval(
            sum(item.weighted_contribution for item in type_scores)
        )

        evidence: list[ResumeSupportEvidence] = []
        for item in best_by_query.values():
            query = item.request.query
            passage = item.request.passage
            if query is None or passage is None or not query.resume_evidence_text:
                continue
            evidence.append(
                ResumeSupportEvidence(
                    query_id=query.query_id,
                    query_type=query.query_type,
                    source_index=query.source_index,
                    resume_evidence_text=query.resume_evidence_text,
                    jd_section=passage.section,
                    jd_passage=passage.text,
                    rerank_score=item.score,
                )
            )
        evidence.sort(key=lambda item: item.rerank_score, reverse=True)
        return support_score, type_scores, evidence

    def _build_decision(
        self,
        *,
        aggregation_result: CandidateAggregationResult,
        candidates: Sequence[TargetJobRerankCandidate],
        recall_candidate_count: int,
        missing_jd_ids: Sequence[int],
        warnings: Sequence[str],
    ) -> TargetJobRerankResult:
        result_warnings = list(dict.fromkeys(warnings))
        if missing_jd_ids:
            result_warnings.append(
                "部分粗召回候选缺少完整JD: "
                + "、".join(str(jd_id) for jd_id in missing_jd_ids)
            )

        eligible = [
            candidate for candidate in candidates if candidate.passes_title_threshold
        ]
        if not candidates:
            status = TargetJobRerankStatus.NO_CANDIDATE
            recommended_jd_id = None
            selected_jd_id = None
            needs_confirmation = False
            top_margin = None
        elif not eligible:
            status = TargetJobRerankStatus.NO_ELIGIBLE_CANDIDATE
            recommended_jd_id = None
            selected_jd_id = None
            needs_confirmation = True
            top_margin = None
            result_warnings.append("没有候选JD通过最低岗位标题相关分阈值")
        else:
            first = eligible[0]
            second = eligible[1] if len(eligible) > 1 else None
            top_margin = (
                first.resolution_score - second.resolution_score
                if second is not None
                else None
            )
            recommended_jd_id = first.jd_id
            score_sufficient = (
                first.resolution_score >= self._config.auto_confirm_threshold
            )
            margin_sufficient = (
                top_margin is None
                or top_margin >= self._config.minimum_top_margin
            )
            can_auto_resolve = (
                not self._config.require_manual_confirmation
                and score_sufficient
                and margin_sufficient
            )
            if can_auto_resolve:
                status = TargetJobRerankStatus.AUTO_RESOLVED
                selected_jd_id = first.jd_id
                needs_confirmation = False
            else:
                status = TargetJobRerankStatus.NEEDS_CONFIRMATION
                selected_jd_id = None
                needs_confirmation = True
                if self._config.require_manual_confirmation:
                    result_warnings.append("语义兜底岗位默认需要人工确认")
                elif not score_sufficient:
                    result_warnings.append("第一候选未达到自动确认分数阈值")
                elif not margin_sufficient:
                    result_warnings.append("前两名岗位解析分差不足，需要人工确认")

        return TargetJobRerankResult(
            target_id=aggregation_result.target_id,
            requested_job_title=aggregation_result.requested_job_title,
            reranker_model=self._reranker.model_name,
            status=status,
            recall_candidate_count=recall_candidate_count,
            reranked_candidate_count=len(candidates),
            recommended_jd_id=recommended_jd_id,
            selected_jd_id=selected_jd_id,
            needs_confirmation=needs_confirmation,
            top_margin=top_margin,
            candidates=list(candidates),
            missing_jd_ids=list(missing_jd_ids),
            warnings=list(dict.fromkeys(result_warnings)),
        )

    def _build_documents(
        self,
        *,
        rows: Sequence[Mapping[str, Any]],
        candidate_ids: Sequence[int],
    ) -> tuple[dict[int, RerankJDDocument], list[int], list[str]]:
        documents: dict[int, RerankJDDocument] = {}
        warnings: list[str] = []
        candidate_id_set = set(candidate_ids)
        for row in rows:
            row_id = row.get("id") if isinstance(row, Mapping) else None
            try:
                document = self._jd_text_builder.build(row)
            except JDTextBuildError as exc:
                warnings.append(f"候选JD无法构造重排文本: jd_id={row_id!r}")
                logger.warning("候选JD重排文本构造失败: %s", exc)
                continue
            if document.jd_id not in candidate_id_set:
                warnings.append(
                    f"Repository返回了未请求的JD: jd_id={document.jd_id}"
                )
                continue
            if document.jd_id in documents:
                warnings.append(f"Repository返回了重复JD: jd_id={document.jd_id}")
                continue
            documents[document.jd_id] = document

        missing_ids = [jd_id for jd_id in candidate_ids if jd_id not in documents]
        return documents, missing_ids, warnings

    @staticmethod
    def _index_scores(
        scored_requests: Sequence[_ScoredRequest],
    ) -> tuple[dict[int, float], dict[int, list[_ScoredRequest]]]:
        title_scores: dict[int, float] = {}
        support_scores: dict[int, list[_ScoredRequest]] = defaultdict(list)
        for item in scored_requests:
            if item.request.kind == "title":
                title_scores[item.request.jd_id] = item.score
            elif item.request.kind == "support":
                support_scores[item.request.jd_id].append(item)
            else:
                raise TargetJobRerankError(
                    f"未知打分请求类型: {item.request.kind}"
                )
        return title_scores, support_scores

    def _empty_result(
        self,
        *,
        aggregation_result: CandidateAggregationResult,
        status: TargetJobRerankStatus,
        missing_jd_ids: Sequence[int] = (),
        warnings: Sequence[str] = (),
    ) -> TargetJobRerankResult:
        return TargetJobRerankResult(
            target_id=aggregation_result.target_id,
            requested_job_title=aggregation_result.requested_job_title,
            reranker_model=self._reranker.model_name,
            status=status,
            recall_candidate_count=len(aggregation_result.candidates),
            reranked_candidate_count=0,
            recommended_jd_id=None,
            selected_jd_id=None,
            needs_confirmation=False,
            top_margin=None,
            candidates=[],
            missing_jd_ids=list(missing_jd_ids),
            warnings=list(dict.fromkeys(warnings)),
        )

    @staticmethod
    def _validate_inputs(
        retrieval_result: VectorRetrievalResult,
        aggregation_result: CandidateAggregationResult,
    ) -> None:
        if retrieval_result.target_id != aggregation_result.target_id:
            raise TargetJobRerankError("retrieval 与 aggregation 的 target_id 不一致")
        if (
            retrieval_result.requested_job_title
            != aggregation_result.requested_job_title
        ):
            raise TargetJobRerankError(
                "retrieval 与 aggregation 的 requested_job_title 不一致"
            )
        candidate_ids = [
            candidate.jd_id for candidate in aggregation_result.candidates
        ]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise TargetJobRerankError("aggregation_result 中存在重复 jd_id")

    @staticmethod
    def _validate_score(score: float) -> float:
        score = float(score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise TargetJobRerankError(
                f"RerankerProvider 分数必须位于 [0, 1]: {score}"
            )
        return score

    @staticmethod
    def _clamp_unit_interval(value: float) -> float:
        if not math.isfinite(value):
            raise TargetJobRerankError(f"候选分数不是有限数: {value}")
        return min(1.0, max(0.0, float(value)))
