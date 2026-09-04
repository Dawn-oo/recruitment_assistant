"""岗位匹配应用服务。

该模块把 job_match 目录下已经完成的组件串联成一个稳定入口：

1. 申请岗位名称精确匹配；
2. Resume Query 构造；
3. JD chunk 向量召回；
4. chunk 到 JD 的语义候选聚合；
5. 精确结果与语义结果融合，并补齐完整 JD。

``CandidateAggregationResult`` 作为语义分支的过程结果保留；真正交给
后续 Agent 上下文构造层消费的是 ``AgentCandidateContext``。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field


from app.services.resume_match.target_job_resolver import (
    AgentCandidateContext,
    CandidateContextStatus,
    CandidateSelector,
)
from app.services.resume_handle.resume_handle_service import ResumeProcessingResult
from app.services.resume_match.exact_match.exact_job_matcher import (
    ExactJobMatcher,
    ExactJobMatchResult,
)
from app.services.resume_match.exact_match.job_intent_norm import JobIntentNormalizer, JobIntentNormalizeResult
from app.services.resume_match.vector_match.recall.resume_query_builder import ResumeQueryBuilder
from app.services.resume_match.vector_match.recall.vector_retriever import VectorRetriever

from app.services.resume_match.vector_match.recall.job_candidate_aggregator import (
    CandidateAggregationResult,
    CandidateAggregator
)


logger = logging.getLogger(__name__)


class JobMatchingServiceError(RuntimeError):
    """岗位匹配应用服务无法继续完成时抛出的异常。"""


class JobMatchingResult(BaseModel):
    """一次完整岗位匹配流程的结果。"""

    model_config = ConfigDict(strict=True,extra="forbid")

    schema_version: str = "job_matching_result_v1"

    # 精确匹配分支没有执行或执行失败时为 None。
    exact_result: ExactJobMatchResult | None = None

    # 语义分支没有执行或执行失败时为 None。
    # 该字段用于审计、调试和前端解释，不建议原样发送给 LLM。
    semantic_result: CandidateAggregationResult | None = None

    # 下一层 AgentContextBuilder 应消费的正式业务结果。
    candidate_context: AgentCandidateContext

    warnings: list[str] = Field(default_factory=list)

    @property
    def consumable(self) -> bool:
        """判断本次结果是否至少包含一个可供 Agent 分析的完整 JD。"""
        return self.candidate_context.consumable

    def to_agent_input(self) -> AgentCandidateContext:
        """返回下一层 Agent 上下文构造器应接收的数据。"""
        if not self.consumable:
            raise JobMatchingServiceError(
                "岗位匹配结果不可供Agent消费: "
                f"status={self.candidate_context.status.value}, "
                f"blocked_reason="
                f"{self.candidate_context.blocked_reason}"
            )

        return self.candidate_context


class JobMatchingService:
    """串联精确匹配、语义检索、候选聚合与候选整合。"""

    def __init__(
        self,
        *,
        job_intent_normalizer: JobIntentNormalizer,
        exact_matcher: ExactJobMatcher,
        query_builder: ResumeQueryBuilder,
        vector_retriever: VectorRetriever,
        candidate_aggregator: CandidateAggregator,
        candidate_selector: CandidateSelector,
        default_top_k_per_query: int = 30,
        default_semantic_top_n: int = 3,
    ) -> None:

        """注入岗位匹配各阶段组件并配置默认召回数量。"""
        if default_top_k_per_query <= 0:
            raise JobMatchingServiceError(
                "default_top_k_per_query必须大于0"
            )

        if default_semantic_top_n <= 0:
            raise JobMatchingServiceError(
                "default_semantic_top_n必须大于0"
            )

        self._job_intent_normalizer = job_intent_normalizer
        self._exact_matcher = exact_matcher
        self._query_builder = query_builder
        self._vector_retriever = vector_retriever
        self._candidate_aggregator = candidate_aggregator
        self._candidate_selector = candidate_selector
        self._default_top_k_per_query = default_top_k_per_query
        self._default_semantic_top_n = default_semantic_top_n

    def match_resume(
        self,
        *,
        resume: ResumeProcessingResult,
        top_k_per_query: int | None = None,
        semantic_top_n: int | None = None,
    ) -> JobMatchingResult:
        """执行一份结构化简历的完整岗位匹配流程。

        Args:
            resume:
                已完成解析和结构化校验的简历模型。

            top_k_per_query:
                每个 ResumeQueryUnit 从 pgvector 召回的 chunk 数量；未传时使用服务初始化配置。

            semantic_top_n:
                CandidateAggregator 最终保留的语义 JD 数量；未传时使用
                服务初始化配置。

        Returns:
            同时包含两条分支过程结果和统一候选上下文的 JobMatchingResult。

        Notes:
            精确分支与语义分支互相独立。单个分支异常会记录日志并降级，
            只要另一分支仍能产生带完整 JD 的候选，结果就可以继续交给
            Agent；CandidateSelector 本身失败则作为服务错误向上抛出。
        """
        actual_top_k = self._resolve_positive_option(
            value=top_k_per_query,
            default=self._default_top_k_per_query,
            option_name="top_k_per_query",
        )
        actual_semantic_top_n = self._resolve_positive_option(
            value=semantic_top_n,
            default=self._default_semantic_top_n,
            option_name="semantic_top_n",
        )

        job_intent_result = self._job_intent_norm(result=resume)

        requested_job_count = (
            len(job_intent_result.job_titles)
            if job_intent_result is not None
            else 0
        )

        start_time = time.perf_counter()
        logger.info(
            "岗位匹配流程开始: "
            "requested_job_count=%d "
            "top_k_per_query=%d "
            "semantic_top_n=%d",
            requested_job_count,
            actual_top_k,
            actual_semantic_top_n,
        )


        exact_result, exact_warnings = (
            self._run_exact_branch(
                job_intent_result=job_intent_result,
            )
        )

        semantic_result, semantic_warnings = self._run_semantic_branch(
            resume=resume,
            top_k_per_query=actual_top_k,
            semantic_top_n=actual_semantic_top_n,
        )

        service_warnings = self._deduplicate_texts(
            [
                *exact_warnings,
                *semantic_warnings,
            ]
        )

        try:
            candidate_context = self._candidate_selector.select(
                exact_result=exact_result,
                semantic_result=semantic_result,
            )
        except Exception as exc:
            logger.exception("精确与语义候选整合失败")
            raise JobMatchingServiceError(
                "CandidateSelector执行失败"
            ) from exc

        candidate_context = self._merge_service_warnings(
            candidate_context=candidate_context,
            service_warnings=service_warnings,
        )

        result = JobMatchingResult(
            exact_result=exact_result,
            semantic_result=semantic_result,
            candidate_context=candidate_context,
            warnings=service_warnings,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "岗位匹配流程完成: status=%s exact_candidates=%d "
            "semantic_candidates=%d selected=%d analyzable=%d "
            "elapsed_ms=%.2f",
            result.candidate_context.status.value,
            sum(1 for intent in exact_result.intent_results if intent.matched_jds) if exact_result else 0,
            len(semantic_result.candidates) if semantic_result else 0,
            result.candidate_context.selected_candidate_count,
            result.candidate_context.analyzable_candidate_count,
            elapsed_ms,
        )

        return result

    def _job_intent_norm(self,*,result: ResumeProcessingResult) -> JobIntentNormalizeResult:
        """对简历中的岗位意图进行标准化处理。"""
        return self._job_intent_normalizer.normalize(result.resume.basic_info.target_job_title)

    def _run_exact_branch(self,*,job_intent_result: JobIntentNormalizeResult | None,) -> tuple[ExactJobMatchResult | None,list[str]]:
        """
        执行岗位名称精确匹配。

        JobIntentNormalizeResult可能包含多个岗位名称，
        ExactJobMatcher会分别匹配每一个岗位，并返回对应的
        ExactIntentMatchResult列表。

        需要注意：
        - 没有岗位意图时正常跳过精确匹配；
        - 全部命中、部分命中、全部未命中都属于正常业务结果；
        - 只有执行过程中抛出异常，才视为精确匹配分支失败；
        - 即使全部未命中，也必须保留ExactJobMatchResult，
          以便CandidateSelector获得unmatched_titles并执行语义兜底。
        """
        if job_intent_result is None:
            logger.debug("没有岗位意图标准化结果，跳过精确匹配")
            return None, []

        if not job_intent_result.job_titles:
            logger.debug(
                "岗位意图结果中没有有效岗位名称，"
                "跳过精确匹配"
            )
            return None, []

        try:
            exact_result = self._exact_matcher.match(job_intent_result)
        except Exception:
            logger.exception(
                "岗位名称精确匹配失败: "
                "requested_job_count=%d",
                len(job_intent_result.job_titles),
            )

            return None, [
                "岗位名称精确匹配执行失败，"
                "本次仅使用语义匹配结果"
            ]

        total_intent_count = len(exact_result.intent_results)

        matched_intent_count = sum(
            bool(intent.matched_jds)
            for intent in exact_result.intent_results
        )

        unmatched_intent_count = (total_intent_count - matched_intent_count)

        matched_jd_count = sum(len(intent.matched_jds) for intent in exact_result.intent_results)

        if matched_intent_count == 0:
            match_state = "none_matched"
        elif unmatched_intent_count == 0:
            match_state = "all_matched"
        else:
            match_state = "partially_matched"

        logger.info(
            "精确匹配分支完成: "
            "state=%s intent_count=%d "
            "matched_intent_count=%d "
            "unmatched_intent_count=%d "
            "matched_jd_count=%d",
            match_state,
            total_intent_count,
            matched_intent_count,
            unmatched_intent_count,
            matched_jd_count,
        )

        logger.debug(
            "精确匹配岗位结果: "
            "apply_job_titles=%s "
            "matched_jd_ids=%s "
            "unmatched_titles=%s",
            job_intent_result.job_titles,
            [matched_id.jd_id for intent in exact_result.intent_results for matched_id in intent.matched_jds],
            [intent.raw_title for intent in exact_result.intent_results if not intent.matched_jds],
        )



        return exact_result, []

    def _run_semantic_branch(self,*,resume: ResumeProcessingResult,top_k_per_query: int,semantic_top_n: int
        ) -> tuple[CandidateAggregationResult | None, list[str]]:
        """执行 Query 构造、向量召回和岗位级语义聚合。"""
        try:
            query_units = self._query_builder.build(resume)

            if not query_units:
                logger.warning(
                    "ResumeQueryBuilder没有生成可用QueryUnit"
                )
                return None, ["简历中没有可用于语义岗位匹配的信息"]

            retrieval_result = self._vector_retriever.retrieve(
                query_units.query_units,
                top_k_per_query=top_k_per_query,
            )
            aggregation_result = self._candidate_aggregator.aggregate(
                retrieval_result,
                top_n=semantic_top_n,
            )
        except Exception:
            logger.exception("语义岗位匹配分支执行失败")
            return None, [
                "语义岗位匹配失败，本次仅使用岗位名称精确匹配结果"
            ]

        logger.debug(
            "语义匹配分支完成: query_count=%d candidate_count=%d",
            len(query_units.query_units),
            len(aggregation_result.candidates),
        )
        return aggregation_result, []

    @staticmethod
    def _resolve_positive_option(*,value: int | None,default: int,option_name: str) -> int:
        """解析调用级正整数配置，并拒绝零和负数。"""
        actual_value = default if value is None else value

        if actual_value <= 0:
            raise JobMatchingServiceError(
                f"{option_name}必须大于0"
            )

        return actual_value

    @staticmethod
    def _deduplicate_texts(values: Sequence[str]) -> list[str]:
        """去除空文本并保持原始顺序去重。"""
        return list(
            dict.fromkeys(
                value.strip()
                for value in values
                if value and value.strip()
            )
        )

    def _merge_service_warnings(
        self,
        *,
        candidate_context: AgentCandidateContext,
        service_warnings: Sequence[str],
    ) -> AgentCandidateContext:
        """把分支降级信息合并到候选上下文，并同步调整上下文状态。"""
        if not service_warnings:
            return candidate_context

        merged_warnings = self._deduplicate_texts(
            [
                *candidate_context.warnings,
                *service_warnings,
            ]
        )
        status = candidate_context.status

        if status == CandidateContextStatus.READY:
            status = CandidateContextStatus.DEGRADED

        payload = candidate_context.model_dump(mode="python")
        payload["status"] = status
        payload["warnings"] = merged_warnings

        return AgentCandidateContext.model_validate(payload)
