from __future__ import annotations

import logging
from enum import Enum
from typing import Any
from collections.abc import Mapping, Sequence
from pydantic import BaseModel,ConfigDict,Field,ValidationError

from app.services.resume_match.exact_match.exact_job_matcher import ExactJobMatchResult,ExactMatchType
from app.services.resume_match.vector_match.recall.job_candidate_aggregator import SemanticJobMatchResult,CandidateAggregationResult
from app.services.resume_match.sql_search.jd_repository import JDRepository

logger = logging.getLogger(__name__)


class CandidateSelectionError(ValueError):
    """候选 JD 选择阶段的输入、配置或数据契约错误。"""


class CandidateSource(str, Enum):
    """一个候选 JD 进入 Agent 上下文的来源。"""

    EXACT = "exact"
    SEMANTIC = "semantic"
    BOTH = "both"


class CandidatePriority(str, Enum):
    """候选 JD 在本次面试分析中的业务优先级。"""

    PRIMARY = "primary"
    ALTERNATIVE = "alternative"


class CandidateSelectionMode(str, Enum):
    """本次候选 JD 使用的整体选择策略。"""

    EXACT_WITH_ALTERNATIVES = "exact_with_alternatives"
    SEMANTIC_FALLBACK = "semantic_fallback"
    NO_CANDIDATE = "no_candidate"


class CandidateContextStatus(str, Enum):
    """候选 JD 上下文是否能够继续交给 Agent 分析。"""
    # 存在至少一个候选 JD，并且被选中的 JD 都成功加载了完整结构化信息，也没有影响分析可靠性的异常
    READY = "ready"
    #至少存在一个能够分析的完整 JD，所以流程可以继续，但数据存在部分缺失或不确定性
    DEGRADED = "degraded"
    # 没有任何可供 Agent 分析的完整候选 JD
    BLOCKED = "blocked"


class ResponsibilityModel(BaseModel):
    """一组岗位职责。"""

    model_config = ConfigDict(strict=True,extra="ignore")

    tasks: list[str] = Field(default_factory=list)
    sequence: int | None = None
    description: str
    time_percentage: int | float | str | None = None


class JobDescriptionContext(BaseModel):
    """提供给Agent分析的结构化JD。"""

    model_config = ConfigDict(strict=True,extra="ignore")

    jd_id: int

    job_title: str
    department: str

    responsibilities: list[ResponsibilityModel] = Field(default_factory=list)

    minimum_education: str
    education_background: str
    work_experience_raw: str | None = None

    competencies: list[str] = Field(default_factory=list)


class ExactMatchEvidence(BaseModel):
    """一个岗位名称对当前JD提供的精确命中依据。"""

    model_config = ConfigDict(strict=True,extra="ignore")

    raw_title: str
    normalized_title: str | None = None
    match_type: ExactMatchType | None = None


class AgentJobCandidateContext(BaseModel):
    """一个可供后续 Agent 消费的统一候选 JD。"""

    model_config = ConfigDict(strict=True, extra="ignore")

    selection_rank: int = Field(ge=1)
    jd_id: int

    source: CandidateSource
    priority: CandidatePriority

    exact_evidence: list[ExactMatchEvidence] = Field(default_factory=list)

    # 该JD在原始语义候选中的排名。
    semantic_rank: int | None = Field(default=None,ge=1)

    semantic_candidate: SemanticJobMatchResult | None = None

    # 如果完整 JD 加载失败，保留候选引用，但禁止 Agent 执行适配分析。
    analyzable: bool
    jd: JobDescriptionContext | None = None

    warnings: list[str] = Field(default_factory=list)


class AgentCandidateContext(BaseModel):
    """CandidateSelector 对 Agent 上下文层输出的统一业务契约。"""

    model_config = ConfigDict(strict=True, extra="ignore")

    schema_version: str = "agent_candidate_context_v1"

    status: CandidateContextStatus
    selection_mode: CandidateSelectionMode

    exact_input_count: int = Field(ge=0)
    semantic_input_count: int = Field(ge=0)
    selected_candidate_count: int = Field(ge=0)
    analyzable_candidate_count: int = Field(ge=0)

    candidates: list[AgentJobCandidateContext] = Field(default_factory=list)

    # 精确匹配中没有找到数据库 JD 的原始岗位名称。
    unmatched_requested_titles: list[str] = Field(default_factory=list)

    # 被选中但未能加载完整 JD 的 ID。
    missing_jd_ids: list[int] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)
    blocked_reason: str | None = None

    @property
    def consumable(self) -> bool:
        """判断当前结果中是否至少存在一个可供 Agent 分析的完整候选。"""
        return (
            self.status != CandidateContextStatus.BLOCKED
            and self.analyzable_candidate_count > 0
        )

    def to_agent_payload(self) -> dict[str, Any]:
        """序列化为适合写入 Agent State 或提示词上下文的 JSON 数据。"""
        return self.model_dump(mode="json",exclude_none=True)


class _CandidateDraft:
    __slots__ = (
        "jd_id",
        "priority",
        "exact_evidence",
        "semantic_candidate",
        "semantic_rank",
    )

    def __init__(self,*,jd_id: int,priority: CandidatePriority,) -> None:

        self.jd_id = jd_id
        self.priority = priority
        self.exact_evidence: list[ExactMatchEvidence] = []
        self.semantic_candidate: (SemanticJobMatchResult | None) = None
        self.semantic_rank: int | None = None


class CandidateSelector:
    """
    合并岗位名称精确匹配与语义召回结果，输出统一 Agent 上下文。

    默认策略：
    1. 有精确匹配时，所有精确命中的 JD 都作为主要候选；
    2. 同一个 JD 同时被语义召回时，合并为 BOTH，不重复返回；
    3. 有精确匹配时，再补充少量语义候选作为备选岗位；
    4. 没有精确匹配时，使用语义TopN作为兜底；
    5. CandidateSelector 不重新计算任何匹配分数；
    6. 完整 JD 缺失时保留候选来源，但标记 analyzable=False。
    """

    def __init__(self,repository: JDRepository,*,semantic_alternative_limit: int = 2,semantic_fallback_limit: int = 3) -> None:

        """配置精确命中后的语义补充数量和精确未命中时的兜底数量。"""
        if semantic_alternative_limit < 0:
            raise CandidateSelectionError("semantic_alternative_limit 不能小于0")

        if semantic_fallback_limit <= 0:
            raise CandidateSelectionError("semantic_fallback_limit 必须大于0")

        self._repository = repository
        self._semantic_alternative_limit = semantic_alternative_limit
        self._semantic_fallback_limit = semantic_fallback_limit

    def select(self,*,exact_result: ExactJobMatchResult | None,semantic_result: CandidateAggregationResult | None
        ) -> AgentCandidateContext:
        """
        合并前置匹配结果并构造供Agent消费的统一候选上下文。

        Args:
            exact_result:
                ExactJobMatcher 输出。精确匹配链路失败或未执行时可为 None。
            semantic_result:
                CandidateAggregator 输出。向量链路失败或未执行时可为 None。

        Returns:
            可直接写入 Agent State 的 AgentCandidateContext。
        """

        semantic_candidates = (
            semantic_result.candidates
            if semantic_result is not None
            else []
        )
        self._validate_semantic_candidates(semantic_candidates)

        (
            exact_evidence_by_jd,
            exact_jd_order,
            unmatched_requested_titles,
            exact_warnings,
        ) = self._collect_exact_matches(exact_result)

        semantic_by_jd, semantic_jd_order = (self._index_semantic_candidates(semantic_candidates))

        drafts, selection_mode = self._select_drafts(
            exact_evidence_by_jd=exact_evidence_by_jd,
            exact_jd_order=exact_jd_order,
            semantic_by_jd=semantic_by_jd,
            semantic_jd_order=semantic_jd_order,
        )

        jd_contexts, hydration_warnings = (
            self._load_selected_jd_contexts(
                drafts=drafts,
            )
        )

        result = self._build_agent_context(
            drafts=drafts,
            selection_mode=selection_mode,
            exact_input_count=len(exact_result.intent_results) if exact_result is not None else 0,
            semantic_input_count=len(semantic_candidates) if semantic_result is not None else 0,
            unmatched_requested_titles=unmatched_requested_titles,
            jd_contexts=jd_contexts,
            initial_warnings=hydration_warnings,
        )

        logger.info(
            "候选JD上下文构造完成: mode=%s status=%s "
            "exact_inputs=%d semantic_inputs=%d selected=%d analyzable=%d",
            result.selection_mode.value,
            result.status.value,
            result.exact_input_count,
            result.semantic_input_count,
            result.selected_candidate_count,
            result.analyzable_candidate_count,
        )
        logger.debug(
            "候选JD选择结果: jd_ids=%s missing_jd_ids=%s",
            [candidate.jd_id for candidate in result.candidates],
            result.missing_jd_ids,
        )

        return result

    def _load_selected_jd_contexts(self,*,drafts: Sequence[_CandidateDraft],) -> tuple[dict[int, JobDescriptionContext],list[str]]:
        """
        根据候选草稿中的jd_id批量查询并构造JD上下文。

        处理过程：
        1. 从drafts中提取并去重jd_id；
        2. 调用JDRepository.find_by_ids()批量查询；
        3. 将每条JDRow转换为JobDescriptionContext；
        4. 忽略Repository意外返回的非目标JD；
        5. 单条JD转换失败时记录告警，不影响其他JD；
        6. 返回以jd_id为键的JD上下文映射。

        数据库整体查询失败属于降级边界，因此这里会记录异常并
        返回空映射，后续_build_agent_context()负责决定最终是
        DEGRADED还是BLOCKED。
        """
        selected_jd_ids = list(
            dict.fromkeys(draft.jd_id for draft in drafts)
        )

        if not selected_jd_ids:
            return {}, []

        selected_jd_id_set = set(selected_jd_ids)

        try:
            rows = self._repository.find_by_ids(selected_jd_ids)
        except Exception:
            logger.exception(
                "批量查询候选岗位JD失败: jd_ids=%s",
                selected_jd_ids,
            )

            return {}, [
                "候选岗位完整JD批量查询失败，"
                "暂时无法获得岗位分析上下文"
            ]

        context_by_id: dict[int,JobDescriptionContext] = {}

        warnings: list[str] = []

        for row in rows:
            row_id = (
                row.get("id")
                if isinstance(row, Mapping)
                else None
            )

            try:
                context = self._map_jd_row_to_context(row)
            except CandidateSelectionError as exc:
                logger.warning("JD记录转换失败: jd_id=%r, error=%s",row_id,exc)

                warnings.append(f"存在无法转换为Agent上下文的JD记录: jd_id={row_id!r}")
                continue

            if context.jd_id not in selected_jd_id_set:
                logger.warning("Repository返回了未请求的JD: jd_id=%d",context.jd_id)

                warnings.append("Repository返回了未请求的JD: "f"jd_id={context.jd_id}")
                continue

            if context.jd_id in context_by_id:
                logger.warning("Repository返回了重复的JD: jd_id=%d",context.jd_id)

                warnings.append("Repository返回了重复的JD: "f"jd_id={context.jd_id}")
                continue

            context_by_id[context.jd_id] = context

        return context_by_id, warnings

    def _validate_semantic_candidates(self,semantic_candidates: Sequence[SemanticJobMatchResult]
    ) -> None:
        """校验语义候选中 jd_id 唯一，避免重复候选造成来源覆盖。"""

        seen_jd_ids: set[int] = set()

        for candidate in semantic_candidates:
            if candidate.jd_id in seen_jd_ids:
                raise CandidateSelectionError(
                    "语义候选中的jd_id不能重复: "
                    f"jd_id={candidate.jd_id}"
                )

            seen_jd_ids.add(candidate.jd_id)

    def _collect_exact_matches(self,exact_result: ExactJobMatchResult | None
    ) -> tuple[dict[int, list[ExactMatchEvidence]],list[int],list[str],list[str]]:

        """
        按jd_id整理精确匹配结果。

        Returns:
            evidence_by_jd:
                每个JD对应的精确匹配依据。

            jd_order:
                精确匹配JD第一次出现的顺序。

            unmatched_titles:
                没有匹配到JD的原始岗位名称。

            warnings:
                匹配状态与matched_jds不一致等异常信息。
        """

        evidence_by_jd: dict[int, list[ExactMatchEvidence]] = {}
        jd_order: list[int] = []
        unmatched_titles: list[str] = []
        warnings: list[str] = []

        if exact_result is None:
            return evidence_by_jd,jd_order,unmatched_titles,warnings

        for index, intent in enumerate(exact_result.intent_results):

            raw_title = intent.raw_title.strip()

            if not raw_title:
                warnings.append(f"第{index + 1}个岗位意图缺少raw_title")
                continue

            # 没有实际JD，不生成候选。这就是对精确匹配未匹配岗位的处理。
            if not intent.matched_jds:
                if raw_title not in unmatched_titles:
                    unmatched_titles.append(raw_title)

                status = getattr(intent.status, "value", intent.status)

                if status == "matched":
                    warnings.append(
                        "岗位状态为matched但matched_jds为空，"
                        "已按unresolved处理: "
                        f"raw_title={raw_title}"
                    )

                continue

            match_evidence = ExactMatchEvidence(
                raw_title=raw_title,
                normalized_title=intent.normalized_title,
                match_type=intent.match_type,
            )

            for matched_jd in intent.matched_jds:
                # matched_jd.jd_id就是job_descriptions.id。
                jd_id = matched_jd.jd_id

                if jd_id not in evidence_by_jd:
                    evidence_by_jd[jd_id] = []
                    jd_order.append(jd_id)

                if match_evidence not in evidence_by_jd[jd_id]:
                    evidence_by_jd[jd_id].append(
                        match_evidence
                    )

        return (
            evidence_by_jd,
            jd_order,
            unmatched_titles,
            warnings,
        )

    def _index_semantic_candidates(self,semantic_candidates: Sequence[SemanticJobMatchResult]
    ) -> tuple[dict[int, tuple[int, SemanticJobMatchResult]],list[int]]:

        """按 CandidateAggregator 已有排序建立jd_id到语义候选的索引。"""

        semantic_by_jd: dict[int,tuple[int, SemanticJobMatchResult]] = {}
        semantic_jd_order: list[int] = []

        for rank, candidate in enumerate(semantic_candidates,start=1):
            semantic_by_jd[candidate.jd_id] = (rank,candidate)
            semantic_jd_order.append(candidate.jd_id)

        return semantic_by_jd, semantic_jd_order

    def _select_drafts(self,
            *,
            exact_evidence_by_jd: Mapping[int,list[ExactMatchEvidence]],
            exact_jd_order: Sequence[int],
            semantic_by_jd: Mapping[int,tuple[int, SemanticJobMatchResult]],
            semantic_jd_order: Sequence[int]
    ) -> tuple[list[_CandidateDraft], CandidateSelectionMode]:

        """根据精确优先、语义补充或语义兜底策略生成候选草稿。"""

        drafts: list[_CandidateDraft] = []

        if exact_jd_order:
            for jd_id in exact_jd_order:
                draft = _CandidateDraft(jd_id=jd_id,priority=CandidatePriority.PRIMARY)
                draft.exact_evidence.extend(exact_evidence_by_jd[jd_id])

                semantic_entry = semantic_by_jd.get(jd_id)
                if semantic_entry is not None:
                    (
                        draft.semantic_rank,
                        draft.semantic_candidate,
                    ) = semantic_entry

                drafts.append(draft)

            exact_jd_ids = set(exact_jd_order)
            alternative_count = 0

            for jd_id in semantic_jd_order:
                if jd_id in exact_jd_ids:
                    continue

                if alternative_count>= self._semantic_alternative_limit:
                    break

                semantic_rank, semantic_candidate = semantic_by_jd[jd_id]
                draft = _CandidateDraft(jd_id=jd_id,priority=CandidatePriority.ALTERNATIVE)
                draft.semantic_rank = semantic_rank
                draft.semantic_candidate = semantic_candidate
                drafts.append(draft)
                alternative_count += 1

            return drafts,CandidateSelectionMode.EXACT_WITH_ALTERNATIVES

        for index, jd_id in enumerate(semantic_jd_order[: self._semantic_fallback_limit]):

            semantic_rank, semantic_candidate = semantic_by_jd[jd_id]
            draft = _CandidateDraft(
                jd_id=jd_id,
                priority=(
                    CandidatePriority.PRIMARY
                    if index == 0
                    else CandidatePriority.ALTERNATIVE
                ),
            )
            draft.semantic_rank = semantic_rank
            draft.semantic_candidate = semantic_candidate
            drafts.append(draft)

        selection_mode = (CandidateSelectionMode.SEMANTIC_FALLBACK if drafts else CandidateSelectionMode.NO_CANDIDATE)
        return drafts, selection_mode

    def _build_agent_context(
            self,
            *,
            drafts: Sequence[_CandidateDraft],
            selection_mode: CandidateSelectionMode,
            exact_input_count: int,
            semantic_input_count: int,
            unmatched_requested_titles: Sequence[str],
            jd_contexts: Mapping[int, JobDescriptionContext],
            initial_warnings: Sequence[str] = (),
    ) -> AgentCandidateContext:
        """
        将候选草稿和完整JD组合成Agent可直接消费的统一上下文。

        负责：
        1. 按候选草稿顺序生成最终selection_rank；
        2. 将精确匹配依据和语义候选结果写入统一候选模型；
        3. 根据jd_id关联完整JobDescriptionContext；
        4. 标记每个候选是否能够执行岗位适配分析；
        5. 统计缺失JD、可分析候选数量；
        6. 根据整体结果生成READY、DEGRADED或BLOCKED状态；
        7. 汇总上游告警、未解析岗位和JD缺失告警。

        """
        if exact_input_count < 0:
            raise CandidateSelectionError("exact_input_count不能小于0")

        if semantic_input_count < 0:
            raise CandidateSelectionError("semantic_input_count不能小于0")

        # 保留顺序，同时去除重复和空告警。
        warnings = list(
            dict.fromkeys(
                warning.strip()
                for warning in initial_warnings
                if warning and warning.strip()
            )
        )

        # 保留未解析岗位的原始顺序并去重。
        unmatched_titles = list(
            dict.fromkeys(
                title.strip()
                for title in unmatched_requested_titles
                if title and title.strip()
            )
        )

        candidates: list[AgentJobCandidateContext] = []
        missing_jd_ids: list[int] = []

        for selection_rank, draft in enumerate(drafts,start=1):
            # semantic_candidate和semantic_rank应当同时存在或同时为空。
            has_semantic_candidate = (draft.semantic_candidate is not None)
            has_semantic_rank = (draft.semantic_rank is not None)

            if has_semantic_candidate != has_semantic_rank:
                raise CandidateSelectionError(
                    "候选JD的semantic_candidate与"
                    "semantic_rank状态不一致: "
                    f"jd_id={draft.jd_id}, "
                    f"semantic_candidate="
                    f"{has_semantic_candidate}, "
                    f"semantic_rank={draft.semantic_rank}"
                )

            jd_context = jd_contexts.get(draft.jd_id)
            analyzable = jd_context is not None

            candidate_warnings: list[str] = []

            if not analyzable:
                missing_jd_ids.append(draft.jd_id)
                candidate_warnings.append("候选岗位缺少完整JD，暂时不能执行岗位适配分析")

            candidate = AgentJobCandidateContext(
                selection_rank=selection_rank,
                jd_id=draft.jd_id,
                source=self._resolve_source(draft),
                priority=draft.priority,

                # 直接使用按JD维度整理后的精确匹配依据。
                exact_evidence=list(draft.exact_evidence),

                # 直接复用CandidateAggregator的输出，
                semantic_rank=draft.semantic_rank,
                semantic_candidate=draft.semantic_candidate,

                analyzable=analyzable,
                jd=jd_context,
                warnings=candidate_warnings,
            )

            candidates.append(candidate)

        analyzable_candidate_count = sum(candidate.analyzable for candidate in candidates)

        if unmatched_titles:
            warnings.append(
                "部分申请岗位未匹配到数据库JD: "
                + "、".join(unmatched_titles)
            )

        if missing_jd_ids:
            warnings.append(
                "部分候选岗位未能加载完整JD: "
                + "、".join(
                    str(jd_id)
                    for jd_id in missing_jd_ids
                )
            )

        if not candidates:
            status = CandidateContextStatus.BLOCKED
            blocked_reason = "no_candidate_job"

            warnings.append("精确匹配和语义召回均未产生候选JD")

        elif analyzable_candidate_count == 0:
            status = CandidateContextStatus.BLOCKED
            blocked_reason = "candidate_jd_context_unavailable"

            warnings.append("所有候选岗位均缺少完整JD上下文,无法执行岗位适配分析")

        elif missing_jd_ids or unmatched_titles or warnings:

            status = CandidateContextStatus.DEGRADED
            blocked_reason = None

        else:
            status = CandidateContextStatus.READY
            blocked_reason = None

        # 新增告警后再次去重，避免数据库查询异常和JD缺失
        # 产生含义相同的重复提示。
        warnings = list(dict.fromkeys(warnings))

        return AgentCandidateContext(
            status=status,
            selection_mode=selection_mode,
            exact_input_count=exact_input_count,
            semantic_input_count=semantic_input_count,
            selected_candidate_count=len(candidates),
            analyzable_candidate_count=analyzable_candidate_count,
            candidates=candidates,
            unmatched_requested_titles=unmatched_titles,
            missing_jd_ids=missing_jd_ids,
            warnings=warnings,
            blocked_reason=blocked_reason,
        )

    @staticmethod
    def _resolve_source(draft: _CandidateDraft) -> CandidateSource:
        """
        根据候选草稿中包含的匹配依据判断候选来源。
        """
        has_exact = bool(draft.exact_evidence)
        has_semantic = (
                draft.semantic_candidate is not None
        )

        if has_exact and has_semantic:
            return CandidateSource.BOTH

        if has_exact:
            return CandidateSource.EXACT

        if has_semantic:
            return CandidateSource.SEMANTIC

        raise CandidateSelectionError(
            "候选JD没有精确匹配或语义召回依据: "
            f"jd_id={draft.jd_id}"
        )

    @staticmethod
    def _map_jd_row_to_context(row) -> JobDescriptionContext:
        """
        将数据库返回的JDRow映射为Agent使用的JobDescriptionContext。

        这里只选择Agent分析需要的业务字段，不传递：
        - content_hash
        - source_payload
        - schema_version
        - created_at
        - updated_at

        PostgreSQL JSONB和TEXT[]字段应当已经由数据库驱动转换为
        Python的list、dict等对象，再由Pydantic完成结构校验。
        """
        if not isinstance(row, Mapping):
            raise CandidateSelectionError("JDRepository返回值必须是Mapping类型")

        jd_id = row.get("id")

        if not isinstance(jd_id, int):
            raise CandidateSelectionError(f"JD数据库记录缺少有效主键id:id={jd_id!r}")

        context_payload: dict[str, Any] = {
            "jd_id": jd_id,
            "job_title": row.get("job_title"),
            "department": row.get("department"),
            "responsibilities": row.get("responsibilities"),
            "minimum_education": row.get("minimum_education"),
            "education_background": row.get("education_background"),
            "work_experience_raw": row.get("work_experience_raw"),
            "competencies": row.get("competencies"),
        }

        try:
            print(context_payload)
            return JobDescriptionContext.model_validate(context_payload)
        except ValidationError as exc:
            raise CandidateSelectionError(
                "JD数据库记录无法转换为"
                "JobDescriptionContext: "
                f"jd_id={jd_id}"
            ) from exc