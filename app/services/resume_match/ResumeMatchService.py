"""简历申请岗位匹配编排服务。

精确匹配是主路径，语义匹配只处理精确匹配未命中的岗位。任何语义候选以及
同名多 JD 都必须经过人工确认；只有全部申请岗位均得到唯一 JD 后，结果才可
进入 Agent。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.services.resume_handle.ResumeHandleService import ResumeProcessingResult
from app.services.resume_match.exact_match.exact_job_matcher import (
    ExactIntentMatchResult,
    ExactJobMatcher,
    ExactJobMatchResult,
)
from app.services.resume_match.exact_match.job_intent_norm import (
    JobIntentNormalizer,
    JobIntentNormalizeResult,
)
from app.services.resume_match.vector_match.SemanticMatchingService import (
    SemanticTargetMatchingResult,
    SemanticTargetMatchingService,
)


logger = logging.getLogger(__name__)


class ResumeMatchServiceError(RuntimeError):
    """匹配流程无法继续或人工确认请求非法。"""


class ResumeMatchStatus(str, Enum):
    READY = "ready"

    # 已有部分岗位可以进入 Agent，但仍有岗位等待人工确认
    PARTIALLY_READY = "partially_ready"

    # 当前没有任何可分析岗位，必须等待人工确认
    NEEDS_CONFIRMATION = "needs_confirmation"

    # 人工拒绝已有候选，需要重新描述或重新检索
    NEEDS_MANUAL_RESOLUTION = "needs_manual_resolution"

    # 无法继续，例如没有岗位意图且没有人工输入
    BLOCKED = "blocked"


class TargetMatchSource(str, Enum):
    EXACT = "申请岗位与 JD 匹配精确"

    SEMANTIC = "申请岗位与 JD 匹配语义匹配"

    # 人工描述后通过工具重新检索
    MANUAL_SEARCH = "人工描述后通过工具重新检索"

class ResolutionMethod(str, Enum):
    UNIQUE_EXACT_MATCH = "唯一精确匹配JD"
    HUMAN_CONFIRMED = "人工确认JD"
    HUMAN_SPECIFIED_JD = "人工指定JD"

class TargetMatchStatus(str, Enum):
    RESOLVED = "resolved"

    # 有候选，等待人工选择
    NEEDS_CONFIRMATION = "needs_confirmation"

    # 人工拒绝所有候选，需要重新输入岗位描述
    NEEDS_MANUAL_RESOLUTION = "needs_manual_resolution"

    # 重新输入描述后正在再次检索
    RESEARCHING = "researching"

    # 检索后没有任何候选
    NO_CANDIDATE = "no_candidate"


class MatchCandidate(BaseModel):
    """可供人工选择的 JD；candidate_rank 仅用于展示，不限制人工选择。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    jd_id: int
    job_title: str = Field(min_length=1)
    department: str | None = None
    source: TargetMatchSource
    candidate_rank: int = Field(ge=1)
    score: float | None = Field(default=None, ge=0.0, le=1.0)


class TargetMatchResult(BaseModel):
    """一个标准化申请岗位的精确/语义匹配结果。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    target_id: str = Field(min_length=1)
    requested_job_title: str = Field(min_length=1)
    status: TargetMatchStatus
    source: TargetMatchSource | None = None
    candidates: list[MatchCandidate] = Field(default_factory=list)
    selected_jd_id: int | None = None
    requires_human_confirmation: bool = False
    # 人工拒绝所有候选的原因
    rejection_reason: str | None = None

    # 人工重新描述的目标岗位要求
    manual_query: str | None = None

    # 候选集合的版本，重新检索后递增
    candidate_version: int = Field(default=1, ge=1)

    # 记录被人工拒绝的候选，便于审计和避免重复推荐
    rejected_candidate_ids: list[int] = Field(default_factory=list)

    resolution_method: ResolutionMethod | None = None

    semantic_result: SemanticTargetMatchingResult | None = None

    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_selection(self) -> "TargetMatchResult":
        candidate_ids = {candidate.jd_id for candidate in self.candidates}
        if self.selected_jd_id is not None and self.selected_jd_id not in candidate_ids:
            raise ValueError("selected_jd_id 必须属于当前岗位的候选集合")
        if self.status == TargetMatchStatus.RESOLVED and self.selected_jd_id is None:
            raise ValueError("RESOLVED 状态必须存在 selected_jd_id")
        if self.status == TargetMatchStatus.NEEDS_CONFIRMATION:
            if not self.requires_human_confirmation or self.selected_jd_id is not None:
                raise ValueError("待确认岗位不能预先选择 JD")
        return self


class ResumeMatchResult(BaseModel):
    """匹配编排结果，也是确认接口的输入快照。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: str = "resume_match_result_v1"
    status: ResumeMatchStatus
    exact_result: ExactJobMatchResult
    targets: list[TargetMatchResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    analyzable_target_count: int = Field(ge=0)
    pending_confirmation_count: int = Field(ge=0)
    manual_resolution_count: int = Field(ge=0)

    @property
    def consumable_by_agent(self) -> bool:
        return self.analyzable_target_count > 0

    @property
    def selected_jd_ids(self) -> list[int]:
        return [
            target.selected_jd_id
            for target in self.targets
            if target.selected_jd_id is not None
        ]

    def to_agent_payload(self) -> dict:
        resolved_targets = [
            target
            for target in self.targets
            if (
                    target.status == TargetMatchStatus.RESOLVED
                    and target.selected_jd_id is not None
            )
        ]

        if not resolved_targets:
            raise ResumeMatchServiceError(
                "当前没有已确认的岗位可以进入 Agent"
            )

        return {
            "schema_version": self.schema_version,
            "match_status": self.status.value,
            "analysis_scope": "partial" if (
                    self.status == ResumeMatchStatus.PARTIALLY_READY
            ) else "complete",
            "selected_targets": [
                {
                    "target_id": target.target_id,
                    "requested_job_title": target.requested_job_title,
                    "source": target.source.value if target.source else None,
                    "resolution_method": (
                        target.resolution_method.value
                        if target.resolution_method
                        else None
                    ),
                    "selected_jd_id": target.selected_jd_id,
                }
                for target in resolved_targets
            ],
            "pending_target_ids": [
                target.target_id
                for target in self.targets
                if target.status != TargetMatchStatus.RESOLVED
            ],
        }


class MatchResumeService:
    """按申请岗位串联精确匹配、语义降级及人工确认。"""

    def __init__(
        self,
        *,
        job_intent_normalizer: JobIntentNormalizer,
        exact_matcher: ExactJobMatcher,
        semantic_matching_service: SemanticTargetMatchingService,
    ) -> None:
        self._job_intent_normalizer = job_intent_normalizer
        self._exact_matcher = exact_matcher
        self._semantic_matching_service = semantic_matching_service

    def match_resume(
        self,
        *,
        resume: ResumeProcessingResult,
        top_k_per_query: int | None = None,
        recall_top_n: int | None = None,
        rerank_top_n: int | None = None,
    ) -> ResumeMatchResult:
        """首次匹配；仅对精确未命中的岗位逐个执行语义降级。"""
        started_at = time.perf_counter()
        intent_result = self._normalize_intents(resume)
        if not intent_result.job_titles:
            raise ResumeMatchServiceError("简历中没有有效的申请岗位")

        try:
            exact_result = self._exact_matcher.match(intent_result)
        except Exception as exc:
            logger.exception("岗位精确匹配失败")
            raise ResumeMatchServiceError("岗位精确匹配失败，未执行语义降级") from exc

        targets: list[TargetMatchResult] = []
        for index, exact_intent in enumerate(exact_result.intent_results, start=1):
            target_id = f"target_{index}"
            targets.append(
                self._resolve_target(
                    resume=resume,
                    target_id=target_id,
                    exact_intent=exact_intent,
                    top_k_per_query=top_k_per_query,
                    recall_top_n=recall_top_n,
                    rerank_top_n=rerank_top_n,
                )
            )

        result = self._build_result(exact_result=exact_result, targets=targets)
        logger.info(
            "简历岗位匹配完成: status=%s target_count=%d semantic_target_count=%d "
            "elapsed_ms=%.2f",
            result.status.value,
            len(result.targets),
            sum(target.semantic_result is not None for target in result.targets),
            (time.perf_counter() - started_at) * 1000,
        )
        return result

    def confirm_candidates(
        self,
        *,
        result: ResumeMatchResult,
        selections: Mapping[str, int],
    ) -> ResumeMatchResult:
        """确认待选 JD。

        ``selections`` 为 ``target_id -> jd_id``。JD 可以不是 Top1，但必须属于
        对应 target 的候选集合。未提交的待确认岗位保持原状态，因此允许前端
        分批确认；所有岗位确认完毕后总体状态恢复为 READY。
        """

        known_target_ids = {target.target_id for target in result.targets}
        unknown_target_ids = set(selections) - known_target_ids
        if unknown_target_ids:
            raise ResumeMatchServiceError(
                f"包含未知 target_id: {sorted(unknown_target_ids)}"
            )

        confirmed_targets: list[TargetMatchResult] = []
        for target in result.targets:
            selected_jd_id = selections.get(target.target_id)
            if selected_jd_id is None:
                confirmed_targets.append(target)
                continue
            if target.status != TargetMatchStatus.NEEDS_CONFIRMATION:
                raise ResumeMatchServiceError(
                    f"岗位 {target.target_id} 当前不处于待确认状态"
                )
            candidate_ids = {candidate.jd_id for candidate in target.candidates}
            if selected_jd_id not in candidate_ids:
                raise ResumeMatchServiceError(
                    f"JD {selected_jd_id} 不属于岗位 {target.target_id} 的候选集合"
                )

            target_payload = target.model_dump(mode="python")
            target_payload.update(
                status=TargetMatchStatus.RESOLVED,
                selected_jd_id=selected_jd_id,
                requires_human_confirmation=False,
                resolution_method=ResolutionMethod.HUMAN_CONFIRMED,
            )
            confirmed_targets.append(TargetMatchResult.model_validate(target_payload))

        return self._build_result(
            exact_result=result.exact_result,
            targets=confirmed_targets,
            inherited_warnings=result.warnings,
        )

    def reject_candidates(
            self,
            *,
            result: ResumeMatchResult,
            target_id: str,
            reason: str | None = None,
    ) -> ResumeMatchResult:
        """拒绝指定目标岗位的当前全部候选 JD。

        该操作只处理处于 NEEDS_CONFIRMATION 状态的岗位。

        拒绝后：
        1. 当前候选集合被清空；
        2. 候选 JD ID 被记录到 rejected_candidate_ids；
        3. selected_jd_id 被清空；
        4. 岗位进入 NEEDS_MANUAL_RESOLUTION；
        5. 其他已经 RESOLVED 的岗位不受影响；
        6. 总体状态由 _build_result() 重新计算。
        """
        normalized_target_id = target_id.strip()

        if not normalized_target_id:
            raise ResumeMatchServiceError("target_id 不能为空")

        normalized_reason = reason.strip() if reason else None

        target_found = False
        updated_targets: list[TargetMatchResult] = []

        for target in result.targets:
            if target.target_id != normalized_target_id:
                updated_targets.append(target)
                continue

            target_found = True

            if target.status != TargetMatchStatus.NEEDS_CONFIRMATION:
                raise ResumeMatchServiceError(
                    f"岗位 {normalized_target_id} 当前不能拒绝候选: "
                    f"status={target.status.value}"
                )

            if not target.candidates:
                raise ResumeMatchServiceError(
                    f"岗位 {normalized_target_id} 没有可拒绝的候选 JD"
                )

            current_candidate_ids = [
                candidate.jd_id
                for candidate in target.candidates
            ]

            rejected_candidate_ids = list(
                dict.fromkeys(
                    [
                        *target.rejected_candidate_ids,
                        *current_candidate_ids,
                    ]
                )
            )

            target_payload = target.model_dump(mode="python")

            target_payload.update(
                status=TargetMatchStatus.NEEDS_MANUAL_RESOLUTION,
                candidates=[],
                selected_jd_id=None,
                requires_human_confirmation=False,
                rejection_reason=normalized_reason,
                rejected_candidate_ids=rejected_candidate_ids,
                manual_query=None,
                warnings=self._deduplicate(
                    [
                        *target.warnings,
                        (
                            "人工已拒绝当前全部候选 JD，"
                            "需要重新描述目标岗位"
                        ),
                    ]
                ),
            )

            updated_targets.append(TargetMatchResult.model_validate(target_payload))

        if not target_found:
            raise ResumeMatchServiceError(f"没有找到目标岗位: target_id={normalized_target_id}")

        logger.info(
            "人工拒绝岗位候选: target_id=%s reason=%s",
            normalized_target_id,
            normalized_reason,
        )

        return self._build_result(
            exact_result=result.exact_result,
            targets=updated_targets,
            inherited_warnings=result.warnings,
        )

    def _resolve_target(
        self,
        *,
        resume: ResumeProcessingResult,
        target_id: str,
        exact_intent: ExactIntentMatchResult,
        top_k_per_query: int | None,
        recall_top_n: int | None,
        rerank_top_n: int | None,
    ) -> TargetMatchResult:
        exact_candidates = [
            MatchCandidate(
                jd_id=matched.jd_id,
                job_title=matched.job_title,
                department=matched.department,
                source=TargetMatchSource.EXACT,
                candidate_rank=rank,
            )
            for rank, matched in enumerate(exact_intent.matched_jds, start=1)
        ]

        if len(exact_candidates) == 1:
            return TargetMatchResult(
                target_id=target_id,
                requested_job_title=exact_intent.raw_title,
                status=TargetMatchStatus.RESOLVED,
                source=TargetMatchSource.EXACT,
                candidates=exact_candidates,
                selected_jd_id=exact_candidates[0].jd_id,
                resolution_method=ResolutionMethod.UNIQUE_EXACT_MATCH,
            )

        if len(exact_candidates) > 1:
            return TargetMatchResult(
                target_id=target_id,
                requested_job_title=exact_intent.raw_title,
                status=TargetMatchStatus.NEEDS_CONFIRMATION,
                source=TargetMatchSource.EXACT,
                candidates=exact_candidates,
                requires_human_confirmation=True,
                warnings=["同名岗位匹配到多个 JD，请人工选择"],
                resolution_method=ResolutionMethod.HUMAN_CONFIRMED,
            )

        return self._run_semantic_fallback(
            resume=resume,
            target_id=target_id,
            requested_job_title=exact_intent.raw_title,
            top_k_per_query=top_k_per_query,
            recall_top_n=recall_top_n,
            rerank_top_n=rerank_top_n,
        )

    def _run_semantic_fallback(
        self,
        *,
        resume: ResumeProcessingResult,
        target_id: str,
        requested_job_title: str,
        top_k_per_query: int | None,
        recall_top_n: int | None,
        rerank_top_n: int | None,
    ) -> TargetMatchResult:
        try:
            semantic_result = self._semantic_matching_service.match(
                resume=resume.resume,
                requested_job_title=requested_job_title,
                target_id=target_id,
                top_k_per_query=top_k_per_query,
                recall_top_n=recall_top_n,
                rerank_top_n=rerank_top_n,
            )
        except Exception as exc:
            logger.exception("岗位语义降级失败: target_id=%s", target_id)
            return TargetMatchResult(
                target_id=target_id,
                requested_job_title=requested_job_title,
                status=TargetMatchStatus.NO_CANDIDATE,
                warnings=[f"语义匹配执行失败: {type(exc).__name__}"],
            )

        candidates = [
            MatchCandidate(
                jd_id=candidate.jd_id,
                job_title=candidate.job_title,
                department=candidate.department,
                source=TargetMatchSource.SEMANTIC,
                candidate_rank=candidate.rerank_rank,
                score=candidate.resolution_score,
            )
            for candidate in semantic_result.rerank_result.candidates
        ]
        if not candidates:
            return TargetMatchResult(
                target_id=target_id,
                requested_job_title=requested_job_title,
                status=TargetMatchStatus.NO_CANDIDATE,
                semantic_result=semantic_result,
                warnings=self._deduplicate(semantic_result.warnings),
            )

        return TargetMatchResult(
            target_id=target_id,
            requested_job_title=requested_job_title,
            status=TargetMatchStatus.NEEDS_CONFIRMATION,
            source=TargetMatchSource.SEMANTIC,
            candidates=candidates,
            requires_human_confirmation=True,
            semantic_result=semantic_result,
            warnings=self._deduplicate(
                ["语义候选必须经人工确认后才能进入 Agent", *semantic_result.warnings]
            ),
        )

    def _normalize_intents(
        self,
        resume: ResumeProcessingResult,
    ) -> JobIntentNormalizeResult:
        return self._job_intent_normalizer.normalize(
            resume.resume.basic_info.target_job_title
        )

    def _build_result(
            self,
            *,
            exact_result: ExactJobMatchResult,
            targets: list[TargetMatchResult],
            inherited_warnings: Sequence[str] = (),
    ) -> ResumeMatchResult:
        status = self._resolve_overall_status(targets)

        warnings = self._deduplicate(
            [
                *inherited_warnings,
                *(
                    warning
                    for target in targets
                    for warning in target.warnings
                ),
            ]
        )

        return ResumeMatchResult(
            status=status,
            exact_result=exact_result,
            targets=targets,
            analyzable_target_count=sum(
                target.status == TargetMatchStatus.RESOLVED
                for target in targets
            ),
            pending_confirmation_count=sum(
                target.status == TargetMatchStatus.NEEDS_CONFIRMATION
                for target in targets
            ),
            manual_resolution_count=sum(
                target.status == TargetMatchStatus.NEEDS_MANUAL_RESOLUTION
                for target in targets
            ),
            warnings=warnings,
        )

    @staticmethod
    def _deduplicate(values: Sequence[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    @staticmethod
    def _resolve_overall_status(
            targets: Sequence[TargetMatchResult],
    ) -> ResumeMatchStatus:
        resolved_count = sum(
            target.status == TargetMatchStatus.RESOLVED
            for target in targets
        )

        confirmation_count = sum(
            target.status == TargetMatchStatus.NEEDS_CONFIRMATION
            for target in targets
        )

        manual_resolution_count = sum(
            target.status == TargetMatchStatus.NEEDS_MANUAL_RESOLUTION
            for target in targets
        )

        if targets and resolved_count == len(targets):
            return ResumeMatchStatus.READY

        if resolved_count > 0:
            return ResumeMatchStatus.PARTIALLY_READY

        if confirmation_count > 0:
            return ResumeMatchStatus.NEEDS_CONFIRMATION

        if manual_resolution_count > 0:
            return ResumeMatchStatus.NEEDS_MANUAL_RESOLUTION

        return ResumeMatchStatus.BLOCKED