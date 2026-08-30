from __future__ import annotations

import re
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field

from job_alias_dict import JOB_TITLE_ALIASES,STANDARD_JOB_TITLES

class JobIntentResolutionType(str, Enum):
    EXACT = "exact"
    ALIAS = "alias"
    UNRESOLVED = "unresolved"


class NormalizedJobIntent(BaseModel):

    model_config = ConfigDict(
        strict=True,
        extra="ignore",
    )

    raw_title: str

    normalized_title: str | None = None

    resolution_type: JobIntentResolutionType


class JobIntentNormalizeResult(BaseModel):

    model_config = ConfigDict(
        strict=True,
        extra="ignore",
    )

    raw_target_job_title: str | None = None

    intents: list[NormalizedJobIntent] = Field(
        default_factory=list
    )

    is_multi_intent: bool = False


class JobIntentNormalizer:
    """
    将 Resume 中的原始 target_job_title 转换为
    后续岗位检索可以使用的多个岗位意图。

    职责：
    1. 基础字符串清理
    2. 尝试识别多岗位
    3. 标准岗位名称匹配
    4. Alias 标准化
    5. 无法确认时保留原文

    不负责：
    - 查询数据库
    - 向量检索
    - JD 匹配评分
    """

    # 第一版只处理比较明确的岗位分隔符。
    # "/" 需要额外判断，避免把 Java/C++开发工程师错误拆开。
    STRONG_SEPARATORS_PATTERN = re.compile(
        r"[、，,；;|]+"
    )

    SLASH_PATTERN = re.compile(r"[/／]+")

    def __init__(
        self,
        known_job_titles: set[str]| None = None,
        aliases: dict[str, str] | None = None,
    ) -> None:

        if not aliases or not known_job_titles:
            aliases = JOB_TITLE_ALIASES
            known_job_titles = STANDARD_JOB_TITLES

        self.known_job_titles = {
            self._clean_title(title)
            for title in known_job_titles
            if title and title.strip()
        }

        self.aliases = {
            self._clean_title(alias): self._clean_title(target)
            for alias, target in (aliases or {}).items()
        }

    def normalize(
        self,
        target_job_title: str | None,
    ) -> JobIntentNormalizeResult:

        if not target_job_title:
            return JobIntentNormalizeResult(
                raw_target_job_title=None,
                intents=[],
                is_multi_intent=False,
            )

        cleaned_title = self._clean_title(
            target_job_title
        )

        if not cleaned_title:
            return JobIntentNormalizeResult(
                raw_target_job_title=target_job_title,
                intents=[],
                is_multi_intent=False,
            )

        # =====================================================
        # 1. 先处理明确分隔符
        # =====================================================

        strong_parts = self._split_by_strong_separators(
            cleaned_title
        )

        if len(strong_parts) > 1:
            intents = [
                self._normalize_single_title(part)
                for part in strong_parts
            ]

            return JobIntentNormalizeResult(
                raw_target_job_title=target_job_title,
                intents=self._deduplicate(intents),
                is_multi_intent=True,
            )

        # =====================================================
        # 2. 再尝试处理 /
        # =====================================================

        slash_parts = self._split_by_slash(
            cleaned_title
        )

        if len(slash_parts) > 1:
            validated_parts = (
                self._validate_slash_split(
                    slash_parts
                )
            )

            if validated_parts:
                intents = [
                    self._normalize_single_title(part)
                    for part in validated_parts
                ]

                return JobIntentNormalizeResult(
                    raw_target_job_title=(
                        target_job_title
                    ),
                    intents=self._deduplicate(intents),
                    is_multi_intent=True,
                )

        # =====================================================
        # 3. 无法确认是多岗位 -> 整体作为一个岗位
        # =====================================================

        intent = self._normalize_single_title(
            cleaned_title
        )

        return JobIntentNormalizeResult(
            raw_target_job_title=target_job_title,
            intents=[intent],
            is_multi_intent=False,
        )

    # =========================================================
    # 单岗位标准化
    # =========================================================

    def _normalize_single_title(
        self,
        raw_title: str,
    ) -> NormalizedJobIntent:

        cleaned = self._clean_title(raw_title)

        # 1. 已经是标准 JD 岗位名
        if cleaned in self.known_job_titles:
            return NormalizedJobIntent(
                raw_title=cleaned,
                normalized_title=cleaned,
                resolution_type=JobIntentResolutionType.EXACT,
            )

        # 2. Alias
        alias_target = self.aliases.get(cleaned)

        if alias_target is not None:
            return NormalizedJobIntent(
                raw_title=cleaned,
                normalized_title=alias_target,
                resolution_type=JobIntentResolutionType.ALIAS,
            )

        # 3. 无法标准化，保留原始岗位名称
        return NormalizedJobIntent(
            raw_title=cleaned,
            normalized_title=cleaned,
            resolution_type=JobIntentResolutionType.UNRESOLVED,
        )

    # =========================================================
    # 多岗位拆分
    # =========================================================

    def _split_by_strong_separators(
        self,
        title: str,
    ) -> list[str]:

        parts = self.STRONG_SEPARATORS_PATTERN.split(
            title
        )

        return self._clean_parts(parts)

    def _split_by_slash(
        self,
        title: str,
    ) -> list[str]:

        parts = self.SLASH_PATTERN.split(title)

        return self._clean_parts(parts)

    def _validate_slash_split(
        self,
        parts: list[str],
    ) -> list[str] | None:
        """
        "/" 比较危险。

        例如：
            实施工程师/项目经理
        很可能是两个岗位。

        但是：
            Java/C++开发工程师
        不应该简单拆成：
            Java
            C++开发工程师

        第一版采用保守策略：

        只有拆分后的每个部分都能通过
        exact 或 alias 识别时，才确认它们是多个岗位。

        否则保留整个原始字符串，
        留给后面的语义检索处理。
        """

        if not parts:
            return None

        for part in parts:
            if not self._is_known_title(part):
                return None

        return parts

    def _is_known_title(
        self,
        title: str,
    ) -> bool:

        cleaned = self._clean_title(title)

        return (
            cleaned in self.known_job_titles
            or cleaned in self.aliases
        )

    # =========================================================
    # 基础文本处理
    # =========================================================

    @staticmethod
    def _clean_title(
        title: str,
    ) -> str:
        """
        这里只做非常保守的清理，
        不修改岗位原本的语义。
        """

        title = title.strip()

        # 连续空白变成一个空格
        title = re.sub(
            r"\s+",
            " ",
            title,
        )

        return title

    @staticmethod
    def _clean_parts(
        parts: list[str],
    ) -> list[str]:

        result: list[str] = []

        for part in parts:
            cleaned = part.strip()

            if cleaned:
                result.append(cleaned)

        return result

    # =========================================================
    # 去重
    # =========================================================

    @staticmethod
    def _deduplicate(
        intents: list[NormalizedJobIntent],
    ) -> list[NormalizedJobIntent]:

        result: list[NormalizedJobIntent] = []
        seen: set[str] = set()

        for intent in intents:
            key = intent.normalized_title

            if key in seen:
                continue

            seen.add(key)
            result.append(intent)

        return result