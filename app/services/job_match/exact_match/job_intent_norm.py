from __future__ import annotations

import re
from pydantic import BaseModel, ConfigDict, Field


class JobIntentNormalizeResult(BaseModel):
    """
    Resume 中求职岗位经过基础清洗和多岗位拆分后的结果,主要用于对申请岗位名称进行标准化处理。
    """

    model_config = ConfigDict(
        strict=True,
        extra="ignore",
    )

    raw_target_job_title: str | None = Field(
        default=None,
        description="简历中原始的求职岗位文本"
    )

    job_titles: list[str] = Field(
        default_factory=list,
        description="拆分并完成基础清洗后的岗位名称"
    )

    is_multi_intent: bool = Field(
        default=False,
        description="是否识别出多个独立岗位意图"
    )


class JobIntentNormalizer:
    """
    求职岗位预处理器。

    负责：
        1. 基础字符串清理
        2. 多岗位拆分
        3. 去重

    """

    # 这些符号比较明确地表示多个岗位。
    STRONG_SEPARATOR_PATTERN = re.compile(
        r"[、，,；;|]+"
    )

    # "/" 单独处理，因为它可能出现在：实施工程师/项目经理
    # 也可能出现在：Java/C++开发工程师

    SLASH_PATTERN = re.compile(
        r"[/／]+"
    )

    def normalize(
        self,
        target_job_title: str | None,
    ) -> JobIntentNormalizeResult:

        # =====================================================
        # 1. None / 空字符串
        # =====================================================

        if target_job_title is None:
            return JobIntentNormalizeResult(
                raw_target_job_title=None,
                job_titles=[],
                is_multi_intent=False,
            )

        cleaned = self._clean_title(
            target_job_title
        )

        if not cleaned:
            return JobIntentNormalizeResult(
                raw_target_job_title=target_job_title,
                job_titles=[],
                is_multi_intent=False,
            )

        # =====================================================
        # 2. 强分隔符
        # =====================================================

        parts = self._split_by_strong_separator(
            cleaned
        )

        if len(parts) > 1:
            return JobIntentNormalizeResult(
                raw_target_job_title=target_job_title,
                job_titles=self._deduplicate(parts),
                is_multi_intent=True,
            )

        # =====================================================
        # 3. "/"
        # =====================================================

        slash_parts = self._split_by_slash(
            cleaned
        )

        if self._should_split_slash(
            cleaned,
            slash_parts,
        ):
            return JobIntentNormalizeResult(
                raw_target_job_title=target_job_title,
                job_titles=self._deduplicate(
                    slash_parts
                ),
                is_multi_intent=True,
            )

        # =====================================================
        # 4. 单岗位
        # =====================================================

        return JobIntentNormalizeResult(
            raw_target_job_title=target_job_title,
            job_titles=[cleaned],
            is_multi_intent=False,
        )

    # =========================================================
    # Split
    # =========================================================

    def _split_by_strong_separator(
        self,
        title: str,
    ) -> list[str]:

        parts = self.STRONG_SEPARATOR_PATTERN.split(
            title
        )

        return self._clean_parts(parts)

    def _split_by_slash(
        self,
        title: str,
    ) -> list[str]:

        parts = self.SLASH_PATTERN.split(
            title
        )

        return self._clean_parts(parts)

    # =========================================================
    # Slash heuristic
    # =========================================================

    @staticmethod
    def _should_split_slash(
        original: str,
        parts: list[str],
    ) -> bool:
        """
        对 "/" 做保守判断。

        第一版只处理比较明显的多岗位情况。

        可以拆：
            项目经理/实施工程师
            销售经理/渠道销售

        暂时不拆：
            Java/C++开发工程师
            Python/C++工程师

        这里不依赖 JD 数据库，因为数据库判断属于
        ExactJobMatcher / Repository 上层逻辑。

        第一版启发式：
        如果拆分后的每一部分都像完整岗位名称，则拆分。

        “完整岗位名称”主要根据常见岗位后缀判断。
        """

        if len(parts) <= 1:
            return False

        job_suffixes = (
            "经理",
            "工程师",
            "主管",
            "专员",
            "顾问",
            "助理",
            "组长",
            "管理员",
            "管理",
            "代表",
            "会计",
            "出纳",
            "监理",
            "文秘",
        )

        return all(
            part.endswith(job_suffixes)
            for part in parts
        )

    # =========================================================
    # Clean
    # =========================================================

    @staticmethod
    def _clean_title(
        title: str,
    ) -> str:
        """
        只做不会改变岗位语义的清理。
        """

        title = title.strip()

        # 连续空白折叠
        title = re.sub(
            r"\s+",
            " ",
            title,
        )

        return title

    @classmethod
    def _clean_parts(
        cls,
        parts: list[str],
    ) -> list[str]:

        result: list[str] = []

        for part in parts:
            cleaned = cls._clean_title(
                part
            )

            if cleaned:
                result.append(cleaned)

        return result

    # =========================================================
    # Deduplicate
    # =========================================================

    @staticmethod
    def _deduplicate(
        titles: list[str],
    ) -> list[str]:

        result: list[str] = []
        seen: set[str] = set()

        for title in titles:
            if title in seen:
                continue

            seen.add(title)
            result.append(title)

        return result