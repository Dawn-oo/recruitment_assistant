from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class JDTextBuildError(ValueError):
    """数据库 JD 无法转换为重排文本。"""


@dataclass(slots=True, frozen=True)
class JDPassage:
    section: str
    text: str


@dataclass(slots=True, frozen=True)
class RerankJDDocument:
    jd_id: int
    job_title: str
    department: str | None
    title_passage: str
    support_passages: tuple[JDPassage, ...]


class JDTextBuilder:

    """把结构化 JD 数据库记录构造成 BGE reranker 文本。

    支持度阶段不把整份JD粗暴拼成一个超长 passage，而是按资格要求、
    岗位职责和胜任能力分段。业务层会对同一简历 Query 取最佳分段得分。
    """

    def __init__(self, *, max_chars_per_passage: int = 1200) -> None:
        if max_chars_per_passage <= 0:
            raise ValueError("max_chars_per_passage 必须大于0")
        self._max_chars_per_passage = max_chars_per_passage

    def build(self, row: Mapping[str, Any]) -> RerankJDDocument:

        if not isinstance(row, Mapping):
            raise JDTextBuildError("JD 记录必须是 Mapping")

        jd_id = row.get("id")
        if not isinstance(jd_id, int):
            raise JDTextBuildError(f"JD 记录缺少有效整数 id: {jd_id!r}")

        job_title = self._clean_scalar(row.get("job_title"))
        if not job_title:
            raise JDTextBuildError(f"JD 记录缺少岗位名称: jd_id={jd_id}")
        department = self._clean_scalar(row.get("department")) or None

        header = f"公司岗位名称：{job_title}"
        if department:
            header += f"\n所属部门：{department}"

        passages: list[JDPassage] = []
        qualification_lines = self._build_qualification_lines(row)
        if qualification_lines:
            passages.extend(
                self._batch_lines(
                    header=header,
                    section_prefix="qualification",
                    lines=qualification_lines,
                )
            )

        responsibilities = row.get("responsibilities")
        if isinstance(responsibilities, Sequence) and not isinstance(
            responsibilities, (str, bytes)
        ):
            for index, responsibility in enumerate(responsibilities):
                lines = self._build_responsibility_lines(responsibility)
                if lines:
                    passages.extend(
                        self._batch_lines(
                            header=header,
                            section_prefix=f"responsibility_{index}",
                            lines=lines,
                        )
                    )

        competency_lines = [
            f"胜任能力：{text}"
            for text in self._clean_sequence(row.get("competencies"))
        ]
        if competency_lines:
            passages.extend(
                self._batch_lines(
                    header=header,
                    section_prefix="competency",
                    lines=competency_lines,
                )
            )

        if not passages:
            passages.append(JDPassage(section="title_only", text=header))

        return RerankJDDocument(
            jd_id=jd_id,
            job_title=job_title,
            department=department,
            title_passage=header,
            support_passages=tuple(passages),
        )

    def _build_qualification_lines(
        self,
        row: Mapping[str, Any],
    ) -> list[str]:
        fields = (
            ("最低学历", row.get("minimum_education")),
            ("专业背景", row.get("education_background")),
            ("工作经验要求", row.get("work_experience_raw")),
        )
        return [
            f"{label}：{text}"
            for label, value in fields
            if (text := self._clean_scalar(value))
        ]

    def _build_responsibility_lines(self, value: Any) -> list[str]:
        if isinstance(value, Mapping):
            lines: list[str] = []
            description = self._clean_scalar(value.get("description"))
            if description:
                lines.append(f"岗位职责：{description}")
            lines.extend(
                f"具体任务：{task}"
                for task in self._clean_sequence(value.get("tasks"))
            )
            return lines
        text = self._clean_scalar(value)
        return [f"岗位职责：{text}"] if text else []

    def _batch_lines(
        self,
        *,
        header: str,
        section_prefix: str,
        lines: Sequence[str],
    ) -> list[JDPassage]:
        passages: list[JDPassage] = []
        current: list[str] = []

        def flush() -> None:
            if not current:
                return
            text = "\n".join([header, *current])
            part_index = len(passages)
            passages.append(
                JDPassage(
                    section=f"{section_prefix}_part_{part_index}",
                    text=text[: self._max_chars_per_passage].rstrip(),
                )
            )
            current.clear()

        for raw_line in lines:
            line = raw_line[: self._max_chars_per_passage].rstrip()
            proposed = "\n".join([header, *current, line])
            if current and len(proposed) > self._max_chars_per_passage:
                flush()
            current.append(line)
        flush()
        return passages

    @staticmethod
    def _clean_scalar(value: Any) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            return ""
        return " ".join(value.strip().split())

    @classmethod
    def _clean_sequence(cls, value: Any) -> list[str]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return []
        result: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = cls._clean_scalar(item)
            if text and text not in seen:
                seen.add(text)
                result.append(text)
        return result
