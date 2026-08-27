from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from resume_extractor.Resume_schema import ResumeModel
from app.services.resume_handle.document_parser.base import (
    DocumentParser,
    DocumentParseResult,
)
from app.services.resume_handle.resume_extractor.base import (
    ResumeExtractor,
)


@dataclass
class ResumeProcessingResult:
    """
    一次完整简历处理的最终结果。
    """

    resume: ResumeModel

    document: DocumentParseResult


class ResumeProcessingService:

    def __init__(
        self,
        document_parser: DocumentParser,
        resume_extractor: ResumeExtractor,
    ):
        self.document_parser = document_parser
        self.resume_extractor = resume_extractor

    async def process(
        self,
        file_path: str | Path,
    ) -> ResumeProcessingResult:
        """
        完整简历处理流程：

        PDF
        -> MinerU
        -> Markdown
        -> LLM
        -> ResumeSchema
        """

        # -----------------------------
        # Step 1
        # PDF -> Markdown
        # -----------------------------

        document_result = await asyncio.to_thread(
            self.document_parser.parse,
            file_path,
        )

        markdown = document_result.markdown

        if not markdown.strip():
            raise ValueError(
                "文档解析成功，但 Markdown 内容为空"
            )

        # -----------------------------
        # Step 2
        # Markdown -> ResumeSchema
        # -----------------------------
        print("开始抽取简历结构化信息...",f"{markdown}")
        resume = await self.resume_extractor.extract(
            markdown
        )
        print("抽取简历结构化信息完成...")
        # -----------------------------
        # Step 3
        # 返回完整处理结果
        # -----------------------------

        return ResumeProcessingResult(
            resume=resume,
            document=document_result,
        )