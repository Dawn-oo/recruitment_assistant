from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from app.services.resume_handle.resume_extractor.base import ResumeExtractor
from app.services.resume_handle.resume_extractor.resume_schema import ResumeModel

from app.services.resume_handle.document_parser.base import (
    DocumentParser,
    DocumentParseResult)

logger = logging.getLogger(__name__)

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
        print("*" * 50)
        logger.info("开始处理简历...")

        start = time.perf_counter()
        document_result = await asyncio.to_thread(
            self.document_parser.parse,
            file_path,
        )

        markdown = document_result.markdown
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "文档解析成功, 耗时latency_ms=%.0f",
            elapsed_ms,
        )

        if not markdown.strip():
            raise ValueError("文档解析成功，但 Markdown 内容为空")

        print("*" * 50)
        # -----------------------------
        # Step 2
        # Markdown -> ResumeSchema
        # ----------------------------
        print()
        print("*" * 50)
        logger.info("开始抽取简历结构化信息...")
        resume = await self.resume_extractor.extract(markdown)
        logger.info("抽取简历结构化信息完成...")
        print("*" * 50)
        print()
        # -----------------------------
        # Step 3
        # 返回完整处理结果
        # -----------------------------

        return ResumeProcessingResult(
            resume=resume,
            document=document_result,
        )