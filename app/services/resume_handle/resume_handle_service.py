from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from .resume_extractor import ResumeExtractor, ResumeModel
from .document_parser import DocumentParser,DocumentParseResult

logger = logging.getLogger(__name__)

@dataclass
class ResumeProcessingResult:
    """
    一次完整简历处理的最终结果。
    """

    resume: ResumeModel

    document: DocumentParseResult


class ResumeProcessingService:

    def __init__(self,document_parser: DocumentParser,resume_extractor: ResumeExtractor):
        self.document_parser = document_parser
        self.resume_extractor = resume_extractor

    async def process(self,file_path: str | Path) -> ResumeProcessingResult:
        """
        完整简历处理流程：PDF-> MinerU-> Markdown-> LLM-> ResumeSchema
        """

        # -----------------------------
        # Step 1
        # PDF -> Markdown
        # -----------------------------
        print("*" * 50)
        try:
            logger.debug("开始处理简历...")

            start = time.perf_counter()
            document_result = await asyncio.to_thread(
                self.document_parser.parse,
                file_path,
            )

            markdown = document_result.markdown

            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info("文档解析成功, 耗时latency_ms=%.0f",elapsed_ms)

            if not markdown.strip():
                raise ValueError("文档解析成功，但 Markdown 内容为空")

        except Exception as e:
            logger.exception(f"处理简历{file_path}时出错: {e}")
            raise
        print("*" * 50)

        # -----------------------------
        # Step 2
        # Markdown -> ResumeSchema
        # ----------------------------

        print()
        print("*" * 50)

        try:
            start = time.perf_counter()

            resume = await self.resume_extractor.extract(markdown)

            elapsed_ms = (time.perf_counter() - start) * 1000

            logger.info("LLM抽取简历结构化信息成功, 耗时latency_ms=%.0f",elapsed_ms)

        except Exception as exc:
            logger.exception(f"LLM抽取简历结构化信息失败: {exc}")
            raise
        print("*" * 50)
        print()

        return ResumeProcessingResult(resume=resume,document=document_result)
