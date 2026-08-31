from __future__ import annotations

import json
import hashlib
import logging
import time


from openai import AsyncOpenAI
from pydantic import ValidationError
from pathlib import Path

from app.core.config import DEEPSEEK_API_KEY
from .base import ResumeExtractor
from .resume_schema import ResumeModel

logger = logging.getLogger(__name__)
path = Path(__file__).resolve().parent
CACHE_PATH = path / "resume_shcema_json_cache"

class ResumeExtractionError(Exception):
    """简历结构化抽取失败。"""

class DeepSeekResumeExtractor(ResumeExtractor):

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-v4-flash",
        cache_dir: str = CACHE_PATH,
    ):
        self.model = model

        self.client = AsyncOpenAI(
            api_key= api_key or DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
        )

        self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    async def _extract_with_llm(self, markdown: str) -> ResumeModel:

        try:
            start = time.perf_counter()

            response = await self.client.responses.create(
                model=self.model,

                reasoning={"effort": "none"},

                instructions=self._build_system_prompt(),

                input=self._build_user_prompt(
                    markdown
                ),

                text={
                    "format": {
                        "type": "json_schema",
                        "name": "resume",
                        "schema": (
                            ResumeModel
                            .model_json_schema()
                        ),
                    }
                },

                max_output_tokens=16000,
            )

            elapsed_ms = (
                                 time.perf_counter() - start
                         ) * 1000

            logger.info(
                "LLM请求成功 model=%s latency_ms=%.0f",
                self.model,
                elapsed_ms,
            )

        except Exception as exc:
            raise ResumeExtractionError(
                f"DeepSeek API 调用失败: {exc}"
            ) from exc

        content = response.output_text
        content = self.normalize_json_output(content)

        if response.status == "failed":
            raise ResumeExtractionError(
                f"DeepSeek 请求失败: {response.error}"
            )

        if response.status == "incomplete":
            raise ResumeExtractionError(
                f"DeepSeek 输出不完整: {response.incomplete_details}"
            )

        if not content or not content.strip():
            raise ResumeExtractionError(
                "DeepSeek 请求 completed，但 output_text 为空"
            )

        try:
            data = json.loads(content)

        except json.JSONDecodeError as exc:
            raise ResumeExtractionError("DeepSeek 返回结果不是合法 JSON") from exc

        try:
            resume = ResumeModel.model_validate(
                data
            )

        except ValidationError as exc:
            raise ResumeExtractionError(
                f"ResumeSchema 校验失败: {exc}"
            ) from exc

        return resume

    @staticmethod
    def _load_cache(
            cache_path: Path,
    ) -> ResumeModel:

        content = cache_path.read_text(
            encoding="utf-8"
        )

        return ResumeModel.model_validate_json(
            content
        )

    @staticmethod
    def _save_cache(
            cache_path: Path,
            resume: ResumeModel,
    ) -> None:

        cache_path.write_text(
            resume.model_dump_json(
                indent=2
            ),
            encoding="utf-8",
        )

    @staticmethod
    def normalize_json_output(content: str) -> str:
        content = content.strip()

        if content.startswith("```json"):
            content = content[len("```json"):]

        elif content.startswith("```"):
            content = content[len("```"):]

        if content.endswith("```"):
            content = content[:-3]

        return content.strip()

    @staticmethod
    def _build_system_prompt() -> str:
        return """
    你是一个简历结构化信息抽取器。

    你的任务是根据用户提供的 Markdown 简历原文，
    抽取其中明确存在的信息，并按照 API 指定的 JSON Schema
    返回一个 JSON 对象。

    要求：

    1. 只能使用简历原文明确存在的信息。
    2. 禁止推测、补充或润色候选人的经历和能力。
    3. 不确定或不存在的可空字段返回 null。
    4. 不存在的列表返回 []。
    5. 教育经历、工作经历和项目经历必须正确区分。
    6. 需要保留原文的字段不得改写。
    7. 只输出符合指定 Schema 的 JSON 对象。
    8. 不要输出 Markdown 代码块。
    9. 不要在 JSON 前后添加解释文字。
    """

    @staticmethod
    def _build_user_prompt(markdown: str) -> str:
        return f"""
    下面是一份经过文档解析得到的 Markdown 简历。

    请严格根据原文进行结构化信息抽取。

    <resume_markdown>
    {markdown}
    </resume_markdown>
    """

    async def extract(
        self,
        markdown: str,
        *,
        force_refresh: bool = False
    ) -> ResumeModel:
        """
        将 document_parser 解析得到的 Markdown
        抽取为 ResumeSchema。
        """

        if not markdown.strip():
            raise ResumeExtractionError("Markdown 内容为空")

        markdown_hash = hashlib.sha256(markdown.encode("utf-8")).hexdigest()

        cache_path = (
                self.cache_dir
                / f"{markdown_hash}.json"
        )

        if (
                not force_refresh
                and cache_path.exists()
        ):
            try:
                resume = self._load_cache(cache_path)
                logger.info("缓存文件加载完成")
                return resume
            except ValidationError as e:
                raise ResumeExtractionError(f"缓存文件 {cache_path} 格式校验失败: {e}")

        resume = await self._extract_with_llm(markdown)

        self._save_cache(cache_path,resume)
        return resume