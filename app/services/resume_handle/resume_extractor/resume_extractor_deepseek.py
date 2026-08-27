from __future__ import annotations

import json

from openai import AsyncOpenAI
from pydantic import ValidationError

from app.core.config import DEEPSEEK_API_KEY
from app.services.resume_handle.resume_extractor.Resume_schema import ResumeModel
from app.services.resume_handle.resume_extractor.base import ResumeExtractor

class ResumeExtractionError(Exception):
    """简历结构化抽取失败。"""


class DeepSeekResumeExtractor(ResumeExtractor):

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-v4-flash",
    ):
        self.model = model

        self.client = AsyncOpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
        )

    async def extract(
        self,
        markdown: str,
    ) -> ResumeModel:
        """
        将 document_parser 解析得到的 Markdown
        抽取为 ResumeSchema。
        """

        if not markdown.strip():
            raise ResumeExtractionError(
                "Markdown 内容为空"
            )

        try:
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

        except Exception as exc:
            raise ResumeExtractionError(
                f"DeepSeek API 调用失败: {exc}"
            ) from exc

        content = self.normalize_json_output(response.output_text)
        print("status:", response.status)
        print("error:", response.error)
        print("content:", content)

        if response.status == "failed":
            raise ResumeExtractionError(
                f"DeepSeek 请求失败: {response.error}"
            )

        if response.status == "incomplete":
            raise ResumeExtractionError(
                f"DeepSeek 输出不完整: {response.incomplete_details}"
            )

        content = response.output_text

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

