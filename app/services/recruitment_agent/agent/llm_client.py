"""通过 OpenAI 兼容 Chat Completions 接口完成一次单 JD 分析。

默认连接 DeepSeek；使用官方支持的 json_object 模式，再执行本地结构和业务校验。JSON 模式不能保证满足 Schema，更不能保证事实正确。
每次analyze_job最多发起一次SDK请求，重试与退避由服务层/state.py 统一管理。
不读取项目配置或密钥文件；由组合根传入api_key、配置或已有AsyncOpenAI实例。

服务层接入::

    state = state.start_job(jd_id)
    try:
        report = await llm.analyze_job(state.agent_input, jd_id=jd_id)
    except AgentLLMError as exc:
        state = state.fail_job(
            jd_id, stage=exc.stage, error_type=exc.code,
            message=str(exc), retryable=exc.retryable,
        )
    else:
        state = state.complete_job(jd_id, report)

retry_pending 的等待、退避与再次 start_job 由调用方安排；不要额外套用SDK重试。
取消异常原样传播。进程中断后running状态的恢复策略也由调度层负责。
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from typing import Literal

from openai import (
    APIConnectionError, APIError, APIResponseValidationError, APIStatusError,
    APITimeoutError, AsyncOpenAI,
)
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.services.middle_layer.models import AgentAnalysisInput
from app.services.recruitment_agent.schema.output_schema import AgentAnalysisOutput, JobAnalysis
from app.services.recruitment_agent.agent.prompts import (
    PROMPT_VERSION, RUBRIC_VERSION, build_system_prompt, build_user_prompt,
    select_job_input, validate_scoring_rules,
)


logger = logging.getLogger(__name__)
ErrorStage = Literal["input_prepare", "llm_call", "output_parse", "output_validate"]


class AgentLLMConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, allow_inf_nan=False,
                              str_strip_whitespace=True)

    model: str = Field(default="deepseek-v4-flash", min_length=1)
    base_url: str = Field(default="https://api.deepseek.com", min_length=1)
    timeout_seconds: float = Field(default=150.0, gt=0, description="单次API调用总等待上限")
    max_tokens: int = Field(default=15000, ge=1)
    temperature: float = Field(default=0.0, ge=0, le=2)


class AgentLLMError(RuntimeError):
    """可供 state.fail_job 消费的脱敏异常，不包含原始响应或简历正文。"""

    def __init__(
        self, message: str, *, stage: ErrorStage, code: str, retryable: bool,
        status_code: int | None = None, retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.code = code
        self.retryable = retryable
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


class RecruitmentLLMClient:
    """可跨多个JD复用的异步客户端；自行创建的SDK实例需通过aclose关闭。"""

    def __init__(self, *, api_key: str | None = None, config: AgentLLMConfig | None = None,client: AsyncOpenAI | None = None) -> None:

        self.config = config or AgentLLMConfig()

        if client is not None and api_key is not None:
            raise ValueError("传入client时不要同时传入api_key")
        if client is None and (not isinstance(api_key, str) or not api_key.strip()):
            raise ValueError("未传入client时必须显式提供非空api_key")

        self._owns_client = client is None
        self._client = client if client is not None else AsyncOpenAI(
            api_key=api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
            max_retries=0,
        )

        # 注入的SDK也按请求关闭内部重试，避免绕过状态机中的重试上限。
        # 注入时沿用该实例的 endpoint/凭据，config.base_url 不覆盖它。
        self._request_client = self._client.with_options(max_retries=0, timeout=self.config.timeout_seconds)
        self._closed = False

    async def analyze_job(self, agent_input: AgentAnalysisInput, *, jd_id: int) -> JobAnalysis:
        """返回已验证的单JD报告；本方法不更改state，不执行自动修复或重试。"""
        if self._closed:
            raise AgentLLMError("客户端已关闭", stage="llm_call", code="client_closed", retryable=False)
        started = time.perf_counter()
        try:
            try:
                job_input = select_job_input(agent_input, jd_id)
                user_prompt = build_user_prompt(job_input)
            except (ValueError, TypeError):
                raise AgentLLMError("标准输入无效或JD未绑定", stage="input_prepare",code="invalid_input", retryable=False) from None

            response = await self._request(build_system_prompt(), user_prompt)
            report = self._parse_report(response)
            try:
                AgentAnalysisOutput(
                    analysis_scope=job_input.analysis_scope, job_analyses=[report],
                ).validate_against_input(job_input)
                validate_scoring_rules(report)
            except ValueError:
                raise AgentLLMError("报告未通过岗位绑定、证据引用或评分规则校验", stage="output_validate",code="report_validation_failed", retryable=True) from None
        except AgentLLMError as exc:
            logger.warning(
                "Agent分析失败 jd_id=%s stage=%s code=%s retryable=%s status=%s elapsed_ms=%.0f",
                jd_id, exc.stage, exc.code, exc.retryable, exc.status_code,
                (time.perf_counter() - started) * 1000,
            )
            raise

        usage = getattr(response, "usage", None)
        logger.info(
            "Agent分析完成 jd_id=%s model=%s prompt=%s rubric=%s elapsed_ms=%.0f total_tokens=%s",
            jd_id, self.config.model, PROMPT_VERSION, RUBRIC_VERSION,
            (time.perf_counter() - started) * 1000, getattr(usage, "total_tokens", None),
        )
        return report

    async def _request(self, system_prompt: str, user_prompt: str) -> ChatCompletion:
        try:
            # SDK的HTTP超时限制单次I/O等待；wait_for另外约束整个请求的等待时间。
            return await asyncio.wait_for(
                self._request_client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": user_prompt}],
                    response_format={"type": "json_object"},
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    stream=False,
                ),
                timeout=self.config.timeout_seconds,
            )
        except (APITimeoutError, asyncio.TimeoutError):
            raise AgentLLMError("模型请求超时", stage="llm_call", code="timeout", retryable=True) from None

        except APIConnectionError:
            raise AgentLLMError("无法连接模型服务", stage="llm_call", code="connection_error", retryable=True) from None

        except APIStatusError as exc:
            status = exc.status_code
            raise AgentLLMError(
                f"模型服务返回HTTP {status}", stage="llm_call", code=f"http_{status}",
                retryable=status in {408, 409, 429} or status >= 500,
                status_code=status, retry_after_seconds=_retry_after(exc),
            ) from None

        except (APIResponseValidationError, json.JSONDecodeError, UnicodeDecodeError):
            raise AgentLLMError("模型服务响应格式异常", stage="llm_call", code="invalid_api_response", retryable=True) from None

        except APIError:
            raise AgentLLMError("模型SDK调用失败", stage="llm_call", code="sdk_error", retryable=False) from None

    @staticmethod
    def _parse_report(response: ChatCompletion) -> JobAnalysis:

        choices = getattr(response, "choices", None)

        if not isinstance(choices, list) or len(choices) != 1:
            raise AgentLLMError("模型响应缺少唯一的分析结果", stage="output_parse",code="invalid_choices", retryable=True)

        choice = choices[0]
        message = getattr(choice, "message", None)
        finish_reason = getattr(choice, "finish_reason", None)

        if getattr(message, "refusal", None) or finish_reason == "content_filter":
            raise AgentLLMError("模型未提供本次分析", stage="llm_call", code="model_refusal", retryable=False)

        if finish_reason == "length":
            raise AgentLLMError("模型输出被截断，请调整max_tokens或输入长度后重新执行",stage="output_parse", code="output_truncated", retryable=False)

        if finish_reason != "stop" or getattr(message, "tool_calls", None):
            raise AgentLLMError("模型未正常完成JSON输出", stage="output_parse",code="unexpected_finish", retryable=True)

        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise AgentLLMError("模型返回空内容", stage="output_parse", code="empty_output", retryable=True)

        try:
            # 不提取围栏、不拼接截断JSON、不将宽松修复后的内容当作有效报告。
            return JobAnalysis.model_validate_json(content)

        except ValidationError as exc:
            invalid_json= any(
                item["type"] == "json_invalid"
                for item in exc.errors(include_input=False, include_context=False)
            )
            raise AgentLLMError(
                "模型输出不是合法 JSON" if invalid_json else "模型输出不符合 JobAnalysis 结构",
                stage="output_parse" if invalid_json else "output_validate",
                code="invalid_json" if invalid_json else "invalid_schema", retryable=True,
            ) from None

    async def aclose(self) -> None:
        """只关闭本类创建的SDK；注入实例由调用方管理生命周期。"""
        if not self._closed:
            if self._owns_client:
                await self._client.close()
            self._closed = True

    async def __aenter__(self) -> RecruitmentLLMClient:
        if self._closed:
            raise RuntimeError("客户端已关闭")
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.aclose()


def _retry_after(exc: APIStatusError) -> float | None:
    """读取秒数形式的 Retry-After，供外层退避使用；未识别的格式返回 None。"""
    try:
        seconds = float(exc.response.headers.get("retry-after", ""))
    except (ValueError, TypeError):
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


__all__ = ["AgentLLMConfig", "AgentLLMError", "RecruitmentLLMClient"]
