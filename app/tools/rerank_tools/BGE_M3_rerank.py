from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


class RerankerError(RuntimeError):
    """BGE-M3 重排模型加载或推理失败。"""


class BgeReranker:
    """基于 ``FlagEmbedding.FlagReranker`` 的本地 BGE-M3 重排适配器。

    默认模型为 ``BAAI/bge-reranker-v2-m3``。建议生产环境将模型提前下载
    到本地目录，并通过 ``BGE_RERANKER_MODEL_PATH`` 指向该目录。

    模型采用懒加载：构造对象不会立刻占用显存或内存，第一次调用
    :meth:`score_pairs` 或 :meth:`warmup` 时才加载权重。
    """

    DEFAULT_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
        use_fp16: bool | None = None,
        lazy_load: bool = True,
    ) -> None:
        resolved_model_path = model_path or os.getenv("BGE_RERANKER_MODEL_PATH",self.DEFAULT_MODEL_NAME)
        resolved_device = device or os.getenv("BGE_RERANKER_DEVICE") or None
        if resolved_device == "auto":
            resolved_device = None

        self._model_path = str(resolved_model_path)
        self._device = resolved_device
        self._batch_size = (
            batch_size
            if batch_size is not None
            else self._read_positive_int_env(
                "BGE_RERANKER_BATCH_SIZE",
                default=8,
            )
        )
        self._max_length = (
            max_length
            if max_length is not None
            else self._read_positive_int_env(
                "BGE_RERANKER_MAX_LENGTH",
                default=512,
            )
        )
        self._use_fp16 = use_fp16
        self._model: Any | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

        if not self._model_path.strip():
            raise ValueError("model_path 不能为空")
        if self._batch_size <= 0:
            raise ValueError("batch_size 必须大于0")
        if self._max_length <= 0:
            raise ValueError("max_length 必须大于0")
        if self._device == "cpu" and self._use_fp16 is True:
            raise ValueError("CPU 推理不能启用 FP16")

        if not lazy_load:
            self._ensure_model_loaded()

    @property
    def model_name(self) -> str:
        return self._model_path

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def warmup(self) -> None:
        """提前加载模型并执行一次短文本推理。"""
        self.score_pairs([("测试查询", "测试文档")])

    def score_pairs(self,pairs: Sequence[tuple[str, str]]) -> list[float]:
        """批量计算文本对相关性，并使用 sigmoid 映射到 ``[0, 1]``。"""
        normalized_pairs = self._validate_pairs(pairs)
        if not normalized_pairs:
            return []

        model = self._ensure_model_loaded()
        started_at = time.perf_counter()
        try:
            # FlagReranker 内部会根据 batch_size 分批。锁用于防止同一模型实例
            # 被多个请求同时迁移设备或执行推理。
            with self._inference_lock:
                raw_scores = model.compute_score(
                    [list(pair) for pair in normalized_pairs],
                    batch_size=self._batch_size,
                    max_length=self._max_length,
                    normalize=True,
                )
        except Exception as exc:
            raise RerankerError(
                "BGE-M3 重排推理失败: "
                f"model={self._model_path}, pair_count={len(normalized_pairs)}"
            ) from exc

        scores = self._normalize_model_output(
            raw_scores,
            expected_count=len(normalized_pairs),
        )
        logger.info(
            "BGE重排推理完成: model=%s pair_count=%d elapsed_ms=%.2f",
            self._model_path,
            len(scores),
            (time.perf_counter() - started_at) * 1000,
        )
        return scores

    def close(self) -> None:
        """释放 FlagEmbedding 可能建立的多进程池和设备缓存。"""
        model = self._model
        if model is None:
            return
        stop_pool = getattr(model, "stop_self_pool", None)
        if callable(stop_pool):
            stop_pool()
        self._model = None

    def _ensure_model_loaded(self) -> Any:
        if self._model is not None:
            return self._model

        with self._load_lock:
            if self._model is not None:
                return self._model

            started_at = time.perf_counter()
            try:
                from FlagEmbedding import FlagReranker
            except ImportError as exc:
                raise RerankerError(
                    "缺少 FlagEmbedding 依赖，请先执行 "
                    "`pip install -U FlagEmbedding`"
                ) from exc

            use_fp16 = self._resolve_use_fp16()
            try:
                self._model = FlagReranker(
                    self._model_path,
                    devices=self._device,
                    use_fp16=use_fp16,
                    batch_size=self._batch_size,
                    max_length=self._max_length,
                    normalize=True,
                )
            except Exception as exc:
                raise RerankerError(
                    "BGE-M3 重排模型加载失败: "
                    f"model={self._model_path}, device={self._device or 'auto'}"
                ) from exc

            logger.info(
                "BGE重排模型加载完成: model=%s device=%s fp16=%s elapsed_ms=%.2f",
                self._model_path,
                self._device or "auto",
                use_fp16,
                (time.perf_counter() - started_at) * 1000,
            )
            return self._model

    def _resolve_use_fp16(self) -> bool:
        if self._use_fp16 is not None:
            return self._use_fp16
        if self._device is not None:
            return self._device.startswith("cuda")

        try:
            import torch
        except ImportError:
            return False
        return bool(torch.cuda.is_available())

    @staticmethod
    def _validate_pairs(pairs: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
        if isinstance(pairs, (str, bytes)):
            raise TypeError("pairs 必须是 (query, passage) 序列")

        normalized: list[tuple[str, str]] = []
        for index, pair in enumerate(pairs):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise TypeError(f"pairs[{index}] 必须包含两个字符串")
            query, passage = pair
            if not isinstance(query, str) or not isinstance(passage, str):
                raise TypeError(f"pairs[{index}] 的 query 和 passage 必须是字符串")
            query = " ".join(query.strip().split())
            passage = " ".join(passage.strip().split())
            if not query or not passage:
                raise ValueError(f"pairs[{index}] 不能包含空文本")
            normalized.append((query, passage))
        return normalized

    @staticmethod
    def _normalize_model_output(raw_scores: Any,*,expected_count: int) -> list[float]:
        if isinstance(raw_scores, (int, float)):
            scores = [float(raw_scores)]
        else:
            try:
                scores = [float(score) for score in raw_scores]
            except (TypeError, ValueError) as exc:
                raise RerankerError("BGE-M3 重排模型返回了无法解析的分数") from exc

        if len(scores) != expected_count:
            raise RerankerError(
                "BGE-M3 重排分数数量与输入文本对数量不一致: "
                f"expected={expected_count}, actual={len(scores)}"
            )
        for index, score in enumerate(scores):
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise RerankerError(
                    "BGE-M3 归一化分数必须位于 [0, 1]: "
                    f"index={index}, score={score}"
                )
        return scores

    @staticmethod
    def _read_positive_int_env(name: str, *, default: int) -> int:
        raw_value = os.getenv(name)
        if raw_value is None:
            return default
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"环境变量 {name} 必须是整数") from exc
        if value <= 0:
            raise ValueError(f"环境变量 {name} 必须大于0")
        return value
