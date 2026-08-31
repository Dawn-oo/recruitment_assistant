from __future__ import annotations

import numpy as np
import os
from pathlib import Path
from typing import Sequence
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel


_ = load_dotenv()

class EmbeddingError(RuntimeError):
    """文本嵌入失败。"""


class BgeM3EmbeddingProvider:
    """
    基于本地 BGE-M3 模型的稠密向量嵌入实现。

    当前只使用 BGE-M3 的 dense_vecs，
    不计算 sparse 和 ColBERT 向量。
    """

    DIMENSION = 1024
    MODEL_NAME = "BAAI/bge-m3"

    MODEL_PATH = os.getenv("BGE_MODEL_PATH")
    DEVICE = os.getenv("BGE_DEVICE")
    BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "8"))
    MAX_LENGTH = int(os.getenv("BGE_MAX_LENGTH", "1024"))

    def __init__(self,*,
                 model_path: str | Path = MODEL_PATH,
                 device: str = DEVICE,
                 batch_size: int = BATCH_SIZE,
                 max_length: int = MAX_LENGTH) -> None:

        self._model_path = str(model_path)
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length

        # CPU 不应使用 FP16。
        use_fp16 = device.startswith("cuda")

        self._model = BGEM3FlagModel(str(model_path),devices=device,use_fp16=use_fp16,normalize_embeddings=True)

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    def embed_query(self,text: str) -> list[float]:
        vectors = self.embed_queries([text])
        return vectors[0]

    def embed_queries(self,texts: Sequence[str]) -> list[list[float]]:
        normalized_texts = self._validate_texts(texts)

        if not normalized_texts:
            return []

        try:
            outputs = self._model.encode(
                normalized_texts,
                batch_size=self._batch_size,
                max_length=self._max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
        except Exception as exc:
            raise EmbeddingError(f"BGE-M3 文本嵌入失败，文本数量={len(normalized_texts)}") from exc

        if "dense_vecs" not in outputs:raise EmbeddingError("BGE-M3 返回结果中不存在 dense_vecs")

        vectors = np.asarray(outputs["dense_vecs"],dtype=np.float32)

        self._validate_vectors(vectors=vectors,expected_count=len(normalized_texts))

        return vectors.tolist()

    @staticmethod
    def _validate_texts(texts: Sequence[str]) -> list[str]:
        # str 本身也是 Sequence[str]，必须单独阻止。
        if isinstance(texts, str):
            raise TypeError(
                "embed_queries() 需要字符串序列；"
                "嵌入单条文本请调用 embed_query()"
            )

        normalized_texts: list[str] = []

        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(
                    f"texts[{index}] 必须是 str，"
                    f"实际类型为 {type(text).__name__}"
                )

            normalized = text.strip()

            if not normalized:
                raise ValueError(f"texts[{index}] 不能为空字符串")

            normalized_texts.append(normalized)

        return normalized_texts

    def _validate_vectors(self,*,vectors: np.ndarray,expected_count: int) -> None:
        if vectors.ndim != 2:
            raise EmbeddingError(f"嵌入结果应为二维数组，实际 shape={vectors.shape}")

        expected_shape = (expected_count,self.dimension,)

        if vectors.shape != expected_shape:
            raise EmbeddingError(f"嵌入结果维度不符合预期：expected={expected_shape}, actual={vectors.shape}")

        if not np.isfinite(vectors).all():
            raise EmbeddingError("嵌入结果中包含 NaN 或 Infinity")