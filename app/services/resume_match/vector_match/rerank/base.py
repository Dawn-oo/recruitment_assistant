from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class RerankerProvider(Protocol):
    """业务重排层依赖的最小文本对打分接口。

    业务代码只依赖该协议，不直接依赖 FlagEmbedding，便于替换模型
    以及在单元测试中注入 FakeReranker。
    """

    @property
    def model_name(self) -> str:
        """返回用于日志和结果审计的模型名称或本地模型路径。"""
        ...

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> list[float]:
        """按输入顺序返回归一化到 ``[0, 1]`` 的相关性分数。"""
        ...


class JDLookupRepository(Protocol):
    """目标岗位重排层依赖的最小完整 JD 查询接口。"""

    def find_by_ids(
        self,
        jd_ids: Sequence[int],
    ) -> list[Mapping[str, Any]]:
        """按输入 ID 批量返回完整 JD 数据库记录。"""
        ...
