from collections.abc import Sequence
from enum import Enum
from typing import Protocol

class ResumeQueryType(str, Enum):
    """目标岗位语义召回使用的 Query 类型。"""

    TARGET_JOB_TITLE = "target_job_title"
    WORK_EXPERIENCE = "work_experience"
    PROJECT_EXPERIENCE = "project_experience"
    SKILLS = "skills"


class EmbeddingProvider(Protocol):
    """向量召回层依赖的最小文本嵌入接口。"""

    def embed_queries(self,texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """按输入顺序返回每段文本的稠密向量。"""
