from abc import ABC, abstractmethod

from .resume_schema import ResumeModel


class ResumeExtracSchema(ResumeModel):
    """
    简历结构化LLM抽取的 JSON Schema。
    """
    pass

class ResumeExtractor(ABC):

    @abstractmethod
    async def extract(
        self,
        markdown: str,
    ) -> ResumeExtracSchema:
        """
        Markdown简历 -> ResumeExtracSchema
        """
        pass

