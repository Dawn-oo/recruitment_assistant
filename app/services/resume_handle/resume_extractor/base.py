from abc import ABC, abstractmethod

from .resume_schema import ResumeModel


class ResumeExtractor(ABC):

    @abstractmethod
    async def extract(
        self,
        markdown: str,
    ) -> ResumeModel:
        """
        Markdown简历 -> ResumeModel
        """
        pass

