from abc import ABC, abstractmethod

from .resume_schema import ResumeModel


class ResumeExtractor(ABC):

    @abstractmethod
    async def extract(
        self,
        markdown: str,
        force_refresh: bool = False,
    ) -> ResumeModel:
        """
        Markdown简历 -> ResumeModel
        """
        pass

