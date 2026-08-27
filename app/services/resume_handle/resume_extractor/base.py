from abc import ABC, abstractmethod

from app.services.resume_handle.resume_extractor.Resume_schema import ResumeModel


class ResumeExtractor(ABC):

    @abstractmethod
    async def extract(
        self,
        markdown: str,
    ) -> ResumeModel:
        """
        Markdown简历 -> ResumeSchema
        """
        pass