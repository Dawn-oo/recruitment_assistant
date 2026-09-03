from app.services.resume_handle import ResumeProcessingService
from app.services.resume_handle.document_parser import MinerUParser
from app.services.resume_handle.resume_extractor import DeepSeekResumeExtractor
from app.services.resume_handle.resume_validate import ResumeValidator

def create_resume_processing_service() -> ResumeProcessingService:
    return ResumeProcessingService(
        document_parser=MinerUParser(),
        resume_extractor=DeepSeekResumeExtractor()
    )