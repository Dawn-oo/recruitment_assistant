import asyncio
import logging

from app.services.resume_handle.resume_pipline import ResumeProcessingService
from document_parser.resume_parser import MinerUParser
from resume_extractor.resume_extractor_deepseek import DeepSeekResumeExtractor
from resume_validate.validate_report import ResumeValidator
from app.core.log_config import setup_logging

logger = logging.getLogger(__name__)

async def main():

    setup_logging()


    document_parser = MinerUParser()

    resume_extractor = DeepSeekResumeExtractor()

    resume_validator = ResumeValidator()

    service = ResumeProcessingService(
        document_parser=document_parser,
        resume_extractor=resume_extractor,
    )

    result = await service.process(
        r"E:\Project\assistant_for_recruitment\app\services\resume_handle\人力资源培训专员简历_张晓婷.pdf"
    )

    # resume, report = resume_validator.validate(result)
    # print(resume)
    # print(report)
    # print()



if __name__ == "__main__":
    asyncio.run(main())
