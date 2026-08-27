import asyncio

from app.services.resume_handle.resume_process import ResumeProcessingService
from document_parser.resume_parser import MinerUParser
from resume_extractor.resume_extractor_deepseek import DeepSeekResumeExtractor



async def main():

    document_parser = MinerUParser()

    resume_extractor = DeepSeekResumeExtractor()

    service = ResumeProcessingService(
        document_parser=document_parser,
        resume_extractor=resume_extractor,
    )

    result = await service.process(
        r"E:\Project\assistant_for_recruitment\app\services\resume_handle\人力资源培训专员简历_张晓婷.pdf"
    )

    print(
        result.resume.model_dump_json(
            indent=2
        )
    )


if __name__ == "__main__":
    asyncio.run(main())