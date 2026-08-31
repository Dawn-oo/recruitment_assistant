import asyncio
import logging

from app.services.resume_handle.resume_pipline import ResumeProcessingService
from app.services.resume_handle.document_parser.resume_parser import MinerUParser
from app.services.resume_handle.resume_extractor.resume_extractor_deepseek import DeepSeekResumeExtractor
from app.services.resume_handle.resume_validate.validate_report import ResumeValidator
from app.core.log_config import setup_logging
from app.tools.database_con import PostgresSSHConfig, PostgresSSHPool
from app.services.job_match.vector_match.resume_query_builder import ResumeQueryBuilder
from app.services.job_match.vector_match.vector_retriever import VectorRetriever
from app.services.job_match.jd_repository import JDRepository
from app.tools.embeding_assist import BgeM3EmbeddingProvider


logger = logging.getLogger(__name__)

async def main():

    setup_logging()

    # 文件解析器，主要负责解析简历文件，提取文本内容，将pdf转为markdown
    document_parser = MinerUParser()

    # 简历提取器，主要负责从markdown中提取简历信息
    resume_extractor = DeepSeekResumeExtractor()

    # 简历验证器，主要负责验证简历信息是否符合要求
    # resume_validator = ResumeValidator()

    # 简历处理服务，主要负责处理简历文件，包括解析、提取、验证等
    service = ResumeProcessingService(
        document_parser=document_parser,
        resume_extractor=resume_extractor,
    )

    result = await service.process(
        r"E:\Project\assistant_for_recruitment\app\services\resume_handle\人力资源培训专员简历_张晓婷.pdf"
    )
    print(result)

    # 查询构建器，主要负责根据简历信息构建查询语句
    query_builder = ResumeQueryBuilder()
    query_result = query_builder.build(result.resume)
    print(query_result)

    # 嵌入模型，主要负责将简历信息转换为向量表示
    embedder = BgeM3EmbeddingProvider()

    config = PostgresSSHConfig.from_env()
    with PostgresSSHPool(config) as db:
        # 向量检索器，主要负责根据查询语句和简历向量表示，从数据库中检索最相关的岗位描述
        retriever = VectorRetriever(repository=JDRepository(db=db),embedder=embedder)


        retrieval_result = retriever.retrieve(query_result.query_units)
        print(retrieval_result)


    # resume, report = resume_validator.validate(result)
    # print(resume)
    # print(report)
    # print()



if __name__ == "__main__":
    asyncio.run(main())