import time
start_run_time = time.perf_counter()
print("开始运行")
import asyncio
import logging

from app.services.resume_handle import create_resume_processing_service
time1 = time.perf_counter()
print(f"导入建立解析文件包耗时为：{time1 - start_run_time}")


from app.core.log_config import setup_logging
from app.tools.database_con import PostgresSSHConfig, PostgresSSHPool
time2 = time.perf_counter()
print(f"导入建立数据库连接包耗时为：{time2 - time1}")

from app.services.resume_match.vector_match.resume_query_builder import ResumeQueryBuilder
from app.services.resume_match.vector_match.vector_retriever import VectorRetriever
from app.services.resume_match.vector_match.candidate_aggregator import CandidateAggregator
from app.services.resume_match.sql_search.jd_repository import JDRepository
time3 = time.perf_counter()
print(f"导入建立向量匹配相关包耗时为：{time3 - time2}")

from app.tools import BgeM3EmbeddingProvider
time4 = time.perf_counter()
print(f"导入建立向量嵌入相关包耗时为：{time4 - time3}")

from app.services.resume_match.exact_match.exact_job_matcher import ExactJobMatcher
from app.services.resume_match.exact_match.job_intent_norm import JobIntentNormalizer
from app.services.resume_match.candidate_selector import CandidateSelector
time5 = time.perf_counter()
print(f"导入建立精确匹配相关包耗时为：{time5 - time4}")



logger = logging.getLogger(__name__)

async def main():

    setup_logging()

    """"# 文件解析器，主要负责解析简历文件，提取文本内容，将pdf转为markdown
    document_parser = MinerUParser()

    # 简历提取器，主要负责从markdown中提取简历信息
    resume_extractor = DeepSeekResumeExtractor()

    # 简历验证器，主要负责验证简历信息是否符合要求
    resume_validator = ResumeValidator()"""

    # 简历处理服务，主要负责处理简历文件，包括解析、提取、验证等
    service = create_resume_processing_service()

    result = await service.process(
        r"E:\Project\assistant_for_recruitment\app\services\resume_handle\人力资源培训专员简历_张晓婷.pdf"
    )

    # print("简历信息为：")
    # print(result.resume.model_dump_json(indent=2, ensure_ascii=False))

    # print("简历验证结果为：")
    # if resume_validator.validate(result.resume.model_dump())[0]:
        # print(resume_validator.validate(result.resume.model_dump())[1].model_dump_json(indent=2, ensure_ascii=False))



    # =====================================================
    # 1. 清洗简历信息，提取岗位名称，用于精确匹配岗位
    # =====================================================
    print("构建岗位意图预处理器")
    job_intent_normalizer = JobIntentNormalizer()
    print("岗位意图预处理器构建完成")

    print("岗位意图预处理中...")
    job_intent_result = job_intent_normalizer.normalize(result.resume.basic_info.target_job_title)
    print("岗位意图预处理完成")

    print()

    # =====================================================
    # 2. 构建查询语句，用于向量检索
    # =====================================================
    print("构建查询构建器,根据简历信息构建查询语句...")
    # 查询构建器，主要负责根据简历信息构建查询语句
    query_builder = ResumeQueryBuilder()
    print("查询构建器构建完成")

    print("查询语句构建中...")
    query_result = query_builder.build(result.resume)
    print("查询语句构建完成")

    print("构建的查询语句结果为：")
    print(query_result.model_dump_json(indent=2, ensure_ascii=False))

    print()

    # 嵌入模型，主要负责将简历信息转换为向量表示
    embedder = BgeM3EmbeddingProvider()

    config = PostgresSSHConfig.from_env()

    start_time = time.perf_counter()
    logger.info("数据库连接中...")
    with PostgresSSHPool(config) as db:
        end_time = time.perf_counter()
        logger.info("数据库连接完成,耗时为：latency_ms=%.0f毫秒", (end_time - start_time) * 1000)

        print("构建精确匹配器")
        jd_rep = JDRepository(db=db)
        exact_job_matcher = ExactJobMatcher(repository=jd_rep)
        print("精确匹配器构建完成")
        print()

        print("构建向量检索器")
        # 向量检索器，主要负责根据查询语句和简历向量表示，从数据库中检索最相关的岗位描述
        retriever = VectorRetriever(repository=jd_rep,embedder=embedder)
        print("向量检索器构建完成")
        print()

        print("检索中...")

        # 1、精确匹配岗位意图
        retrieval_result1 = exact_job_matcher.match(job_intent_result)
        # print(retrieval_result1.intent_results[0].matched_jds[0].model_dump_json(indent=2, ensure_ascii=False))


        # 2、向量检索岗位描述
        retrieval_result2 = retriever.retrieve(query_result.query_units)
        print("检索完成")
        print(f"检索结果为：{len(retrieval_result2.query_results)}条符合条件的岗位描述")
        # print(retrieval_result2.query_results[0].model_dump_json(indent=2, ensure_ascii=False))


        # 3、聚合岗位岗位描述
        print("构建候选岗位聚合器")
        candidate_aggregator = CandidateAggregator()
        print("候选岗位聚合器构建完成")
        print("聚合中...")
        aggregated_result = candidate_aggregator.aggregate(retrieval_result2)
        print("聚合完成")
        # print(aggregated_result.model_dump_json(indent=2, ensure_ascii=False))

        # 4、jd返回路由选择
        candidateselector = CandidateSelector(repository=jd_rep)
        result = candidateselector.select(exact_result=retrieval_result1,semantic_result=aggregated_result)

        with open("result.json", "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2, ensure_ascii=False))

        # with open("test.json","w",encoding="utf-8") as f:
        #     f.write("{\"精确匹配岗位意图结果\":\n")
        #     f.write(retrieval_result1.model_dump_json(indent=2, ensure_ascii=False) + "\n")
        #     f.write(",\n")
        #
        #     f.write("\n")
        #
        #     f.write("\"向量检索岗位描述结果\":\n")
        #     f.write(retrieval_result2.model_dump_json(indent=2, ensure_ascii=False) + "\n")
        #     f.write("}\n")


if __name__ == "__main__":
    asyncio.run(main())
    end_run_time = time.perf_counter()
    print("运行完成")
    print(f"运行耗时为：{end_run_time - start_run_time}秒")

