"""岗位匹配真实链路测试入口。

流程：本地 PDF -> MinerU -> DeepSeek -> PostgreSQL 精确匹配 ->
未命中岗位 BGE-M3-M3 向量召回 -> BGE-M3 reranker -> 人工确认 -> Agent 输入。

在项目根目录运行：
    python -m app.services.resume_match.test --resume "E:/path/resume.pdf"
"""

from __future__ import annotations

from dotenv import load_dotenv

print("开始测试")
import argparse
import asyncio
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
import os
print("系统库加载完成")

from app.core.log_config import setup_logging
from app.services.resume_handle import create_resume_processing_service
from app.services.resume_match.exact_match.exact_job_matcher import ExactJobMatcher
from app.services.resume_match.exact_match.job_intent_norm import JobIntentNormalizer
from app.services.resume_match.ResumeMatchService import (
    MatchResumeService,
    ResumeMatchResult,
    ResumeMatchStatus,
    TargetMatchStatus,
)
from app.database_search.jd_repository import JDRepository
from app.services.resume_match.vector_match.recall.job_candidate_aggregator import CandidateAggregator
from app.services.resume_match.vector_match.recall.resume_query_builder import ResumeQueryBuilder
from app.services.resume_match.vector_match.recall.vector_retriever import VectorRetriever
from app.tools import BgeReranker, PostgresSSHConfig, PostgresSSHPool, BgeM3EmbeddingProvider
from app.services.resume_match.vector_match.rerank.target_job_reranker import TargetJobReranker
from app.services.resume_match.vector_match.semantic_matching_service import SemanticTargetMatchingService
from app.services.middle_layer.agent_input_assembler import AgentInputAssembler


print("应用库加载完成")

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

_ = load_dotenv()



class LazyBgeM3EmbeddingProvider:
    """首次真正请求向量时才加载 BGE-M3-M3，避免精确命中时占用模型资源。"""

    dimension = BgeM3EmbeddingProvider.DIMENSION

    def __init__(self, **model_kwargs) -> None:
        self._model_kwargs = model_kwargs
        self._provider: BgeM3EmbeddingProvider | None = None

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        if self._provider is None:
            logger.info("精确匹配存在未命中岗位，开始加载本地 BGE-M3-M3")
            self._provider = BgeM3EmbeddingProvider(**self._model_kwargs)
        return self._provider.embed_queries(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_queries([text])[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试简历岗位匹配完整链路")
    parser.add_argument(
        "--resume",
        type=Path,
        help="本地简历 PDF 路径",
        default=r"E:\Project\assistant_for_recruitment\app\services\resume_handle\人力资源培训专员简历_张晓婷.pdf"
    )
    parser.add_argument(
        "--embedding-model",
        type=Path,
        help="本地 BGE-M3-M3 目录；省略时使用 BGE_MODEL_PATH",
        default=os.getenv("BGE_MODEL_PATH")
    )
    parser.add_argument(
        "--reranker-model",
        type=Path,
        help="本地 BGE-M3 reranker 目录；省略时使用 BGE_RERANKER_MODEL_PATH",
        default=os.getenv("BGE_RERANKER_MODEL_PATH")
    )
    parser.add_argument("--device", default='cpu', help="cpu、cuda 或 cuda:0；省略时使用环境变量")
    parser.add_argument("--top-k-per-query", type=int, default=30)
    parser.add_argument("--recall-top-n", type=int, default=10)
    parser.add_argument("--rerank-top-n", type=int, default=5)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--reranker-batch-size", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "resume_match_result.json",
    )
    parser.add_argument(
        "--confirmed-output",
        type=Path,
        default=PROJECT_ROOT / "resume_match_confirmed_result.json",
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="只输出首次匹配结果，不在终端进行人工确认",
    )
    return parser.parse_args()


def checked_path(path: Path, name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{name}不存在: {resolved}")
    return resolved


def save_result(result: ResumeMatchResult, path: Path) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        result.model_dump_json(indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"结果文件: {path}")


def show_result(result: ResumeMatchResult) -> None:
    print("\n" + "=" * 80)
    print(f"总体状态: {result.status.value}")
    for target in result.targets:
        source = target.source.value if target.source else "-"
        print(
            f"\n{target.target_id}: 申请岗位={target.requested_job_title} "
            f"状态={target.status.value} 来源={source}"
        )
        if target.selected_jd_id is not None:
            print(f" 自动/人工选定 jd_id={target.selected_jd_id}")
        for candidate in target.candidates:
            score = "-" if candidate.score is None else f"{candidate.score:.6f}"
            print(
                f"  rank={candidate.candidate_rank:<2} jd_id={candidate.jd_id:<6} "
                f"score={score:<10} 岗位={candidate.job_title} "
                f"部门={candidate.department or '-'}"
            )
        for warning in target.warnings:
            print(f"  warning: {warning}")
    print("=" * 80)


def ask_for_confirmations(result: ResumeMatchResult) -> dict[str, int]:
    selections: dict[str, int] = {}
    for target in result.targets:
        if target.status != TargetMatchStatus.NEEDS_CONFIRMATION:
            continue

        allowed_ids = {candidate.jd_id for candidate in target.candidates}
        while True:
            raw = input(
                f"请选择 {target.target_id}（{target.requested_job_title}）的 jd_id "
                "[回车表示暂不确认]: "
            ).strip()
            if not raw:
                break
            try:
                selected_id = int(raw)
            except ValueError:
                print("jd_id 必须是整数。")
                continue
            if selected_id not in allowed_ids:
                print(f"只能选择当前候选集合中的 JD: {sorted(allowed_ids)}")
                continue
            selections[target.target_id] = selected_id
            break
    return selections


def create_embedder(args: argparse.Namespace) -> LazyBgeM3EmbeddingProvider:
    kwargs = {"batch_size": args.embedding_batch_size}
    if args.embedding_model is not None:
        kwargs["model_path"] = checked_path(args.embedding_model, "BGE-M3-M3 模型目录")
    if args.device is not None:
        kwargs["device"] = args.device
    return LazyBgeM3EmbeddingProvider(**kwargs)


async def run(args: argparse.Namespace) -> None:
    started_at = time.perf_counter()
    resume_path = checked_path(args.resume, r"E:\Project\assistant_for_recruitment\app\services\resume_handle\人力资源培训专员简历_张晓婷.pdf")

    logger.info("第一阶段：解析并结构化简历 path=%s", resume_path)
    resume_processing_service = create_resume_processing_service()
    resume_result = await resume_processing_service.process(resume_path)
    print(f"简历申请岗位: {resume_result.resume.basic_info.target_job_title!r}")

    logger.info("第二阶段：准备本地 BGE-M3 模型（按需加载）")
    embedder = create_embedder(args)
    reranker_model = (
        checked_path(args.reranker_model, "BGE-M3 reranker 模型目录")
        if args.reranker_model is not None
        else None
    )
    # BgeReranker 是懒加载，只有出现精确未命中岗位时才真正加载模型。
    reranker = BgeReranker(model_path=reranker_model,device=args.device,batch_size=args.reranker_batch_size)

    try:
        logger.info("第三阶段：建立 SSH 隧道和 PostgreSQL 连接池")
        with PostgresSSHPool(PostgresSSHConfig.from_env()) as db:
            repository = JDRepository(db=db)
            semantic_service = SemanticTargetMatchingService(
                query_builder=ResumeQueryBuilder(),
                vector_retriever=VectorRetriever(repository, embedder),
                candidate_aggregator=CandidateAggregator(),
                target_job_reranker=TargetJobReranker(
                    repository=repository,
                    reranker=reranker,
                ),
            )
            match_service = MatchResumeService(
                job_intent_normalizer=JobIntentNormalizer(),
                exact_matcher=ExactJobMatcher(repository),
                semantic_matching_service=semantic_service,
            )

            logger.info("第四阶段：精确匹配，未命中岗位按需执行语义降级")
            result = match_service.match_resume(
                resume=resume_result,
                top_k_per_query=args.top_k_per_query,
                recall_top_n=args.recall_top_n,
                rerank_top_n=args.rerank_top_n,
            )
            show_result(result)
            save_result(result, args.output)

            if (
                result.status == ResumeMatchStatus.NEEDS_CONFIRMATION
                and not args.no_confirm
                and sys.stdin.isatty()
            ):
                selections = ask_for_confirmations(result)
                if selections:
                    result = match_service.confirm_candidates(
                        result=result,
                        selections=selections,
                    )
                    print("\n人工确认后的结果：")
                    show_result(result)
                    save_result(result, args.confirmed_output)

            if result.status == ResumeMatchStatus.READY:
                print("\nAgent 输入：")
                print(result.to_agent_payload())
            else:
                print("\n当前结果不能进入 Agent。")

            logger.info("第五阶段：构造 Agent 输入")
            assembler = AgentInputAssembler(repository)
            agent_input = assembler.build(resume=resume_result.resume, match_result=result)
            print(agent_input.model_dump_json(ensure_ascii=False, indent=2))

    finally:
        reranker.close()
        logger.info("完整链路结束，总耗时 %.2f 秒", time.perf_counter() - started_at)



def main() -> None:
    setup_logging()
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    main()
