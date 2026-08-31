from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, TypeAlias

from app.tools.database_con import PostgresSSHPool,get_default_db

from app.services.job_match.exact_match.job_alias import STANDARD_JOB_TITLES

JDRow: TypeAlias = dict[str, Any]
JDChunkRow: TypeAlias = dict[str, Any]


class JDRepository:
    """
    JD 数据访问层。
    只负责数据库访问：
    1. 查询内部标准岗位名称
    2. 根据岗位名称精确查询 JD
    3. 根据 JD ID 查询完整 JD
    4. 使用 pgvector 对 JD chunks 做向量召回
    """

    JD_TABLE = "jd_descriptions"
    CHUNK_TABLE = "jd_chunks"

    # 当前项目 BGE-M3 embedding 维度
    EMBEDDING_DIM = 1024

    def __init__(self,db: PostgresSSHPool | None = None,) -> None:
        """
        可以显式传入连接池，方便测试和依赖注入。
        不传则使用应用启动时初始化好的默认连接池。
        """
        self._db = db or get_default_db()

    # 1. 标准岗位名称

    def get_all_job_titles(self) -> list[str]:

        return list(STANDARD_JOB_TITLES)

    # 2. 精确岗位名称查询

    def find_all_by_job_title(self,job_title: str) -> list[JDRow]:
        """
        根据内部标准岗位名称做严格精确匹配。
        返回 list 而不是单个 JD，是因为同一岗位名称可能存在于不同部门
        """

        title = job_title.strip()

        if not title:
            return []

        query = f"""
            SELECT
                id,
                job_title,
                department,
                responsibilities,
                minimum_education,
                education_background,
                work_experience_raw,
                competencies,
                content_hash,
                source_payload,
                schema_version,
                created_at,
                updated_at
            FROM {self.JD_TABLE}
            WHERE job_title = %s
            ORDER BY id
        """

        return self._db.fetch_all(query,(title,))

    def find_all_by_job_titles(self,job_titles: Sequence[str]) -> list[JDRow]:
        """
        一次查询多个标准岗位名称。适合：多岗位申请在上层完成拆分和 alias 标准化以后，一次性查询对应JD。
        """

        titles = self._normalize_non_empty_strings(job_titles)

        if not titles:
            return []

        query = f"""
            SELECT
                id,
                job_title,
                department,
                responsibilities,
                minimum_education,
                education_background,
                work_experience_raw,
                competencies,
                content_hash,
                source_payload,
                schema_version,
                created_at,
                updated_at
            FROM {self.JD_TABLE}
            WHERE job_title = ANY(%s)
            ORDER BY job_title, id
        """

        return self._db.fetch_all(query,(titles,))

    # 3. JD ID查询

    def find_by_id(self,jd_id: int) -> JDRow | None:
        """
        根据主键查询一条完整 JD。
        """

        query = f"""
            SELECT
                id,
                job_title,
                department,
                responsibilities,
                minimum_education,
                education_background,
                work_experience_raw,
                competencies,
                content_hash,
                source_payload,
                schema_version,
                created_at,
                updated_at
            FROM {self.JD_TABLE}
            WHERE id = %s
        """

        return self._db.fetch_one(query,(jd_id,))

    def find_by_ids(self,jd_ids: Sequence[int]) -> list[JDRow]:
        """
        根据多个 jd_id 一次查询完整 JD。

        典型链路：
            VectorRetriever
                -> TopK chunks
                -> CandidateAggregator
                -> TopN jd_id
                -> find_by_ids()
        """

        ids = self._deduplicate_ints(
            jd_ids
        )

        if not ids:
            return []

        query = f"""
            SELECT
                id,
                job_title,
                department,
                responsibilities,
                minimum_education,
                education_background,
                work_experience_raw,
                competencies,
                content_hash,
                source_payload,
                schema_version,
                created_at,
                updated_at
            FROM {self.JD_TABLE}
            WHERE id = ANY(%s)
        """

        rows = self._db.fetch_all(
            query,
            (ids,),
        )

        # 保持调用方给出的 jd_id 顺序
        row_map = {
            row["id"]: row
            for row in rows
        }

        return [
            row_map[jd_id]
            for jd_id in ids
            if jd_id in row_map
        ]

    # 4. pgvector向量召回(查询接口)

    def search_similar_chunks(self,embedding: Sequence[float],*,top_k: int = 20,chunk_types: Sequence[str] | None = None,
        exclude_jd_ids: Sequence[int] | None = None) -> list[JDChunkRow]:
        """
        对 JD chunks 做余弦距离检索。similarity只是召回相似度，不是最终岗位匹配分数。
        """

        if top_k <= 0:
            raise ValueError(f"top_k 必须大于 0，实际为 {top_k}")

        # 类型转换：把embedding向量转换为pgvector可以解析的字符串
        vector_literal = self._to_vector_literal(embedding)

        conditions = [ "is_embedding_target = TRUE","embedding IS NOT NULL"]

        # SELECT distance / similarity 各需要一个 vector 参数
        params: list[Any] = [vector_literal,vector_literal]

        if chunk_types:
            normalized_types =self._normalize_non_empty_strings(chunk_types)


            if normalized_types:
                conditions.append("chunk_type = ANY(%s)")
                params.append(normalized_types)

        if exclude_jd_ids:
            excluded_ids = self._deduplicate_ints(exclude_jd_ids)

            if excluded_ids:
                conditions.append("NOT (jd_id = ANY(%s))")
                params.append(excluded_ids)

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT
                id AS chunk_id,
                jd_id,
                chunk_key,
                chunk_type,
                source_sequence,
                part_index,
                item_index,
                content,
                metadata,
                embedding_model,
                embedded_at,

                (
                    embedding <=> %s::vector
                ) AS distance,

                1 - (
                    embedding <=> %s::vector
                ) AS similarity

            FROM {self.CHUNK_TABLE}

            WHERE {where_clause}

            ORDER BY
                embedding <=> %s::vector

            LIMIT %s
        """

        # ORDER BY 还需要一次 query vector
        params.append(vector_literal)

        # LIMIT
        params.append(top_k)

        return self._db.fetch_all(query,tuple(params))

    # 5. Chunk 调试 / 证据追踪

    def find_chunk_by_id(self,chunk_id: int) -> JDChunkRow | None:
        """
        根据 chunk_id 查询单条 chunk。
        用于：
        - 调试召回结果
        - Agent evidence 追踪
        - Review 页面展示检索证据
        """

        query = f"""
            SELECT
                id AS chunk_id,
                jd_id,
                chunk_key,
                chunk_type,
                source_sequence,
                part_index,
                item_index,
                content,
                metadata,
                embedding_model,
                embedded_at
            FROM {self.CHUNK_TABLE}
            WHERE id = %s
        """

        return self._db.fetch_one(query,(chunk_id,))

    # Internal Helpers

    @classmethod
    def _to_vector_literal(cls,embedding: Sequence[float]) -> str:
        """
        将 Python embedding 转换为 pgvector 可以解析的字符串。[0.1, 0.2]->"[0.1,0.2]"
        这里使用 %s::vector，
        因此不要求连接池提前注册 pgvector psycopg adapter。
        """

        if len(embedding) != cls.EMBEDDING_DIM:
            raise ValueError(
                "embedding 维度错误："
                f"期望 {cls.EMBEDDING_DIM}，"
                f"实际 {len(embedding)}"
            )

        values: list[str] = []

        for index, value in enumerate(embedding):
            number = float(value)

            if not math.isfinite(number):
                raise ValueError(
                    "embedding 中存在 NaN 或 Infinity："
                    f"index={index}, value={value}"
                )

            values.append(str(number))

        return "["+ ",".join(values)+ "]"

    @staticmethod
    def _normalize_non_empty_strings(values: Sequence[str]) -> list[str]:
        """
        去掉空字符串，并保持原顺序去重。
        """

        result: list[str] = []
        seen: set[str] = set()

        for value in values:
            normalized = value.strip()

            if not normalized:
                continue

            if normalized in seen:
                continue

            seen.add(normalized)
            result.append(normalized)

        return result

    @staticmethod
    def _deduplicate_ints(values: Sequence[int]) -> list[int]:
        """
        对整数 ID 保持原顺序去重。
        """

        result: list[int] = []
        seen: set[int] = set()

        for value in values:
            normalized = int(value)

            if normalized in seen:
                continue

            seen.add(normalized)
            result.append(normalized)

        return result
