import asyncio
import json
import re
import selectors
import dotenv,os
from dataclasses import dataclass
from typing import Any

import asyncssh
import psycopg

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from transformers import AutoTokenizer


# ============================================================
# 配置
# ============================================================
_ = dotenv.load_dotenv()

SSH_HOST = os.getenv("SSH_HOST")
SSH_USERNAME = os.getenv("SSH_USERNAME")
SSH_PASSWORD = os.getenv("SSH_PASSWORD")


# SSH 本地端口转发
LOCAL_HOST = os.getenv("LOCAL_HOST")
LOCAL_PORT = int(os.getenv("LOCAL_PORT"))

REMOTE_DB_HOST = os.getenv("REMOTE_DB_HOST")
REMOTE_DB_PORT = int(os.getenv("REMOTE_DB_PORT"))


# PostgreSQL 配置
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


TOKENIZER_PATH = r"/app/RAG知识库构建/embeding_models/bge-m3"


# ---------------- Chunk策略 ----------------

CHUNK_VERSION = 1

# 岗位职责：
# description + tasks 总长度不超过该值时，
# 整个 responsibility 保留为一个 chunk。
RESPONSIBILITY_MAX_TOKENS = 400

# 任职资格单条要求最大长度
QUALIFICATION_MAX_TOKENS = 300


# ============================================================
# jd_chunks 表
# ============================================================

CREATE_CHUNK_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS jd_chunks (
    id BIGSERIAL PRIMARY KEY,

    -- 对应 job_descriptions.id
    jd_id BIGINT NOT NULL
        REFERENCES job_descriptions(id)
        ON DELETE CASCADE,

    -- 当前 chunk 的唯一业务标识
    chunk_key TEXT NOT NULL UNIQUE,

    -- responsibility / competency /
    -- education / experience / education_background
    chunk_type VARCHAR(50) NOT NULL,

    -- responsibilities.sequence
    source_sequence INTEGER,

    -- 同一个 responsibility 被切成多个 child 时使用
    part_index INTEGER NOT NULL DEFAULT 1,

    -- competencies 中原始条目的位置
    item_index INTEGER,

    -- 真正送入 embedding 模型的文本
    content TEXT NOT NULL,

    -- 保存 time_percentage、job_title 等信息
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- 后面 embed_jd.py 可以只处理 TRUE 的数据
    is_embedding_target BOOLEAN NOT NULL DEFAULT TRUE,

    -- 方便以后修改 chunk 策略
    chunk_version INTEGER NOT NULL DEFAULT 1,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


CREATE_JD_ID_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_jd_chunks_jd_id
ON jd_chunks(jd_id);
"""


CREATE_CHUNK_TYPE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_jd_chunks_chunk_type
ON jd_chunks(chunk_type);
"""


# ============================================================
# SQL
# ============================================================

SELECT_JDS_SQL = """
SELECT
    id,
    job_title,
    department,
    responsibilities,
    minimum_education,
    education_background,
    work_experience_raw,
    competencies
FROM job_descriptions
ORDER BY id;
"""


INSERT_CHUNK_SQL = """
INSERT INTO jd_chunks (
    jd_id,
    chunk_key,
    chunk_type,
    source_sequence,
    part_index,
    item_index,
    content,
    metadata,
    is_embedding_target,
    chunk_version
)
VALUES (
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s
)
ON CONFLICT (chunk_key)
DO UPDATE SET
    content = EXCLUDED.content,
    metadata = EXCLUDED.metadata,
    is_embedding_target = EXCLUDED.is_embedding_target,
    chunk_version = EXCLUDED.chunk_version;
"""


# ============================================================
# Chunk 数据模型
# ============================================================

@dataclass
class JDChunk:
    jd_id: int

    chunk_key: str

    chunk_type: str

    content: str

    metadata: dict[str, Any]

    source_sequence: int | None = None

    part_index: int = 1

    item_index: int | None = None

    is_embedding_target: bool = True


# ============================================================
# Tokenizer
# ============================================================

print("正在加载 tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_PATH
)

print("Tokenizer 加载完成")


def count_tokens(
    text: str,
) -> int:
    """
    统计文本 token 数量。
    """

    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
        )
    )


# ============================================================
# 通用文本辅助函数
# ============================================================

def ensure_list(
    value,
) -> list:
    """
    PostgreSQL JSONB / TEXT[] 正常情况下 psycopg
    会自动转换成 Python list。

    此函数只是增加防御性。
    """

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, str):
        try:
            parsed = json.loads(value)

            if isinstance(parsed, list):
                return parsed

        except json.JSONDecodeError:
            pass

    raise TypeError(
        f"预期 list，实际得到 {type(value)}"
    )


# ============================================================
# 超长文本二次切分
# ============================================================

def split_by_sentence(
    text: str,
    max_tokens: int,
) -> list[str]:
    """
    对一个非常长的单条 requirement / task 进行兜底切分。

    优先按照：
        。 ！ ？ ； ;

    进行拆分。

    不做 SemanticChunker。
    """

    if count_tokens(text) <= max_tokens:
        return [text]

    parts = re.split(
        r"(?<=[。！？；;])",
        text,
    )

    parts = [
        part.strip()
        for part in parts
        if part.strip()
    ]

    results = []

    current = ""

    for part in parts:

        candidate = (
            current + part
            if current
            else part
        )

        if count_tokens(candidate) <= max_tokens:
            current = candidate
            continue

        if current:
            results.append(current)
            current = ""

        # --------------------------------------------
        # 极端情况：
        # 一个句子本身就超过 max_tokens
        #
        # 最后才按 token 硬切
        # --------------------------------------------

        if count_tokens(part) > max_tokens:

            token_ids = tokenizer.encode(
                part,
                add_special_tokens=False,
            )

            for start in range(
                0,
                len(token_ids),
                max_tokens,
            ):

                chunk_ids = token_ids[
                    start:
                    start + max_tokens
                ]

                chunk_text = tokenizer.decode(
                    chunk_ids,
                    skip_special_tokens=True,
                ).strip()

                if chunk_text:
                    results.append(
                        chunk_text
                    )

        else:
            current = part

    if current:
        results.append(current)

    return results


# ============================================================
# 构建 responsibility 文本
# ============================================================

def build_responsibility_content(
    job_title: str,
    department: str,
    description: str,
    tasks: list[str],
) -> str:

    lines = [
        f"岗位：{job_title}",
        f"部门：{department}",
        f"岗位职责：{description}",
    ]

    if tasks:

        lines.append(
            "具体工作任务："
        )

        for task in tasks:
            lines.append(
                f"- {task}"
            )

    return "\n".join(lines)


# ============================================================
# 切岗位职责
# ============================================================

def chunk_responsibilities(
    jd: dict,
) -> list[JDChunk]:

    jd_id = jd["id"]

    job_title = jd["job_title"]
    department = jd["department"]

    responsibilities = ensure_list(
        jd["responsibilities"]
    )

    chunks = []

    for responsibility in responsibilities:

        sequence = responsibility[
            "sequence"
        ]

        description = responsibility[
            "description"
        ]

        time_percentage = responsibility.get(
            "time_percentage"
        )

        tasks = ensure_list(
            responsibility.get(
                "tasks",
                []
            )
        )

        # ================================================
        # 1. 先尝试完整 responsibility
        # ================================================

        full_content = (
            build_responsibility_content(
                job_title=job_title,
                department=department,
                description=description,
                tasks=tasks,
            )
        )

        # ================================================
        # 2. 如果不长：
        #
        # description + 所有tasks
        # 直接作为一个 chunk
        # ================================================

        if (
            count_tokens(full_content)
            <= RESPONSIBILITY_MAX_TOKENS
        ):

            chunks.append(
                JDChunk(
                    jd_id=jd_id,

                    chunk_key=(
                        f"v{CHUNK_VERSION}:"
                        f"jd:{jd_id}:"
                        f"responsibility:"
                        f"{sequence}:part:1"
                    ),

                    chunk_type="responsibility",

                    source_sequence=sequence,

                    part_index=1,

                    content=full_content,

                    metadata={
                        "job_title": job_title,
                        "department": department,
                        "responsibility_sequence": sequence,
                        "time_percentage": time_percentage,
                        "task_count": len(tasks),
                    },

                    is_embedding_target=True,
                )
            )

            continue

        # ================================================
        # 3. 超长 responsibility：
        #
        # 按 task 边界进行分组
        #
        # 每个 child 都重复 description
        # ================================================

        task_units = []

        for task in tasks:

            # --------------------------------------------
            # 一个 task 本身过长时，
            # 才进一步按句子拆
            # --------------------------------------------

            task_prefix = (
                f"岗位：{job_title}\n"
                f"部门：{department}\n"
                f"岗位职责：{description}\n"
                f"具体工作任务：\n"
            )

            available_tokens = (
                RESPONSIBILITY_MAX_TOKENS
                - count_tokens(task_prefix)
            )

            # 防止 description 本身太长
            available_tokens = max(
                available_tokens,
                50,
            )

            task_units.extend(
                split_by_sentence(
                    task,
                    available_tokens,
                )
            )

        # ================================================
        # 4. 贪心把 task 分组
        # ================================================

        groups = []

        current_tasks = []

        for task in task_units:

            candidate_tasks = (
                current_tasks
                +
                [task]
            )

            candidate_content = (
                build_responsibility_content(
                    job_title,
                    department,
                    description,
                    candidate_tasks,
                )
            )

            if (
                not current_tasks
                or count_tokens(candidate_content)
                <= RESPONSIBILITY_MAX_TOKENS
            ):

                current_tasks.append(
                    task
                )

            else:

                groups.append(
                    current_tasks
                )

                current_tasks = [
                    task
                ]

        if current_tasks:
            groups.append(
                current_tasks
            )

        # ================================================
        # 5. 生成多个 child chunk
        # ================================================

        for part_index, group in enumerate(
            groups,
            start=1,
        ):

            content = (
                build_responsibility_content(
                    job_title,
                    department,
                    description,
                    group,
                )
            )

            chunks.append(
                JDChunk(
                    jd_id=jd_id,

                    chunk_key=(
                        f"v{CHUNK_VERSION}:"
                        f"jd:{jd_id}:"
                        f"responsibility:"
                        f"{sequence}:"
                        f"part:{part_index}"
                    ),

                    chunk_type="responsibility",

                    source_sequence=sequence,

                    part_index=part_index,

                    content=content,

                    metadata={
                        "job_title": job_title,
                        "department": department,
                        "responsibility_sequence": sequence,
                        "time_percentage": time_percentage,

                        # 原 responsibility 有多少 task
                        "original_task_count": len(tasks),

                        # 当前 child 有多少 task
                        "chunk_task_count": len(group),

                        "is_split": True,
                    },

                    is_embedding_target=True,
                )
            )

    return chunks


# ============================================================
# 构建 qualification 文本
# ============================================================

def build_qualification_content(
    job_title: str,
    department: str,
    label: str,
    value: str,
) -> str:

    return (
        f"岗位：{job_title}\n"
        f"部门：{department}\n"
        f"{label}：{value}"
    )


# ============================================================
# 单个任职资格字段生成 Chunk
# ============================================================

def create_qualification_chunks(
    *,
    jd_id: int,
    job_title: str,
    department: str,
    chunk_type: str,
    label: str,
    value: str | None,
    item_index: int | None = None,
    is_embedding_target: bool = True,
) -> list[JDChunk]:

    if value is None:
        return []

    value = value.strip()

    if not value:
        return []

    base_content = (
        build_qualification_content(
            job_title,
            department,
            label,
            value,
        )
    )

    # ================================================
    # 不超长
    # ================================================

    if (
        count_tokens(base_content)
        <= QUALIFICATION_MAX_TOKENS
    ):

        part_values = [value]

    else:

        # ============================================
        # 超长 requirement
        # 按句子边界继续拆
        # ============================================

        prefix = (
            f"岗位：{job_title}\n"
            f"部门：{department}\n"
            f"{label}："
        )

        available_tokens = (
            QUALIFICATION_MAX_TOKENS
            - count_tokens(prefix)
        )

        available_tokens = max(
            available_tokens,
            50,
        )

        part_values = split_by_sentence(
            value,
            available_tokens,
        )

    chunks = []

    for part_index, part_value in enumerate(
        part_values,
        start=1,
    ):

        content = (
            build_qualification_content(
                job_title,
                department,
                label,
                part_value,
            )
        )

        index_part = (
            item_index
            if item_index is not None
            else 0
        )

        chunk_key = (
            f"v{CHUNK_VERSION}:"
            f"jd:{jd_id}:"
            f"{chunk_type}:"
            f"item:{index_part}:"
            f"part:{part_index}"
        )

        chunks.append(
            JDChunk(
                jd_id=jd_id,

                chunk_key=chunk_key,

                chunk_type=chunk_type,

                item_index=item_index,

                part_index=part_index,

                content=content,

                metadata={
                    "job_title": job_title,
                    "department": department,
                    "source_field": chunk_type,
                    "original_value": value,
                    "is_split": (
                        len(part_values) > 1
                    ),
                },

                is_embedding_target=(
                    is_embedding_target
                ),
            )
        )

    return chunks


# ============================================================
# 切任职资格
# ============================================================

def chunk_qualifications(
    jd: dict,
) -> list[JDChunk]:

    jd_id = jd["id"]

    job_title = jd["job_title"]
    department = jd["department"]

    chunks = []

    # ================================================
    # 1. 学历
    #
    # 这是硬条件。
    #
    # 我暂时也生成 chunk，
    # 但是标记为不需要 embedding。
    # 后面直接结构化规则判断。
    # ================================================

    chunks.extend(
        create_qualification_chunks(
            jd_id=jd_id,
            job_title=job_title,
            department=department,

            chunk_type="education",

            label="最低学历要求",

            value=jd[
                "minimum_education"
            ],

            is_embedding_target=False,
        )
    )

    # ================================================
    # 2. 专业背景
    #
    # 专业之间存在一定语义关系，
    # 可以参与 embedding。
    # ================================================

    chunks.extend(
        create_qualification_chunks(
            jd_id=jd_id,
            job_title=job_title,
            department=department,

            chunk_type="education_background",

            label="专业背景要求",

            value=jd[
                "education_background"
            ],

            is_embedding_target=True,
        )
    )

    # ================================================
    # 3. 工作经验
    #
    # 主要用于硬条件判断。
    # ================================================

    chunks.extend(
        create_qualification_chunks(
            jd_id=jd_id,
            job_title=job_title,
            department=department,

            chunk_type="work_experience",

            label="工作经验要求",

            value=jd[
                "work_experience_raw"
            ],

            is_embedding_target=False,
        )
    )

    # ================================================
    # 4. competencies
    #
    # 一条 competency = 一个天然语义 requirement
    #
    # 不把：
    #
    # 团队合作
    # +
    # 熟练使用Office
    # +
    # 财政行业知识
    #
    # 强行合成同一个向量。
    # ================================================

    competencies = ensure_list(
        jd["competencies"]
    )

    for item_index, competency in enumerate(
        competencies,
        start=1,
    ):

        chunks.extend(
            create_qualification_chunks(
                jd_id=jd_id,

                job_title=job_title,

                department=department,

                chunk_type="competency",

                label="任职资格要求",

                value=competency,

                item_index=item_index,

                is_embedding_target=True,
            )
        )

    return chunks


# ============================================================
# 单条 JD 总切分入口
# ============================================================

def chunk_jd(
    jd: dict,
) -> list[JDChunk]:

    chunks = []

    chunks.extend(
        chunk_responsibilities(
            jd
        )
    )

    chunks.extend(
        chunk_qualifications(
            jd
        )
    )

    return chunks


# ============================================================
# 初始化 jd_chunks 表
# ============================================================

async def init_chunk_table(
    cursor,
):

    await cursor.execute(
        CREATE_CHUNK_TABLE_SQL
    )

    await cursor.execute(
        CREATE_JD_ID_INDEX_SQL
    )

    await cursor.execute(
        CREATE_CHUNK_TYPE_INDEX_SQL
    )


# ============================================================
# 写入 chunk
# ============================================================

async def insert_chunks(
    cursor,
    chunks: list[JDChunk],
):

    for chunk in chunks:

        await cursor.execute(
            INSERT_CHUNK_SQL,
            (
                chunk.jd_id,

                chunk.chunk_key,

                chunk.chunk_type,

                chunk.source_sequence,

                chunk.part_index,

                chunk.item_index,

                chunk.content,

                Jsonb(
                    chunk.metadata
                ),

                chunk.is_embedding_target,

                CHUNK_VERSION,
            )
        )


# ============================================================
# 主程序
# ============================================================

async def main():

    # ========================================================
    # 1. SSH
    # ========================================================

    async with asyncssh.connect(
        SSH_HOST,

        username=SSH_USERNAME,

        password=SSH_PASSWORD,

        known_hosts=None,
    ) as ssh_conn:

        listener = (
            await ssh_conn.forward_local_port(
                LOCAL_HOST,
                LOCAL_PORT,

                REMOTE_DB_HOST,
                REMOTE_DB_PORT,
            )
        )

        print(
            "SSH 通道成功，端口转发已建立"
        )

        try:

            # =================================================
            # 2. PostgreSQL
            # =================================================

            async with await psycopg.AsyncConnection.connect(
                host=LOCAL_HOST,
                port=LOCAL_PORT,

                dbname=DB_NAME,

                user=DB_USER,

                password=DB_PASSWORD,

                connect_timeout=5,

                row_factory=dict_row,
            ) as db_conn:

                print(
                    "PostgreSQL 连接成功"
                )

                async with db_conn.cursor() as cursor:

                    # =========================================
                    # 3. 创建 chunk 表
                    # =========================================

                    await init_chunk_table(
                        cursor
                    )

                    # =========================================
                    # 4. 读取全部 JD
                    # =========================================

                    await cursor.execute(
                        SELECT_JDS_SQL
                    )

                    jd_list = (
                        await cursor.fetchall()
                    )

                    print(
                        f"读取 JD 数量："
                        f"{len(jd_list)}"
                    )

                    total_chunks = 0

                    # =========================================
                    # 5. 对每一份 JD 切块
                    # =========================================

                    for jd in jd_list:

                        chunks = chunk_jd(
                            jd
                        )

                        await insert_chunks(
                            cursor,
                            chunks,
                        )

                        total_chunks += len(
                            chunks
                        )

                        print(
                            f"[CHUNK] "
                            f"jd_id={jd['id']}, "
                            f"岗位={jd['job_title']}, "
                            f"生成={len(chunks)} 个"
                        )

                print("=" * 60)

                print(
                    f"JD 总数：{len(jd_list)}"
                )

                print(
                    f"Chunk 总数：{total_chunks}"
                )

                print("=" * 60)

        finally:

            listener.close()

            await listener.wait_closed()

            print(
                "SSH 通道已关闭"
            )


# ============================================================
# Windows asyncio
# ============================================================

def loop_factory():

    return asyncio.SelectorEventLoop(
        selectors.SelectSelector()
    )


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main(),
        loop_factory=loop_factory,
    )