import asyncio
import hashlib
import json
import selectors
import dotenv,os
from pathlib import Path

import asyncssh
import psycopg
from psycopg.types.json import Jsonb


# ============================================================
# 配置
project_root = Path(__file__).resolve().parents[1]
input_path = project_root / "jobs_raw.json"

INPUT_FILE = Path(input_path)

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


# 当前 JD schema 版本
SCHEMA_VERSION = 1


# ============================================================
# SQL
# ============================================================

INSERT_JD_SQL = """
INSERT INTO job_descriptions (
    job_title,
    department,
    responsibilities,
    minimum_education,
    education_background,
    work_experience_raw,
    competencies,
    content_hash,
    source_payload,
    schema_version
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
ON CONFLICT (content_hash)
DO NOTHING
RETURNING id;
"""


# ============================================================
# 读取 JSON 文件
# ============================================================

def load_jd_data(
    file_path: Path,
) -> list[dict]:

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    if not isinstance(data, list):
        raise TypeError(
            "JD JSON 顶层结构必须是 list"
        )

    return data


# ============================================================
# content_hash
# ============================================================

def calculate_content_hash(
    jd: dict,
) -> str:
    """
    根据真正的 JD 业务内容计算 SHA256。
    """

    qualification = jd["qualification"]

    business_content = {
        "job_title": jd["job_title"],
        "department": jd["department"],
        "responsibilities": jd["responsibilities"],

        "qualification": {
            "minimum_education": qualification[
                "minimum_education"
            ],

            "education_background": qualification[
                "education_background"
            ],

            "work_experience_raw": qualification.get(
                "work_experience_raw"
            ),

            "competencies": qualification[
                "competencies"
            ],
        },
    }

    canonical_json = json.dumps(
        business_content,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()


# ============================================================
# 检查单条 JD 是否包含数据库需要的字段
# ============================================================

def validate_required_fields(
    jd: dict,
    index: int,
) -> None:

    required_top_fields = (
        "job_title",
        "department",
        "responsibilities",
        "qualification",
    )

    for field in required_top_fields:

        if field not in jd:
            raise ValueError(
                f"JD index={index} "
                f"缺少字段: {field}"
            )

    qualification = jd["qualification"]

    required_qualification_fields = (
        "minimum_education",
        "education_background",
        "competencies",
    )

    for field in required_qualification_fields:

        if field not in qualification:
            raise ValueError(
                f"JD index={index} "
                f"qualification 缺少字段: {field}"
            )


# ============================================================
# 插入单条 JD
# ============================================================

async def insert_jd(
    cursor,
    jd: dict,
) -> int | None:

    qualification = jd["qualification"]

    content_hash = calculate_content_hash(
        jd
    )

    await cursor.execute(
        INSERT_JD_SQL,
        (
            jd["job_title"],

            jd["department"],

            Jsonb(
                jd["responsibilities"]
            ),

            qualification[
                "minimum_education"
            ],

            qualification[
                "education_background"
            ],

            # 允许 None
            qualification.get(
                "work_experience_raw"
            ),

            qualification[
                "competencies"
            ],

            content_hash,

            Jsonb(jd),

            SCHEMA_VERSION,
        ),
    )

    row = await cursor.fetchone()

    if row is None:
        return None

    return row[0]


# ============================================================
# 主程序
# ============================================================

async def main():

    # --------------------------------------------------------
    # 1. 先读取 JSON
    # --------------------------------------------------------

    jd_list = load_jd_data(
        INPUT_FILE
    )

    print(
        f"读取 JD JSON 成功，共 {len(jd_list)} 条"
    )

    # --------------------------------------------------------
    # 2. SSH 连接
    # --------------------------------------------------------

    async with asyncssh.connect(
        SSH_HOST,
        username=SSH_USERNAME,
        password=SSH_PASSWORD,
        known_hosts=None,
    ) as ssh_conn:

        listener = await ssh_conn.forward_local_port(
            LOCAL_HOST,
            LOCAL_PORT,
            REMOTE_DB_HOST,
            REMOTE_DB_PORT,
        )

        print(
            "SSH 通道成功，端口转发已建立"
        )

        try:

            # ------------------------------------------------
            # 4. PostgreSQL 异步连接
            # ------------------------------------------------

            async with await psycopg.AsyncConnection.connect(
                host=LOCAL_HOST,
                port=LOCAL_PORT,
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                connect_timeout=5,
            ) as db_conn:

                print(
                    "PostgreSQL 连接成功"
                )

                inserted_count = 0
                skipped_count = 0

                # --------------------------------------------
                # 整批 JD 使用同一个事务
                # --------------------------------------------

                async with db_conn.cursor() as cursor:

                    for index, jd in enumerate(
                        jd_list
                    ):

                        validate_required_fields(
                            jd,
                            index,
                        )

                        try:

                            inserted_id = await insert_jd(
                                cursor,
                                jd,
                            )

                            if inserted_id is None:

                                skipped_count += 1

                                print(
                                    f"[SKIP] "
                                    f"index={index}, "
                                    f"job_id={jd.get('job_id')}, "
                                    f"job_title={jd['job_title']} "
                                    f"数据已存在"
                                )

                            else:

                                inserted_count += 1

                                print(
                                    f"[INSERT] "
                                    f"index={index}, "
                                    f"id={inserted_id}, "
                                    f"job_id={jd.get('job_id')}, "
                                    f"job_title={jd['job_title']}"
                                )

                        except Exception:

                            print(
                                "\n"
                                "插入 JD 失败："
                            )

                            print(
                                f"index = {index}"
                            )

                            print(
                                f"job_id = "
                                f"{jd.get('job_id')}"
                            )

                            print(
                                f"job_title = "
                                f"{jd.get('job_title')}"
                            )

                            # 继续把异常抛出去
                            # 整个事务随后 rollback
                            raise

                # --------------------------------------------
                # async with db_conn 正常退出后自动 commit
                # --------------------------------------------

                print()
                print("=" * 60)

                print(
                    f"JD 总数: {len(jd_list)}"
                )

                print(
                    f"成功插入: {inserted_count}"
                )

                print(
                    f"重复跳过: {skipped_count}"
                )

                print("=" * 60)

                # --------------------------------------------
                # 5. 查询数据库当前 JD 总数
                # --------------------------------------------

                async with db_conn.cursor() as cursor:

                    await cursor.execute(
                        """
                        SELECT COUNT(*)
                        FROM job_descriptions;
                        """
                    )

                    row = await cursor.fetchone()

                    print(
                        f"数据库当前 JD 数量: {row[0]}"
                    )

        finally:

            # ------------------------------------------------
            # 6. 关闭 SSH 转发
            # ------------------------------------------------

            listener.close()

            await listener.wait_closed()

            print(
                "SSH 端口转发已关闭"
            )


# ============================================================
# Windows psycopg async
# ============================================================

def loop_factory():

    return asyncio.SelectorEventLoop(
        selectors.SelectSelector()
    )


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main(),
        loop_factory=loop_factory,
    )