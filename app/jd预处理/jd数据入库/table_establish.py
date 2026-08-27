import asyncio
import selectors

import asyncssh
import psycopg


CREATE_JD_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS job_descriptions (
    id BIGSERIAL PRIMARY KEY,

    -- 岗位基础信息
    job_title TEXT NOT NULL,
    department TEXT NOT NULL,

    -- 岗位职责
    -- 保存：
    -- sequence
    -- description
    -- time_percentage
    -- tasks
    responsibilities JSONB NOT NULL,

    -- 任职资格
    minimum_education TEXT NOT NULL,
    education_background TEXT NOT NULL,
    work_experience_raw TEXT,

    -- Python list[str] 对应 PostgreSQL TEXT[]
    competencies TEXT[] NOT NULL DEFAULT '{}',

    -- 用于防止完全相同 JD 重复导入
    content_hash VARCHAR(64) NOT NULL UNIQUE,

    -- 保存最终人工确认后的完整 JSON 快照
    source_payload JSONB,

    -- 数据结构版本
    schema_version INTEGER NOT NULL DEFAULT 1,

    -- 时间字段
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- 最基本的数据约束
    CONSTRAINT ck_job_title_not_blank
        CHECK (btrim(job_title) <> ''),

    CONSTRAINT ck_department_not_blank
        CHECK (btrim(department) <> ''),

    CONSTRAINT ck_minimum_education_not_blank
        CHECK (btrim(minimum_education) <> ''),

    CONSTRAINT ck_education_background_not_blank
        CHECK (btrim(education_background) <> ''),

    -- responsibilities 必须是 JSON 数组
    CONSTRAINT ck_responsibilities_array
        CHECK (
            jsonb_typeof(responsibilities) = 'array'
        )
);
"""

async def main():

    async with asyncssh.connect(
        "192.168.181.128",
        username="dawn",
        password="123456",
        known_hosts=None,
    ) as ssh_conn:

        listener = await ssh_conn.forward_local_port(
            "127.0.0.1",
            15432,
            "127.0.0.1",
            5432,
        )

        print("SSH通道成功，端口转发已建立")

        try:
            async with await psycopg.AsyncConnection.connect(
                host="127.0.0.1",
                port=15432,
                dbname="resume_agent",
                user="resume_user",
                password="123456",
                connect_timeout=5,
            ) as db_conn:

                async with db_conn.cursor() as cursor:

                    # =========================================
                    # 4. 创建 JD 表
                    # =========================================

                    await cursor.execute(
                        CREATE_JD_TABLE_SQL
                    )

                    print(
                        "job_descriptions 表创建成功"
                    )

                    # =========================================
                    # 5. 查询确认表是否存在
                    # =========================================

                    await cursor.execute(
                        """
                        SELECT table_name
                        FROM information_schema.tables
                        WHERE table_schema = 'public'
                          AND table_name = 'job_descriptions';
                        """
                    )

                    row = await cursor.fetchone()

                    print(
                        "查询结果:",
                        row
                    )

        finally:
            listener.close()
            await listener.wait_closed()


def loop_factory():
    return asyncio.SelectorEventLoop(
        selectors.SelectSelector()
    )


if __name__ == "__main__":
    asyncio.run(
        main(),
        loop_factory=loop_factory
    )