import psycopg
import os
import dotenv
from FlagEmbedding import BGEM3FlagModel
from sshtunnel import SSHTunnelForwarder
from pgvector.psycopg import register_vector

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


model = BGEM3FlagModel(r'/app/RAG知识库构建/embedding_tools/bge-m3',
                       use_fp16=False,
                       devices=["cpu"])
# Setting use_fp16 to True speeds up computation with a slight performance degradation
text = "具有较强的学习能力"


query_vector = model.encode(text, batch_size=4,max_length=1024,return_dense=True,
    return_sparse=False,
    return_colbert_vecs=False)['dense_vecs']

print(query_vector)
# [[0.6265, 0.3477], [0.3499, 0.678 ]]


def create_ssh_tunnel() -> SSHTunnelForwarder:

    tunnel = SSHTunnelForwarder(
        (
            os.getenv("SSH_HOST"),
            int(os.getenv("SSH_PORT", "22")),
        ),

        ssh_username=os.getenv("SSH_USERNAME"),
        ssh_password=os.getenv("SSH_PASSWORD"),

        # 从 Ubuntu SSH Server 的角度连接 PostgreSQL
        remote_bind_address=(
            "127.0.0.1",
            int(os.getenv("REMOTE_DB_PORT", "5432")),
        ),

        # Windows 本机创建转发端口
        # 0 表示让系统自动选择空闲端口
        local_bind_address=(
            "127.0.0.1",
            0,
        ),
    )

    tunnel.start()

    print(
        f"SSH Tunnel 已建立: "
        f"127.0.0.1:{tunnel.local_bind_port}"
        f" -> Ubuntu:127.0.0.1:5432"
    )

    return tunnel

def create_connection(
    tunnel: SSHTunnelForwarder,
) -> psycopg.Connection:
    """
    通过 SSH Tunnel 连接 PostgreSQL。
    """

    conn = psycopg.connect(
        host="127.0.0.1",

        # 使用 SSH Tunnel 在 Windows 上创建的本地端口
        port=tunnel.local_bind_port,

        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),

        connect_timeout=10,
    )

    register_vector(conn)

    return conn

def search_similar_chunks(
    conn,
    query_vector,
    top_k: int = 20,
):
    """
    使用 pgvector 查询与 query_vector 最相似的 JD chunks。

    <=> 表示 cosine distance
    cosine similarity = 1 - cosine distance
    """

    sql = """
        SELECT
            id,
            jd_id,
            chunk_key,
            chunk_type,
            content,
            metadata,
            embedding_model,
            1 - (embedding <=> %s) AS similarity
        FROM jd_chunks
        WHERE is_embedding_target = TRUE
          AND embedding IS NOT NULL
        ORDER BY embedding <=> %s
        LIMIT %s;
    """

    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                query_vector,
                query_vector,
                top_k,
            ),
        )

        rows = cur.fetchall()

    return rows

if __name__ == "__main__":
    tunnel = create_ssh_tunnel()
    conn = create_connection(tunnel)

    rows = search_similar_chunks(
        conn=conn,
        query_vector=query_vector,
        top_k=20,
    )

    for row in rows:
        print(row)