# scripts/embed_jd_chunks.py

from __future__ import annotations

import logging
import os

import numpy as np
import psycopg
import torch
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel
from pgvector.psycopg import register_vector
from sshtunnel import SSHTunnelForwarder

# =========================
# 配置
# =========================

load_dotenv()

MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

MODEL_PATH = os.getenv("BGE_MODEL_PATH")
DEVICE = os.getenv("BGE_DEVICE")
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "8"))
MAX_LENGTH = int(os.getenv("BGE_MAX_LENGTH", "1024"))

# 日志
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("chunk_embedding.log", encoding="utf-8")
Formater = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
file_handler.setFormatter(Formater)
console_handler.setFormatter(Formater)


logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler]
)

logger = logging.getLogger(__name__)

# 远程连接

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

# =========================
# 数据库
# =========================

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


# =========================
# Embedding Model
# =========================

def load_embedding_model() -> BGEM3FlagModel:
    """
    从本地目录加载 BGE-M3。
    """

    if not MODEL_PATH:
        raise ValueError("没有配置 BGE_MODEL_PATH")

    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(
            f"BGE-M3 模型目录不存在: {MODEL_PATH}"
        )

    # 如果没有显式设置设备，自动检测
    device = DEVICE

    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # GPU 才启用 fp16
    use_fp16 = device.startswith("cuda")

    logger.info(
        "加载 BGE-M3: path=%s, device=%s, fp16=%s",
        MODEL_PATH,
        device,
        use_fp16,
    )

    model = BGEM3FlagModel(
        MODEL_PATH,
        use_fp16=use_fp16,
        devices=[device],
    )

    return model


# =========================
# 查询待 Embedding Chunk
# =========================

def fetch_unembedded_chunks(
    conn: psycopg.Connection,
    limit: int,
) -> list[tuple[int, str]]:
    """
    查询需要 embedding、但目前还没有 embedding 的 chunk。
    """

    sql = """
        SELECT
            id,
            content
        FROM jd_chunks
        WHERE is_embedding_target = TRUE
          AND embedding IS NULL
          AND btrim(content) <> ''
        ORDER BY id
        LIMIT %s;
    """

    with conn.cursor() as cur:
        cur.execute(sql, (limit,))
        rows = cur.fetchall()

    # SELECT 同样会开启事务，
    # 提交掉，避免模型推理过程中一直处于 idle in transaction
    conn.commit()

    return rows


# =========================
# BGE-M3 Embedding
# =========================

def encode_chunks(
    model: BGEM3FlagModel,
    texts: list[str],
) -> np.ndarray:
    """
    将 chunk 文本批量转换成 Dense Embedding。
    """

    result = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,

        # 当前 RAG 只使用 Dense Embedding
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    vectors = result["dense_vecs"]

    vectors = np.asarray(
        vectors,
        dtype=np.float32,
    )

    # 防止模型配置或者代码修改导致向量维度错误
    if vectors.ndim != 2:
        raise ValueError(
            f"Embedding 输出维度异常: {vectors.shape}"
        )

    if vectors.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding 维度错误: "
            f"期望 {EMBEDDING_DIM}, "
            f"实际 {vectors.shape[1]}"
        )

    return vectors


# =========================
# 写回数据库
# =========================

def save_embeddings(
    conn: psycopg.Connection,
    chunk_ids: list[int],
    vectors: np.ndarray,
) -> None:
    """
    将 embedding 写回 jd_chunks。
    """

    if len(chunk_ids) != len(vectors):
        raise ValueError(
            "chunk 数量和 embedding 数量不一致"
        )

    sql = """
        UPDATE jd_chunks
        SET
            embedding = %s,
            embedding_model = %s,
            embedded_at = CURRENT_TIMESTAMP
        WHERE id = %s;
    """

    params = [
        (
            vector,
            MODEL_NAME,
            chunk_id,
        )
        for chunk_id, vector in zip(
            chunk_ids,
            vectors,
        )
    ]

    try:
        with conn.cursor() as cur:
            cur.executemany(sql, params)

        conn.commit()

    except Exception:
        conn.rollback()
        raise


# =========================
# Pipeline
# =========================

def run_embedding_pipeline():
    tunnel = None
    conn = None

    try:
        # 1. 建立 SSH Tunnel
        print("正在建立 SSH Tunnel...")
        tunnel = create_ssh_tunnel()

        # 2. 通过 Tunnel 连接 PostgreSQL
        print("正在连接 PostgreSQL...")
        conn = create_connection(tunnel)
        print("PostgreSQL 连接成功")

        # 3. 加载 BGE-M3
        print("正在加载 BGE-M3...")
        model = load_embedding_model()
        print("BGE-M3 加载成功")
        # 4. 后面执行你的 embedding pipeline
        print("开始执行 embedding pipeline...")

        while True:

            rows = fetch_unembedded_chunks(
                conn,
                limit=BATCH_SIZE,
            )

            if not rows:
                break

            chunk_ids = [
                row[0]
                for row in rows
            ]

            texts = [
                row[1]
                for row in rows
            ]

            vectors = encode_chunks(
                model,
                texts,
            )

            save_embeddings(
                conn,
                chunk_ids,
                vectors,
            )

    finally:

        # 顺序很重要：
        # 先关闭 DB
        if conn is not None:
            conn.close()
            print("PostgreSQL 连接已关闭")

        # 再关闭 SSH Tunnel
        if tunnel is not None:
            tunnel.stop()
            print("SSH Tunnel 已关闭")

        print("embedding pipeline 已完成")


if __name__ == "__main__":
    run_embedding_pipeline()
