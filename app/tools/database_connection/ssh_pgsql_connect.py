from __future__ import annotations

import logging
import os
import select
import socketserver
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence
from dotenv import load_dotenv

import paramiko
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

_ = load_dotenv()
logger = logging.getLogger(__name__)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"缺少环境变量: {name}")
    return value.strip()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    raise ValueError(
        f"环境变量 {name} 应为 true/false、1/0、yes/no，实际为: {value!r}"
    )


@dataclass(frozen=True, slots=True)
class PostgresSSHConfig:
    """
    Windows -> SSH -> 虚拟机 -> PostgreSQL 的连接配置。

    注意：
    - ssh_host: Windows 能访问到的虚拟机 SSH 地址。
    - db_host: 从虚拟机自身视角访问 PostgreSQL 的地址。
      如果 PostgreSQL 就部署在 SSH 主机上，通常填 127.0.0.1。
    """

    # SSH
    ssh_host: str
    ssh_username: str
    ssh_port: int = 22
    ssh_password: str | None = None
    ssh_private_key: str | None = None
    ssh_private_key_password: str | None = None
    ssh_connect_timeout: float = 10.0
    allow_unknown_host_key: bool = False

    # PostgreSQL
    db_name: str = ""
    db_user: str = ""
    db_password: str = ""
    db_host: str = "127.0.0.1"
    db_port: int = 5432
    db_connect_timeout: float = 10.0

    # 本地转发
    local_bind_host: str = "127.0.0.1"

    # 连接池
    pool_min_size: int = 1
    pool_max_size: int = 5
    pool_timeout: float = 10.0
    pool_max_idle: float = 300.0
    pool_max_lifetime: float = 3600.0

    @classmethod
    def from_env(cls) -> "PostgresSSHConfig":
        """
        从环境变量构造配置。

        必填：
            SSH_HOST
            SSH_USER
            POSTGRES_DB
            POSTGRES_USER
            POSTGRES_PASSWORD

        SSH_PASSWORD / SSH_PRIVATE_KEY 至少配置一种；
        如果两者都不配置，则 Paramiko 会尝试 SSH Agent / 默认密钥。
        """
        return cls(
            ssh_host=_require_env("SSH_HOST"),
            ssh_username=_require_env("SSH_USERNAME"),
            ssh_port=_env_int("SSH_PORT", 22),
            ssh_password=os.getenv("SSH_PASSWORD") or None,
            ssh_private_key=os.getenv("SSH_PRIVATE_KEY") or None,
            ssh_private_key_password=(
                os.getenv("SSH_PRIVATE_KEY_PASSWORD") or None
            ),
            ssh_connect_timeout=_env_float(
                "SSH_CONNECT_TIMEOUT", 10.0
            ),
            allow_unknown_host_key=_env_bool(
                "SSH_ALLOW_UNKNOWN_HOST_KEY", False
            ),
            db_name=_require_env("DB_NAME"),
            db_user=_require_env("DB_USER"),
            db_password=_require_env("DB_PASSWORD"),
            db_host=os.getenv("REMOTE_DB_HOST", "127.0.0.1"),
            db_port=_env_int("REMOTE_DB_PORT", 5432),
            db_connect_timeout=_env_float(
                "POSTGRES_CONNECT_TIMEOUT", 10.0
            ),
            local_bind_host=os.getenv(
                "DB_LOCAL_BIND_HOST", "127.0.0.1"
            ),
            pool_min_size=_env_int("DB_POOL_MIN_SIZE", 1),
            pool_max_size=_env_int("DB_POOL_MAX_SIZE", 5),
            pool_timeout=_env_float("DB_POOL_TIMEOUT", 10.0),
            pool_max_idle=_env_float(
                "DB_POOL_MAX_IDLE", 300.0
            ),
            pool_max_lifetime=_env_float(
                "DB_POOL_MAX_LIFETIME", 3600.0
            ),
        )


class _SSHForwardServer(socketserver.ThreadingTCPServer):
    """
    本地 TCP Server。
    每收到一个本地 PostgreSQL 连接，就通过已有 SSH Transport
    打开一个 direct-tcpip channel 转发到远端 PostgreSQL。
    """

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        ssh_transport: paramiko.Transport,
        remote_address: tuple[str, int],
    ) -> None:
        self.ssh_transport = ssh_transport
        self.remote_address = remote_address
        super().__init__(server_address, _SSHForwardHandler)


class _SSHForwardHandler(socketserver.BaseRequestHandler):
    server: _SSHForwardServer

    def handle(self) -> None:
        channel: paramiko.Channel | None = None

        try:
            source_address = self.request.getpeername()

            channel = self.server.ssh_transport.open_channel(
                kind="direct-tcpip",
                dest_addr=self.server.remote_address,
                src_addr=source_address,
            )

            if channel is None:
                raise RuntimeError(
                    "SSH direct-tcpip channel 创建失败"
                )

            while True:
                readable, _, _ = select.select(
                    [self.request, channel],
                    [],
                    [],
                )

                if self.request in readable:
                    data = self.request.recv(65536)
                    if not data:
                        break
                    channel.sendall(data)

                if channel in readable:
                    data = channel.recv(65536)
                    if not data:
                        break
                    self.request.sendall(data)

        except (
            OSError,
            EOFError,
            paramiko.SSHException,
        ) as exc:
            logger.info(
                "SSH tunnel connection closed: %s",
                exc,
            )

        finally:
            if channel is not None:
                channel.close()


class SSHLocalForwarder:
    """
    管理一条：
        Windows localhost:随机端口
            -> SSH
            -> 虚拟机 db_host:db_port
    的本地端口转发。
    """

    def __init__(
        self,
        config: PostgresSSHConfig,
    ) -> None:
        self._config = config

        self._ssh_client: paramiko.SSHClient | None = None
        self._server: _SSHForwardServer | None = None
        self._server_thread: threading.Thread | None = None
        self._local_port: int | None = None

        self._lock = threading.RLock()

    @property
    def local_port(self) -> int:
        if self._local_port is None:
            raise RuntimeError("SSH tunnel 尚未启动")
        return self._local_port

    @property
    def is_running(self) -> bool:
        return (
            self._ssh_client is not None
            and self._server is not None
            and self._server_thread is not None
            and self._server_thread.is_alive()
        )

    def start(self) -> None:
        with self._lock:
            if self.is_running:
                return

            client = paramiko.SSHClient()
            client.load_system_host_keys()

            if self._config.allow_unknown_host_key:
                # 仅建议本地开发环境使用。
                client.set_missing_host_key_policy(
                    paramiko.AutoAddPolicy()
                )
            else:
                client.set_missing_host_key_policy(
                    paramiko.RejectPolicy()
                )

            connect_kwargs: dict[str, Any] = {
                "hostname": self._config.ssh_host,
                "port": self._config.ssh_port,
                "username": self._config.ssh_username,
                "timeout": self._config.ssh_connect_timeout,
                "banner_timeout": self._config.ssh_connect_timeout,
                "auth_timeout": self._config.ssh_connect_timeout,
            }

            if self._config.ssh_password:
                connect_kwargs["password"] = (
                    self._config.ssh_password
                )

            if self._config.ssh_private_key:
                connect_kwargs["key_filename"] = (
                    self._config.ssh_private_key
                )

            if self._config.ssh_private_key_password:
                connect_kwargs["passphrase"] = (
                    self._config.ssh_private_key_password
                )

            # 没显式提供密码/密钥时，允许使用 ssh-agent 和默认私钥。
            explicit_auth = bool(
                self._config.ssh_password
                or self._config.ssh_private_key
            )
            connect_kwargs["allow_agent"] = not explicit_auth
            connect_kwargs["look_for_keys"] = not explicit_auth

            try:
                client.connect(**connect_kwargs)

                transport = client.get_transport()
                if transport is None or not transport.is_active():
                    raise RuntimeError(
                        "SSH 连接已建立，但 Transport 不可用"
                    )

                # SSH keepalive，避免开发时长时间无操作导致 tunnel 被回收。
                transport.set_keepalive(30)

                server = _SSHForwardServer(
                    server_address=(
                        self._config.local_bind_host,
                        0,  # 让系统随机分配空闲端口，避免端口冲突。
                    ),
                    ssh_transport=transport,
                    remote_address=(
                        self._config.db_host,
                        self._config.db_port,
                    ),
                )

                thread = threading.Thread(
                    target=server.serve_forever,
                    name="postgres-ssh-tunnel",
                    daemon=True,
                )
                thread.start()

                local_port = int(
                    server.server_address[1]
                )

                self._ssh_client = client
                self._server = server
                self._server_thread = thread
                self._local_port = local_port

                logger.info(
                    "SSH tunnel started: %s:%s -> %s:%s via %s:%s",
                    self._config.local_bind_host,
                    local_port,
                    self._config.db_host,
                    self._config.db_port,
                    self._config.ssh_host,
                    self._config.ssh_port,
                )

            except Exception:
                client.close()
                raise

    def close(self) -> None:
        with self._lock:
            server = self._server
            thread = self._server_thread
            client = self._ssh_client

            self._server = None
            self._server_thread = None
            self._ssh_client = None
            self._local_port = None

            if server is not None:
                server.shutdown()
                server.server_close()

            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)

            if client is not None:
                client.close()

            logger.info("SSH tunnel closed")

    def __enter__(self) -> "SSHLocalForwarder":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()


class PostgresSSHPool:
    """
    SSH Tunnel + psycopg ConnectionPool 的统一封装。

    生命周期：
        start()
            1. 启动 SSH tunnel
            2. 创建 psycopg_pool.ConnectionPool
            3. 预热 min_size 个连接

        connection()
            从池中借一个连接；with 结束后自动归还。
            正常退出自动 commit，异常退出自动 rollback。

        close()
            1. 关闭连接池
            2. 关闭 SSH tunnel
    """

    def __init__(
        self,
        config: PostgresSSHConfig,
    ) -> None:
        self._config = config
        self._tunnel = SSHLocalForwarder(config)

        self._pool: ConnectionPool | None = None
        self._lock = threading.RLock()

    @property
    def is_started(self) -> bool:
        return self._pool is not None

    def start(self) -> None:
        with self._lock:
            if self._pool is not None:
                return

            self._tunnel.start()

            pool: ConnectionPool | None = None

            try:
                pool = ConnectionPool(
                    kwargs={
                        # PostgreSQL 客户端实际连接的是 Windows 本地隧道端口。
                        "host": self._config.local_bind_host,
                        "port": self._tunnel.local_port,
                        "dbname": self._config.db_name,
                        "user": self._config.db_user,
                        "password": self._config.db_password,
                        "connect_timeout": (
                            self._config.db_connect_timeout
                        ),
                        # 查询结果默认返回 dict，业务层使用更方便。
                        "row_factory": dict_row,
                    },
                    min_size=self._config.pool_min_size,
                    max_size=self._config.pool_max_size,
                    timeout=self._config.pool_timeout,
                    max_idle=self._config.pool_max_idle,
                    max_lifetime=(
                        self._config.pool_max_lifetime
                    ),
                    # 从池中借连接前做基本可用性检查。
                    check=ConnectionPool.check_connection,
                    # 显式控制生命周期，避免 import 模块时就连数据库。
                    open=False,
                    name="resume-agent-postgres",
                )

                pool.open(
                    wait=True,
                    timeout=self._config.pool_timeout,
                )

                self._pool = pool

                logger.info(
                    "PostgreSQL connection pool started: "
                    "min=%s, max=%s, local_port=%s",
                    self._config.pool_min_size,
                    self._config.pool_max_size,
                    self._tunnel.local_port,
                )

            except Exception:
                if pool is not None:
                    pool.close()
                self._tunnel.close()
                raise

    def _require_pool(self) -> ConnectionPool:
        pool = self._pool
        if pool is None:
            raise RuntimeError(
                "PostgresSSHPool 尚未启动，请先调用 start()"
            )
        return pool

    @contextmanager
    def connection(
        self,
        timeout: float | None = None,
    ) -> Iterator[psycopg.Connection]:
        """
        推荐的数据库使用方式：

            with db.connection() as conn:
                row = conn.execute(...).fetchone()

        with 正常结束：
            transaction commit + connection 归还连接池

        with 内抛异常：
            transaction rollback + connection 归还连接池
        """
        pool = self._require_pool()

        with pool.connection(timeout=timeout) as conn:
            yield conn

    def fetch_one(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute(query, params).fetchone()
            return row

    def fetch_all(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return list(rows)

    def execute(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] | None = None,
    ) -> int:
        """
        执行 INSERT / UPDATE / DELETE。

        返回 cursor.rowcount。
        commit 由 connection() 上下文自动完成。
        """
        with self.connection() as conn:
            cursor = conn.execute(query, params)
            return cursor.rowcount

    def health_check(self) -> bool:
        row = self.fetch_one(
            "SELECT 1 AS ok"
        )
        return bool(row and row.get("ok") == 1)

    def pool_stats(self) -> dict[str, Any]:
        """
        返回 psycopg_pool 当前统计信息，后续调连接池大小时很有用。
        """
        return dict(
            self._require_pool().get_stats()
        )

    def close(self) -> None:
        with self._lock:
            pool = self._pool
            self._pool = None

            try:
                if pool is not None:
                    pool.close()
                    logger.info(
                        "PostgreSQL connection pool closed"
                    )
            finally:
                self._tunnel.close()

    def __enter__(self) -> "PostgresSSHPool":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()


# ============================================================
# 可选：应用级单例
# ============================================================

_default_db: PostgresSSHPool | None = None
_default_db_lock = threading.RLock()


def init_default_db(
    config: PostgresSSHConfig | None = None,
) -> PostgresSSHPool:
    """
    应用启动时调用一次。

    如果不传 config，则从环境变量读取。
    """
    global _default_db

    with _default_db_lock:
        if _default_db is not None:
            return _default_db

        db = PostgresSSHPool(
            config or PostgresSSHConfig.from_env()
        )
        db.start()

        _default_db = db
        return db


def get_default_db() -> PostgresSSHPool:
    """
    业务代码获取已经启动的数据库工具。
    不会隐式创建连接，避免 import 时产生副作用。
    """
    if _default_db is None:
        raise RuntimeError(
            "数据库尚未初始化，请在应用启动阶段调用 "
            "init_default_db()"
        )

    return _default_db


def close_default_db() -> None:
    """
    应用退出时调用一次。
    """
    global _default_db

    with _default_db_lock:
        db = _default_db
        _default_db = None

        if db is not None:
            db.close()


if __name__ == "__main__":
    # 简单连通性测试：
    #
    #   python postgres_ssh_pool.py
    #
    # 运行前配置好环境变量。
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    config = PostgresSSHConfig.from_env()

    with PostgresSSHPool(config) as db:
        print("health_check =", db.health_check())

        version = db.fetch_one(
            "SELECT version() AS version"
        )
        print(version)

        print("pool_stats =", db.pool_stats())
