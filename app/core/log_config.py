import logging
import sys
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler


# 主函数中调用

def setup_logging(
    log_level: int = logging.INFO,
    log_dir: str = "./logs"
) -> None:
    """
    全局日志配置。

    应在项目启动时调用一次。
    """

    # ==========================
    # 1. 创建日志目录
    # ==========================
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # ==========================
    # 2. 日志格式
    # ==========================
    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s "
            "%(levelname)s "
            "%(name)s "
            "%(filename)s:%(lineno)d "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ==========================
    # 3. 获取 Root Logger
    # ==========================
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # ==========================
    # 4. 清理我们自己之前配置的 Handler
    # ==========================
    for handler in root_logger.handlers[:]:
        if getattr(handler, "_app_handler", False):
            root_logger.removeHandler(handler)
            handler.close()

    # ==========================
    # 5. Console Handler
    # ==========================
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level=logging.DEBUG)
    console_handler.setFormatter(formatter)

    console_handler._app_handler = True

    # ==========================
    # 6. File Handler
    # ==========================
    file_handler = TimedRotatingFileHandler(
        filename=log_path / "app.log",
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8"
    )

    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    file_handler._app_handler = True

    # ==========================
    # 7. 注册 Handler
    # ==========================
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)