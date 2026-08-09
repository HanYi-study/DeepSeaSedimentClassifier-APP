"""
应用程序日志模块
提供统一的日志记录接口，同时输出到文件和控制台。
"""

import logging
import os
import sys
from datetime import datetime


class AppLogger:
    """应用程序日志管理器"""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, log_dir: str = None, level: int = logging.INFO):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True

        self.logger = logging.getLogger("DeepSeaSedimentClassifier")
        self.logger.setLevel(level)
        self.logger.handlers.clear()

        # 格式化
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # 控制台输出
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(fmt)
        console_handler.setLevel(level)
        self.logger.addHandler(console_handler)

        # 文件输出
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(
                log_dir,
                f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            )
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(fmt)
            file_handler.setLevel(level)
            self.logger.addHandler(file_handler)
            self.logger.info(f"日志文件: {log_file}")

    def debug(self, msg: str):
        self.logger.debug(msg)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def exception(self, msg: str):
        self.logger.exception(msg)


# 全局单例
logger = AppLogger(os.path.join(os.path.expanduser("~"), ".deepsea_logs"))
