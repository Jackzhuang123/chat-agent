#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
监控日志模块 - 记录系统运行状态、请求耗时、错误堆栈
支持按天轮转、多进程安全、控制台彩色输出
"""

import inspect
import logging
import logging.handlers
import sys
import time
import traceback
from functools import wraps
from pathlib import Path
from typing import Callable, Optional

# 日志存储目录（项目根目录下的 logs/）
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 日志文件命名格式：monitor_YYYY-MM-DD.log
LOG_FILENAME = LOG_DIR / "monitor.log"

# 日志格式（含时间、级别、模块、行号、消息）
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 全局单例日志记录器
_monitor_logger: Optional[logging.Logger] = None


def _setup_logger() -> logging.Logger:
    """初始化监控日志记录器（只执行一次）"""
    logger = logging.getLogger("monitor")
    logger.setLevel(logging.DEBUG)

    # 避免重复添加 Handler
    if logger.handlers:
        return logger

    # 1. 控制台 Handler（带颜色，便于开发调试）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(_ColoredFormatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(console_handler)

    # 2. 文件 Handler（按天轮转，保留30天）
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=str(LOG_FILENAME),
        when="midnight",          # 每天午夜轮转
        interval=1,
        backupCount=30,           # 保留最近30天的日志
        encoding="utf-8",
        delay=False,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    # 轮转后的文件命名后缀（默认是 .YYYY-MM-DD）
    file_handler.suffix = "%Y-%m-%d"
    logger.addHandler(file_handler)

    # 3. 错误专用 Handler（ERROR 及以上单独记录到 error.log，方便快速定位）
    error_handler = logging.handlers.TimedRotatingFileHandler(
        filename=str(LOG_DIR / "error.log"),
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(error_handler)

    return logger


class _ColoredFormatter(logging.Formatter):
    """控制台彩色日志格式器"""
    COLOR_CODES = {
        logging.DEBUG: "\033[36m",      # 青色
        logging.INFO: "\033[32m",       # 绿色
        logging.WARNING: "\033[33m",    # 黄色
        logging.ERROR: "\033[31m",      # 红色
        logging.CRITICAL: "\033[35m",   # 紫色
    }
    RESET_CODE = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        # 先获取原始格式字符串
        msg = super().format(record)
        # 添加颜色
        color = self.COLOR_CODES.get(record.levelno, "")
        return f"{color}{msg}{self.RESET_CODE}"


def get_monitor_logger() -> logging.Logger:
    """获取全局监控日志记录器（单例）"""
    global _monitor_logger
    if _monitor_logger is None:
        _monitor_logger = _setup_logger()
    return _monitor_logger


def log_execution_time(func: Optional[Callable] = None, *, level: int = logging.INFO):
    """
    装饰器：记录函数执行耗时和异常

    用法：
        @log_execution_time
        def my_func():
            pass

        @log_execution_time(level=logging.WARNING)
        def slow_func():
            pass
    """
    def decorator(f: Callable):
        @wraps(f)
        def wrapper(*args, **kwargs):
            logger = get_monitor_logger()
            start = time.perf_counter()
            try:
                result = f(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.log(
                    level,
                    f"{f.__module__}.{f.__qualname__} 执行完成，耗时 {elapsed:.3f}s"
                )
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.error(
                    f"{f.__module__}.{f.__qualname__} 执行失败，耗时 {elapsed:.3f}s\n"
                    f"异常类型: {type(e).__name__}\n"
                    f"异常信息: {str(e)}\n"
                    f"堆栈跟踪:\n{traceback.format_exc()}"
                )
                raise
        return wrapper

    if func is None:
        return decorator
    return decorator(func)


def log_async_execution_time(func: Optional[Callable] = None, *, level: int = logging.INFO):
    """
    装饰器：记录异步函数执行耗时和异常。
    同时支持：
      - 普通 async 函数（coroutine function）
      - 异步生成器函数（async def + yield）
    """
    def decorator(f: Callable):
        if inspect.isasyncgenfunction(f):
            # ---- 异步生成器函数：用 async for 透传每个 yield 值 ----
            @wraps(f)
            async def async_gen_wrapper(*args, **kwargs):
                logger = get_monitor_logger()
                start = time.perf_counter()
                try:
                    async for item in f(*args, **kwargs):
                        yield item
                    elapsed = time.perf_counter() - start
                    logger.log(
                        level,
                        f"{f.__module__}.{f.__qualname__} 执行完成，耗时 {elapsed:.3f}s"
                    )
                except Exception as e:
                    elapsed = time.perf_counter() - start
                    logger.error(
                        f"{f.__module__}.{f.__qualname__} 执行失败，耗时 {elapsed:.3f}s\n"
                        f"异常类型: {type(e).__name__}\n"
                        f"异常信息: {str(e)}\n"
                        f"堆栈跟踪:\n{traceback.format_exc()}"
                    )
                    raise
            return async_gen_wrapper
        else:
            # ---- 普通 async 函数：await 调用 ----
            @wraps(f)
            async def wrapper(*args, **kwargs):
                logger = get_monitor_logger()
                start = time.perf_counter()
                try:
                    result = await f(*args, **kwargs)
                    elapsed = time.perf_counter() - start
                    logger.log(
                        level,
                        f"{f.__module__}.{f.__qualname__} 执行完成，耗时 {elapsed:.3f}s"
                    )
                    return result
                except Exception as e:
                    elapsed = time.perf_counter() - start
                    logger.error(
                        f"{f.__module__}.{f.__qualname__} 执行失败，耗时 {elapsed:.3f}s\n"
                        f"异常类型: {type(e).__name__}\n"
                        f"异常信息: {str(e)}\n"
                        f"堆栈跟踪:\n{traceback.format_exc()}"
                    )
                    raise
            return wrapper

    if func is None:
        return decorator
    return decorator(func)


# ========== 便捷函数 ==========
def info(msg: str, *args, **kwargs):
    get_monitor_logger().info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs):
    get_monitor_logger().warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs):
    get_monitor_logger().error(msg, *args, **kwargs)


def debug(msg: str, *args, **kwargs):
    get_monitor_logger().debug(msg, *args, **kwargs)


def exception(msg: str, *args, **kwargs):
    """记录异常信息（自动附带堆栈）"""
    get_monitor_logger().exception(msg, *args, **kwargs)


# ========== 启动日志记录 ==========
def log_startup(app_name: str = "QwenAgent", port: Optional[int] = None):
    """应用启动时记录一条日志"""
    logger = get_monitor_logger()
    msg = f"{app_name} 启动"
    if port:
        msg += f"，监听端口 {port}"
    logger.info(msg)


def log_shutdown(app_name: str = "QwenAgent"):
    """应用关闭时记录一条日志"""
    get_monitor_logger().info(f"{app_name} 关闭")


# ========== 请求日志中间件（用于 Gradio 或自定义框架） ==========
def log_http_request(method: str, path: str, status: int, duration_ms: float):
    """记录 HTTP 请求摘要"""
    get_monitor_logger().info(f"{method} {path} → {status} ({duration_ms:.2f}ms)")