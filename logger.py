import logging
from typing import Optional

_LEVELS = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}


def get_logger(filename: str, verbosity: int = 0, name: Optional[str] = None) -> logging.Logger:
    level = _LEVELS.get(verbosity, logging.DEBUG)
    formatter = logging.Formatter("[%(levelname)s] %(message)s")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # 防止日志冒泡至根 logger 造成重复输出

    if logger.handlers:
        logger.handlers.clear()  # 多次调用时避免重复添加 handler

    file_handler = logging.FileHandler(filename, mode="w", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger