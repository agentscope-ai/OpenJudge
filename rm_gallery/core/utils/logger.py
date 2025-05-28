import os
import sys

from loguru import logger


def init_logger():
    logger.remove()
    admin_log_path = "./log"
    os.makedirs(admin_log_path, exist_ok=True)
    sensitive_log_path = os.path.join(admin_log_path, "rm_gallery.log")

    # TODO add logger prefix
    logger.add(sys.stdout, colorize=True, enqueue=True, level="INFO")
    logger.add(sensitive_log_path, rotation="10 MB", enqueue=True, level="INFO")
    logger.info("start!")
