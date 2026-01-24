"""Configuration and utility functions."""

import base64
import hashlib
import logging
import os
import pathlib
import platform
import time


# ROOT_DIR is used for default model download paths
ROOT_DIR = pathlib.Path(__file__).parent.parent


def setup_logging(log_file="ami_processing.log"):
    """Configure logging for both main process and worker processes"""
    # File handler with DEBUG level
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s'))
    # Stream handler with INFO level
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s'))
    # Root logger configuration
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set to lowest level you want to capture
    # Remove any existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def get_uuid():
    """Generate a unique identifier."""
    uuid_str = platform.node() + str(os.getpid()) + f"{time.time()*1000:0.0f}"
    return base64.b32encode(hashlib.md5(uuid_str.encode()).digest()).decode()[:-6]
