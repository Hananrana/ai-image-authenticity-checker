"""
Professional Logging Configuration
===================================

Provides structured logging with file rotation and console output.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional

# Import config - handle case where config might not be available yet
try:
    from config import LOGGING_CONFIG, LOGS_DIR
except ImportError:
    LOGGING_CONFIG = None
    LOGS_DIR = Path("outputs/logs")


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    log_to_console: bool = True,
    log_to_file: bool = True
) -> logging.Logger:
    """
    Set up the root logger with console and file handlers.
    
    Parameters
    ----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : Path, optional
        Path to log file. Defaults to config setting.
    log_to_console : bool
        Whether to log to console
    log_to_file : bool
        Whether to log to file
        
    Returns
    -------
    logging.Logger
        Configured root logger
    """
    # Get configuration
    if LOGGING_CONFIG:
        level = level or LOGGING_CONFIG.level
        log_file = log_file or LOGGING_CONFIG.log_file
        log_format = LOGGING_CONFIG.format
        date_format = LOGGING_CONFIG.date_format
        max_bytes = LOGGING_CONFIG.max_log_size_mb * 1024 * 1024
        backup_count = LOGGING_CONFIG.backup_count
    else:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        max_bytes = 10 * 1024 * 1024
        backup_count = 5
        log_file = log_file or LOGS_DIR / "app.log"
    
    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_to_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Parameters
    ----------
    name : str
        Logger name (typically __name__ of the calling module)
        
    Returns
    -------
    logging.Logger
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # Only set up if root logger isn't configured
    if not logging.getLogger().handlers:
        setup_logging()
    
    return logger


class LoggerMixin:
    """
    Mixin class to add logging capability to any class.
    
    Usage
    -----
    class MyClass(LoggerMixin):
        def my_method(self):
            self.logger.info("Doing something")
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger


# Performance logging decorator
def log_execution_time(func):
    """
    Decorator to log function execution time.
    
    Usage
    -----
    @log_execution_time
    def my_slow_function():
        ...
    """
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            logger.debug(f"{func.__name__} completed in {elapsed:.4f}s")
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(f"{func.__name__} failed after {elapsed:.4f}s: {e}")
            raise
    
    return wrapper
