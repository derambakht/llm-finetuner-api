"""
Logging utilities for the application.
"""
import logging
import os
import sys
from datetime import datetime
from typing import Optional
from pathlib import Path

from core.config import settings


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Set up application-wide logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(settings.LOGS_DIR, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("llm-finetuner")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler for main application log
    log_file = os.path.join(settings.LOGS_DIR, "app.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(funcName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "llm-finetuner") -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class JobLogger:
    """Logger for individual fine-tuning jobs."""
    
    def __init__(self, job_id: str):
        """
        Initialize job logger.
        
        Args:
            job_id: Unique job identifier
        """
        self.job_id = job_id
        self.log_dir = os.path.join(settings.JOBS_DIR, job_id)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.log_file = os.path.join(self.log_dir, "training.log")
        self.logger = logging.getLogger(f"job-{job_id}")
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # File handler for job-specific log
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def get_recent_logs(self, lines: int = 50) -> list[str]:
        """
        Get recent log entries.
        
        Args:
            lines: Number of recent lines to return
            
        Returns:
            List of recent log lines
        """
        if not os.path.exists(self.log_file):
            return []
        
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
                return [line.strip() for line in all_lines[-lines:]]
        except Exception:
            return []
    
    def get_log_path(self) -> str:
        """Get path to log file."""
        return self.log_file


# Global application logger
app_logger = setup_logging()
