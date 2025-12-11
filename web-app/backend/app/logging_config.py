"""Logging configuration for the application."""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Dict, Any


def get_log_level() -> int:
    """Get log level from environment variable."""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


def get_logging_config() -> Dict[str, Any]:
    """Get logging configuration dictionary."""
    log_level = get_log_level()
    environment = os.getenv("ENVIRONMENT", "development")
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(funcName)s %(lineno)d %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "default",
                "stream": sys.stdout
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": log_level,
                "formatter": "detailed",
                "filename": log_dir / "app.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf-8"
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": logging.ERROR,
                "formatter": "detailed",
                "filename": log_dir / "error.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf-8"
            }
        },
        "loggers": {
            "": {  # Root logger
                "level": log_level,
                "handlers": ["console", "file", "error_file"]
            },
            "uvicorn": {
                "level": log_level,
                "handlers": ["console", "file"],
                "propagate": False
            },
            "uvicorn.access": {
                "level": logging.WARNING if environment == "production" else logging.INFO,
                "handlers": ["console"],
                "propagate": False
            },
            "sqlalchemy.engine": {
                "level": logging.WARNING if environment == "production" else logging.INFO,
                "handlers": ["console", "file"],
                "propagate": False
            }
        }
    }
    
    # Use JSON formatter in production
    if environment == "production":
        try:
            import pythonjsonlogger  # noqa
            config["handlers"]["console"]["formatter"] = "json"
            config["handlers"]["file"]["formatter"] = "json"
        except ImportError:
            pass
    
    return config


def setup_logging():
    """Setup logging configuration."""
    config = get_logging_config()
    logging.config.dictConfig(config)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {logging.getLevelName(get_log_level())}")
