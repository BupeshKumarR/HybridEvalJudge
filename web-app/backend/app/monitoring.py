"""Application monitoring and observability."""

import logging
import time
import psutil
import os
from typing import Dict, Any, Optional
from datetime import datetime
from sqlalchemy import text
from .database import SessionLocal
from .cache import check_redis_health

logger = logging.getLogger(__name__)


class HealthChecker:
    """Health check utilities for monitoring application status."""
    
    @staticmethod
    def check_database() -> Dict[str, Any]:
        """Check database connectivity and health."""
        try:
            db = SessionLocal()
            start_time = time.time()
            
            # Simple query to check connection
            db.execute(text("SELECT 1"))
            
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            db.close()
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time, 2)
            }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    @staticmethod
    def check_cache() -> Dict[str, Any]:
        """Check Redis cache health."""
        try:
            is_healthy = check_redis_health()
            return {
                "status": "healthy" if is_healthy else "unavailable",
                "enabled": is_healthy
            }
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    @staticmethod
    def get_system_metrics() -> Dict[str, Any]:
        """Get system resource metrics."""
        try:
            process = psutil.Process(os.getpid())
            
            return {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory": {
                    "total_mb": round(psutil.virtual_memory().total / (1024 * 1024), 2),
                    "available_mb": round(psutil.virtual_memory().available / (1024 * 1024), 2),
                    "percent": psutil.virtual_memory().percent,
                    "process_mb": round(process.memory_info().rss / (1024 * 1024), 2)
                },
                "disk": {
                    "total_gb": round(psutil.disk_usage('/').total / (1024 * 1024 * 1024), 2),
                    "used_gb": round(psutil.disk_usage('/').used / (1024 * 1024 * 1024), 2),
                    "percent": psutil.disk_usage('/').percent
                }
            }
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def get_application_info() -> Dict[str, Any]:
        """Get application information."""
        return {
            "name": "LLM Judge Auditor Backend",
            "version": "0.1.0",
            "environment": os.getenv("ENVIRONMENT", "development"),
            "python_version": os.sys.version.split()[0],
            "uptime_seconds": time.time() - psutil.Process(os.getpid()).create_time()
        }


class PerformanceMonitor:
    """Monitor application performance metrics."""
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        self.start_time = time.time()
    
    def record_request(self, response_time: float, is_error: bool = False):
        """Record a request for metrics."""
        self.request_count += 1
        self.total_response_time += response_time
        if is_error:
            self.error_count += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        uptime = time.time() - self.start_time
        
        return {
            "requests": {
                "total": self.request_count,
                "errors": self.error_count,
                "success_rate": round(
                    (self.request_count - self.error_count) / max(self.request_count, 1) * 100,
                    2
                )
            },
            "response_time": {
                "average_ms": round(
                    self.total_response_time / max(self.request_count, 1) * 1000,
                    2
                )
            },
            "uptime_seconds": round(uptime, 2)
        }


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


class SentryIntegration:
    """Sentry error tracking integration."""
    
    def __init__(self):
        self.enabled = False
        self.dsn = os.getenv("SENTRY_DSN")
        
        if self.dsn:
            try:
                import sentry_sdk
                from sentry_sdk.integrations.fastapi import FastApiIntegration
                from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
                
                sentry_sdk.init(
                    dsn=self.dsn,
                    environment=os.getenv("ENVIRONMENT", "development"),
                    traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
                    integrations=[
                        FastApiIntegration(),
                        SqlalchemyIntegration()
                    ],
                    send_default_pii=False,
                    attach_stacktrace=True
                )
                
                self.enabled = True
                logger.info("Sentry error tracking initialized")
            except ImportError:
                logger.warning("Sentry SDK not installed. Error tracking disabled.")
            except Exception as e:
                logger.error(f"Failed to initialize Sentry: {e}")
    
    def capture_exception(self, exception: Exception, context: Optional[Dict] = None):
        """Capture an exception to Sentry."""
        if self.enabled:
            try:
                import sentry_sdk
                
                if context:
                    with sentry_sdk.push_scope() as scope:
                        for key, value in context.items():
                            scope.set_context(key, value)
                        sentry_sdk.capture_exception(exception)
                else:
                    sentry_sdk.capture_exception(exception)
            except Exception as e:
                logger.error(f"Failed to capture exception to Sentry: {e}")


# Global Sentry integration instance
sentry = SentryIntegration()


def log_structured(
    level: str,
    message: str,
    **kwargs
):
    """Log structured data for better observability."""
    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "message": message,
        **kwargs
    }
    
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(str(log_data))
