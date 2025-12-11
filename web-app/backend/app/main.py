"""FastAPI application entry point."""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
import uuid
import time
from contextlib import asynccontextmanager

from .database import init_db
from .routers import auth, evaluations, preferences, ollama, chat
from .websocket import socket_app, sio
from .cache import check_redis_health
from .security import SecurityHeadersMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Starting LLM Judge Auditor API...")
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
    
    # Check Redis connection
    if check_redis_health():
        logger.info("Redis cache is available and healthy")
    else:
        logger.warning("Redis cache is not available - caching will be disabled")
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLM Judge Auditor API...")


# Create FastAPI application
app = FastAPI(
    title="LLM Judge Auditor API",
    description="API for evaluating LLM outputs with comprehensive metrics",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


# Request ID Middleware
@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """Add unique request ID to each request for tracing."""
    from .monitoring import performance_monitor, sentry
    
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Log request
    start_time = time.time()
    logger.info(f"Request started: {request.method} {request.url.path} [{request_id}]")
    
    try:
        response = await call_next(request)
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        # Log response
        process_time = time.time() - start_time
        logger.info(
            f"Request completed: {request.method} {request.url.path} "
            f"- Status: {response.status_code} - Time: {process_time:.3f}s [{request_id}]"
        )
        
        # Record metrics
        is_error = response.status_code >= 400
        performance_monitor.record_request(process_time, is_error)
        
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"Request failed: {request.method} {request.url.path} "
            f"- Error: {str(e)} - Time: {process_time:.3f}s [{request_id}]",
            exc_info=True
        )
        
        # Record error metrics
        performance_monitor.record_request(process_time, is_error=True)
        
        # Capture to Sentry
        sentry.capture_exception(e, {
            "request": {
                "method": request.method,
                "path": request.url.path,
                "request_id": request_id
            }
        })
        
        raise


# Add security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# Add response compression middleware
app.add_middleware(
    GZipMiddleware,
    minimum_size=1000,  # Only compress responses larger than 1KB
    compresslevel=6  # Balance between compression ratio and speed
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:80",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:80",
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"]
)


# Global exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions with structured error responses."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail} [{request_id}]")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "request_id": request_id
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors with detailed error messages."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.warning(f"Validation error: {exc.errors()} [{request_id}]")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error",
            "details": exc.errors(),
            "request_id": request_id
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions with generic error response."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(f"Unhandled exception: {str(exc)} [{request_id}]", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "request_id": request_id
        }
    )


# Root endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "LLM Judge Auditor API",
        "version": "0.1.0",
        "docs": "/docs",
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Basic health check endpoint for load balancers."""
    return {
        "status": "healthy",
        "service": "llm-judge-auditor-backend",
        "version": "0.1.0",
        "timestamp": time.time()
    }


@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with component status."""
    from .monitoring import HealthChecker
    
    checker = HealthChecker()
    
    # Check all components
    database_health = checker.check_database()
    cache_health = checker.check_cache()
    app_info = checker.get_application_info()
    
    # Determine overall status
    overall_status = "healthy"
    if database_health["status"] != "healthy":
        overall_status = "degraded"
    
    return {
        "status": overall_status,
        "timestamp": time.time(),
        "application": app_info,
        "components": {
            "database": database_health,
            "cache": cache_health
        }
    }


@app.get("/metrics")
async def metrics():
    """Application metrics endpoint."""
    from .monitoring import HealthChecker, performance_monitor
    
    checker = HealthChecker()
    
    return {
        "timestamp": time.time(),
        "system": checker.get_system_metrics(),
        "performance": performance_monitor.get_metrics()
    }


# Include routers
app.include_router(auth.router)
app.include_router(evaluations.router)
app.include_router(preferences.router)
app.include_router(ollama.router)
app.include_router(chat.router)

# Mount Socket.IO app
app.mount("/ws", socket_app)
