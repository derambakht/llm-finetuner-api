"""
LLM Fine-Tuner API
==================

A FastAPI-based REST API service for fine-tuning Large Language Models.

Supports:
- Multiple models from Hugging Face Hub (Llama, Mistral, Phi, Gemma, etc.)
- LoRA, QLoRA, and Full fine-tuning methods
- Various dataset formats (JSON, JSONL, CSV, Parquet, HF datasets)
- Async job processing with progress tracking
- Model testing and download

Author: AI Course
Version: 1.0.0
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from core.config import settings
from core.dependencies import init_database
from api.v1 import api_router, health_router
from utils.logging import setup_logging, get_logger


# Initialize logging
setup_logging("DEBUG" if settings.DEBUG else "INFO")
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler for startup and shutdown events.
    """
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    
    # Initialize database
    logger.info("Initializing database...")
    init_database()
    
    # Log device info
    from utils.helpers import get_device_info
    device_info = get_device_info()
    logger.info(f"Device: {device_info['device']}")
    if device_info['cuda_available']:
        for gpu in device_info['cuda_devices']:
            logger.info(f"  GPU {gpu['index']}: {gpu['name']} ({gpu['total_memory_gb']} GB)")
    
    logger.info("Application started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    logger.info("Application stopped")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="""
## LLM Fine-Tuner API

A comprehensive REST API service for fine-tuning Large Language Models.

### Features

- **Multiple Model Support**: Fine-tune models from Hugging Face Hub including Llama 3, Mistral, Phi-3, Gemma, and more.
- **Efficient Fine-tuning Methods**: Support for LoRA, QLoRA (4-bit), and Full fine-tuning.
- **Flexible Dataset Handling**: Accept JSON, JSONL, CSV, Parquet files or load directly from Hugging Face datasets.
- **Async Job Processing**: Background training with real-time progress tracking.
- **Model Testing**: Test your fine-tuned models with custom prompts.
- **Easy Deployment**: Download trained models or push directly to Hugging Face Hub.

### Authentication

All endpoints (except health checks) require an API key passed in the `X-API-Key` header.

### Quick Start

1. Create a fine-tuning job with `POST /api/v1/jobs/create`
2. Monitor progress with `GET /api/v1/jobs/{job_id}`
3. Test the model with `POST /api/v1/jobs/{job_id}/test`
4. Download the model with `GET /api/v1/jobs/{job_id}/download`
    """,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed messages."""
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        errors.append(f"{field}: {error['msg']}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Validation error",
            "errors": errors,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected error occurred",
            "error_type": type(exc).__name__,
        },
    )


# Include routers
app.include_router(health_router)
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "api": settings.API_V1_PREFIX,
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info",
    )
