"""
API endpoints for fine-tuning operations.
"""
import os
from datetime import datetime
from typing import Optional, List
import asyncio

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    UploadFile,
    File,
    Form,
    Query,
    BackgroundTasks,
    status,
)
from fastapi.responses import FileResponse, StreamingResponse

from core.config import settings
from core.security import validate_api_key
from models.job import (
    JobStatus,
    FineTuneMethod,
    Hyperparameters,
    JobCreateRequest,
    JobCreateResponse,
    JobInfo,
    JobSummary,
    JobListResponse,
    TestModelRequest,
    TestModelResponse,
    SupportedModel,
    SupportedModelsResponse,
    CancelJobResponse,
    ErrorResponse,
)
from services.storage_service import storage_service
from services.dataset_service import dataset_service
from services.finetune_service import finetune_service, FineTuneConfig
from utils.helpers import generate_job_id, get_device_info
from utils.logging import get_logger, JobLogger

logger = get_logger(__name__)

router = APIRouter(prefix="/jobs", tags=["Fine-Tuning Jobs"])


@router.post(
    "/create",
    response_model=JobCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new fine-tuning job",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        413: {"model": ErrorResponse, "description": "File too large"},
    },
)
async def create_job(
    background_tasks: BackgroundTasks,
    model_name: str = Form(..., description="Hugging Face model name"),
    dataset_source: str = Form(..., description="Dataset source: HF name or 'upload'"),
    method: FineTuneMethod = Form(default=FineTuneMethod.LORA, description="Fine-tuning method"),
    dataset_file: Optional[UploadFile] = File(default=None, description="Dataset file if source is 'upload'"),
    hyperparameters_json: Optional[str] = Form(default=None, description="JSON string of hyperparameters"),
    hf_token: Optional[str] = Form(default=None, description="Hugging Face token"),
    push_to_hub: bool = Form(default=False, description="Push to Hugging Face Hub"),
    hub_model_id: Optional[str] = Form(default=None, description="Hub model ID"),
    description: Optional[str] = Form(default=None, description="Job description"),
    api_key: str = Depends(validate_api_key),
) -> JobCreateResponse:
    """
    Create a new fine-tuning job.
    
    This endpoint accepts a model name, dataset source, and optional hyperparameters
    to create a new fine-tuning job. The job will be queued and processed in the background.
    
    For dataset upload, use `dataset_source="upload"` and provide `dataset_file`.
    For HuggingFace datasets, provide the dataset name as `dataset_source`.
    """
    # Generate job ID
    job_id = generate_job_id()
    
    # Parse hyperparameters
    hyperparams = Hyperparameters()
    if hyperparameters_json:
        import json
        try:
            hp_dict = json.loads(hyperparameters_json)
            hyperparams = Hyperparameters(**hp_dict)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid hyperparameters JSON: {str(e)}"
            )
    
    # Handle dataset upload
    dataset_path = None
    if dataset_source.lower() == "upload":
        if not dataset_file:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Dataset file is required when source is 'upload'"
            )
        
        # Validate file extension
        if not dataset_service.validate_file_extension(dataset_file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format. Supported: {dataset_service.SUPPORTED_EXTENSIONS}"
            )
        
        # Check file size
        content = await dataset_file.read()
        if len(content) > settings.MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE_GB}GB"
            )
        
        # Save file
        dataset_path = dataset_service.save_uploaded_file(
            job_id, content, dataset_file.filename
        )
    
    # Validate push_to_hub requirements
    if push_to_hub and not hub_model_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="hub_model_id is required when push_to_hub is True"
        )
    
    # Create job in database
    storage_service.create_job(
        job_id=job_id,
        model_name=model_name,
        dataset_source=dataset_source,
        method=method.value,
        hyperparameters=hyperparams.model_dump(),
        dataset_path=dataset_path,
        hf_token=hf_token,
        description=description,
    )
    
    # Create fine-tune config
    config = FineTuneConfig(
        job_id=job_id,
        model_name=model_name,
        dataset_source=dataset_source,
        dataset_path=dataset_path,
        method=method,
        hyperparameters=hyperparams,
        hf_token=hf_token,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
    )
    
    # Start training in background
    finetune_service.start_training(config)
    
    logger.info(f"Created job {job_id} for model {model_name}")
    
    return JobCreateResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        message="Job created and queued for processing"
    )


@router.post(
    "/create-json",
    response_model=JobCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new fine-tuning job (JSON body)",
)
async def create_job_json(
    request: JobCreateRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(validate_api_key),
) -> JobCreateResponse:
    """
    Create a new fine-tuning job using JSON body.
    
    Note: This endpoint does not support file upload. Use /create for file uploads
    or provide a Hugging Face dataset name as dataset_source.
    """
    if request.dataset_source.lower() == "upload":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File upload not supported in JSON endpoint. Use /create with form data."
        )
    
    job_id = generate_job_id()
    hyperparams = request.hyperparameters or Hyperparameters()
    
    # Validate push_to_hub
    if request.push_to_hub and not request.hub_model_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="hub_model_id is required when push_to_hub is True"
        )
    
    # Create job
    storage_service.create_job(
        job_id=job_id,
        model_name=request.model_name,
        dataset_source=request.dataset_source,
        method=request.method.value,
        hyperparameters=hyperparams.model_dump(),
        hf_token=request.hf_token,
        description=request.description,
    )
    
    # Create config and start training
    config = FineTuneConfig(
        job_id=job_id,
        model_name=request.model_name,
        dataset_source=request.dataset_source,
        dataset_path=None,
        method=request.method,
        hyperparameters=hyperparams,
        hf_token=request.hf_token,
        push_to_hub=request.push_to_hub,
        hub_model_id=request.hub_model_id,
    )
    
    finetune_service.start_training(config)
    
    return JobCreateResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        message="Job created and queued for processing"
    )


@router.get(
    "/{job_id}",
    response_model=JobInfo,
    summary="Get job details",
    responses={404: {"model": ErrorResponse, "description": "Job not found"}},
)
async def get_job(
    job_id: str,
    api_key: str = Depends(validate_api_key),
) -> JobInfo:
    """
    Get detailed information about a specific fine-tuning job.
    
    Returns status, progress, logs, timestamps, and error messages if any.
    """
    job = storage_service.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    # Get recent logs
    logs = storage_service.get_job_logs(job_id, limit=50)
    log_messages = [f"[{l['level']}] {l['message']}" for l in logs]
    
    return JobInfo(
        id=job["id"],
        model_name=job["model_name"],
        dataset_source=job["dataset_source"],
        method=FineTuneMethod(job["method"]),
        hyperparameters=job.get("hyperparameters"),
        description=job.get("description"),
        status=JobStatus(job["status"]),
        progress=job.get("progress", 0.0),
        error_message=job.get("error_message"),
        output_path=job.get("output_path"),
        logs=log_messages,
        created_at=job["created_at"],
        started_at=job.get("started_at"),
        finished_at=job.get("finished_at"),
    )


@router.get(
    "",
    response_model=JobListResponse,
    summary="List all jobs",
)
async def list_jobs(
    limit: int = Query(default=20, ge=1, le=100, description="Maximum number of jobs to return"),
    offset: int = Query(default=0, ge=0, description="Number of jobs to skip"),
    status_filter: Optional[str] = Query(default=None, alias="status", description="Filter by status"),
    api_key: str = Depends(validate_api_key),
) -> JobListResponse:
    """
    List all fine-tuning jobs with pagination.
    
    Supports filtering by status: queued, running, completed, failed, cancelled.
    """
    jobs, total = storage_service.get_jobs(limit=limit, offset=offset, status=status_filter)
    
    summaries = [
        JobSummary(
            id=job["id"],
            model_name=job["model_name"],
            method=FineTuneMethod(job["method"]),
            status=JobStatus(job["status"]),
            progress=job.get("progress", 0.0),
            created_at=job["created_at"],
            description=job.get("description"),
        )
        for job in jobs
    ]
    
    return JobListResponse(
        jobs=summaries,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post(
    "/{job_id}/cancel",
    response_model=CancelJobResponse,
    summary="Cancel a running job",
    responses={
        404: {"model": ErrorResponse, "description": "Job not found"},
        400: {"model": ErrorResponse, "description": "Job cannot be cancelled"},
    },
)
async def cancel_job(
    job_id: str,
    api_key: str = Depends(validate_api_key),
) -> CancelJobResponse:
    """
    Cancel a running or queued fine-tuning job.
    
    Only jobs in 'queued' or 'running' status can be cancelled.
    """
    job = storage_service.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    current_status = JobStatus(job["status"])
    if current_status not in [JobStatus.QUEUED, JobStatus.RUNNING]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job cannot be cancelled. Current status: {current_status.value}"
        )
    
    # Request cancellation
    if current_status == JobStatus.RUNNING:
        cancelled = finetune_service.cancel_job(job_id)
        if not cancelled:
            # Job might have finished or not found in running jobs
            storage_service.update_job_status(job_id, JobStatus.CANCELLED)
    else:
        # Queued job - just update status
        storage_service.update_job_status(job_id, JobStatus.CANCELLED)
    
    logger.info(f"Cancelled job {job_id}")
    
    return CancelJobResponse(
        job_id=job_id,
        status=JobStatus.CANCELLED,
        message="Job cancellation requested"
    )


@router.get(
    "/{job_id}/download",
    summary="Download trained model",
    responses={
        404: {"model": ErrorResponse, "description": "Job or model not found"},
        400: {"model": ErrorResponse, "description": "Job not completed"},
    },
)
async def download_model(
    job_id: str,
    merge: bool = Query(default=False, description="Merge LoRA adapter with base model"),
    api_key: str = Depends(validate_api_key),
):
    """
    Download the fine-tuned model as a ZIP archive.
    
    For LoRA/QLoRA models, use `merge=true` to merge the adapter with the base model
    before downloading. This creates a standalone model that doesn't require PEFT.
    """
    job = storage_service.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    if JobStatus(job["status"]) != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model download only available for completed jobs"
        )
    
    if not storage_service.model_exists(job_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model files not found"
        )
    
    # Handle merge if requested
    if merge and job["method"] in ["lora", "qlora"]:
        try:
            merged_path = finetune_service.merge_adapter(job_id)
            # Create archive from merged path
            zip_path = os.path.join(settings.MODELS_DIR, f"{job_id}_merged.zip")
            from utils.helpers import create_zip_archive
            create_zip_archive(merged_path, zip_path)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to merge model: {str(e)}"
            )
    else:
        # Create archive from adapter/model
        zip_path = storage_service.create_model_archive(job_id)
        if not zip_path:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create model archive"
            )
    
    filename = f"{job_id}_merged.zip" if merge else f"{job_id}.zip"
    
    return FileResponse(
        path=zip_path,
        filename=filename,
        media_type="application/zip",
    )


@router.post(
    "/{job_id}/test",
    response_model=TestModelResponse,
    summary="Test trained model",
    responses={
        404: {"model": ErrorResponse, "description": "Job or model not found"},
        400: {"model": ErrorResponse, "description": "Job not completed"},
    },
)
async def test_model(
    job_id: str,
    request: TestModelRequest,
    api_key: str = Depends(validate_api_key),
) -> TestModelResponse:
    """
    Test a fine-tuned model with a prompt.
    
    Loads the model and generates a response based on the provided prompt.
    """
    job = storage_service.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    if JobStatus(job["status"]) != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model testing only available for completed jobs"
        )
    
    try:
        result = await asyncio.to_thread(
            finetune_service.test_model,
            job_id=job_id,
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            do_sample=request.do_sample,
        )
        
        return TestModelResponse(**result)
        
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to test model: {str(e)}"
        )


@router.delete(
    "/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a job",
    responses={
        404: {"model": ErrorResponse, "description": "Job not found"},
        400: {"model": ErrorResponse, "description": "Cannot delete running job"},
    },
)
async def delete_job(
    job_id: str,
    api_key: str = Depends(validate_api_key),
):
    """
    Delete a job and all associated files.
    
    Running jobs must be cancelled before deletion.
    """
    job = storage_service.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    if JobStatus(job["status"]) == JobStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete a running job. Cancel it first."
        )
    
    # Clean up dataset files
    dataset_service.cleanup_job_files(job_id)
    
    # Delete job
    storage_service.delete_job(job_id)
    
    logger.info(f"Deleted job {job_id}")
    return None
