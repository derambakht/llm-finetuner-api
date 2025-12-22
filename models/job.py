"""
Pydantic models and schemas for fine-tuning jobs.
"""
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator
import uuid


class JobStatus(str, Enum):
    """Enumeration of possible job statuses."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FineTuneMethod(str, Enum):
    """Enumeration of fine-tuning methods."""
    LORA = "lora"
    QLORA = "qlora"
    FULL = "full"


class DatasetFormat(str, Enum):
    """Supported dataset formats."""
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"
    HF_DATASET = "hf_dataset"


class Hyperparameters(BaseModel):
    """Fine-tuning hyperparameters configuration."""
    
    learning_rate: float = Field(default=2e-4, ge=1e-7, le=1.0, description="Learning rate for training")
    per_device_train_batch_size: int = Field(default=4, ge=1, le=256, description="Batch size per device")
    num_train_epochs: int = Field(default=3, ge=1, le=100, description="Number of training epochs")
    max_seq_length: int = Field(default=2048, ge=128, le=32768, description="Maximum sequence length")
    warmup_steps: int = Field(default=100, ge=0, description="Number of warmup steps")
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0, description="Weight decay coefficient")
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=128, description="Gradient accumulation steps")
    fp16: bool = Field(default=False, description="Use FP16 mixed precision")
    bf16: bool = Field(default=False, description="Use BF16 mixed precision")
    
    # LoRA specific parameters
    lora_rank: int = Field(default=16, ge=1, le=256, description="LoRA rank (r)")
    lora_alpha: int = Field(default=32, ge=1, le=512, description="LoRA alpha scaling factor")
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0, description="LoRA dropout probability")
    lora_target_modules: Optional[List[str]] = Field(
        default=None, 
        description="Target modules for LoRA. None means auto-detect."
    )
    
    # Additional settings
    save_steps: int = Field(default=100, ge=1, description="Save checkpoint every N steps")
    logging_steps: int = Field(default=10, ge=1, description="Log every N steps")
    eval_steps: Optional[int] = Field(default=None, description="Evaluate every N steps")
    max_steps: int = Field(default=-1, description="Maximum training steps. -1 means use epochs")
    
    class Config:
        json_schema_extra = {
            "example": {
                "learning_rate": 2e-4,
                "per_device_train_batch_size": 4,
                "num_train_epochs": 3,
                "max_seq_length": 2048,
                "lora_rank": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05
            }
        }


class JobCreateRequest(BaseModel):
    """Request model for creating a new fine-tuning job."""
    
    model_name: str = Field(
        ..., 
        min_length=1, 
        description="Hugging Face model name or path (e.g., 'meta-llama/Meta-Llama-3-8B')"
    )
    dataset_source: str = Field(
        ..., 
        description="Dataset source: HF dataset name or 'upload' for file upload"
    )
    method: FineTuneMethod = Field(
        default=FineTuneMethod.LORA,
        description="Fine-tuning method: lora, qlora, or full"
    )
    hyperparameters: Optional[Hyperparameters] = Field(
        default_factory=Hyperparameters,
        description="Training hyperparameters"
    )
    hf_token: Optional[str] = Field(
        default=None, 
        description="Hugging Face token for private models or pushing to Hub"
    )
    push_to_hub: bool = Field(
        default=False,
        description="Push trained model to Hugging Face Hub"
    )
    hub_model_id: Optional[str] = Field(
        default=None,
        description="Model ID for Hub (required if push_to_hub is True)"
    )
    description: Optional[str] = Field(
        default=None, 
        max_length=1000, 
        description="Optional description for this job"
    )
    
    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name format."""
        v = v.strip()
        if not v:
            raise ValueError("Model name cannot be empty")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "meta-llama/Meta-Llama-3-8B",
                "dataset_source": "upload",
                "method": "lora",
                "hyperparameters": {
                    "learning_rate": 2e-4,
                    "num_train_epochs": 3,
                    "lora_rank": 16
                },
                "description": "Fine-tuning Llama 3 on custom dataset"
            }
        }


class JobCreateResponse(BaseModel):
    """Response model for job creation."""
    
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(default=JobStatus.QUEUED, description="Current job status")
    message: str = Field(default="Job created successfully", description="Status message")


class JobInfo(BaseModel):
    """Complete job information model."""
    
    id: str = Field(..., description="Unique job identifier")
    model_name: str = Field(..., description="Model being fine-tuned")
    dataset_source: str = Field(..., description="Dataset source")
    method: FineTuneMethod = Field(..., description="Fine-tuning method")
    hyperparameters: Optional[Dict[str, Any]] = Field(default=None, description="Training hyperparameters")
    description: Optional[str] = Field(default=None, description="Job description")
    status: JobStatus = Field(..., description="Current job status")
    progress: float = Field(default=0.0, ge=0.0, le=100.0, description="Progress percentage")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    output_path: Optional[str] = Field(default=None, description="Path to output model")
    logs: Optional[List[str]] = Field(default=None, description="Recent log entries")
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(default=None, description="Training start timestamp")
    finished_at: Optional[datetime] = Field(default=None, description="Training completion timestamp")
    
    class Config:
        from_attributes = True


class JobSummary(BaseModel):
    """Summary model for job listing."""
    
    id: str = Field(..., description="Unique job identifier")
    model_name: str = Field(..., description="Model being fine-tuned")
    method: FineTuneMethod = Field(..., description="Fine-tuning method")
    status: JobStatus = Field(..., description="Current job status")
    progress: float = Field(default=0.0, description="Progress percentage")
    created_at: datetime = Field(..., description="Job creation timestamp")
    description: Optional[str] = Field(default=None, description="Job description")
    
    class Config:
        from_attributes = True


class JobListResponse(BaseModel):
    """Response model for job listing."""
    
    jobs: List[JobSummary] = Field(default_factory=list, description="List of jobs")
    total: int = Field(..., description="Total number of jobs")
    limit: int = Field(..., description="Limit used for pagination")
    offset: int = Field(..., description="Offset used for pagination")


class TestModelRequest(BaseModel):
    """Request model for testing a fine-tuned model."""
    
    prompt: str = Field(..., min_length=1, max_length=10000, description="Input prompt for generation")
    max_new_tokens: int = Field(default=256, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p (nucleus) sampling")
    top_k: int = Field(default=50, ge=1, le=1000, description="Top-k sampling")
    do_sample: bool = Field(default=True, description="Use sampling (vs greedy decoding)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "What is machine learning?",
                "max_new_tokens": 256,
                "temperature": 0.7
            }
        }


class TestModelResponse(BaseModel):
    """Response model for model testing."""
    
    output: str = Field(..., description="Generated text output")
    prompt: str = Field(..., description="Original prompt")
    tokens_generated: int = Field(..., description="Number of tokens generated")


class SupportedModel(BaseModel):
    """Model information for supported models list."""
    
    model_id: str = Field(..., description="Hugging Face model ID")
    name: str = Field(..., description="Display name")
    size: Optional[str] = Field(default=None, description="Model size (e.g., '7B', '13B')")
    architecture: Optional[str] = Field(default=None, description="Model architecture")
    recommended_method: FineTuneMethod = Field(
        default=FineTuneMethod.LORA, 
        description="Recommended fine-tuning method"
    )
    min_gpu_memory_gb: Optional[float] = Field(
        default=None, 
        description="Minimum GPU memory required (GB)"
    )


class SupportedModelsResponse(BaseModel):
    """Response model for supported models endpoint."""
    
    models: List[SupportedModel] = Field(..., description="List of supported models")
    total: int = Field(..., description="Total number of models")
    cached_at: Optional[datetime] = Field(default=None, description="Cache timestamp")


class CancelJobResponse(BaseModel):
    """Response model for job cancellation."""
    
    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="New job status")
    message: str = Field(..., description="Status message")


class DownloadRequest(BaseModel):
    """Query parameters for download endpoint."""
    
    merge: bool = Field(
        default=False, 
        description="Merge LoRA adapter with base model before download"
    )


class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(default=None, description="Error code")
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Job not found",
                "error_code": "JOB_NOT_FOUND"
            }
        }
