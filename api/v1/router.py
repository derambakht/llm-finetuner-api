"""
API router combining all endpoints.
"""
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, Depends, Query
from huggingface_hub import HfApi, list_models

from core.config import settings
from core.security import validate_api_key
from models.job import (
    SupportedModel,
    SupportedModelsResponse,
    FineTuneMethod,
)
from utils.helpers import get_device_info, estimate_model_memory_gb
from utils.logging import get_logger

from .endpoints import router as jobs_router

logger = get_logger(__name__)

# Main API router
api_router = APIRouter()

# Include jobs router
api_router.include_router(jobs_router)


# ==================== Models Endpoint ====================

models_router = APIRouter(prefix="/models", tags=["Supported Models"])


# Predefined list of recommended models
RECOMMENDED_MODELS = [
    SupportedModel(
        model_id="meta-llama/Meta-Llama-3-8B",
        name="Llama 3 8B",
        size="8B",
        architecture="LlamaForCausalLM",
        recommended_method=FineTuneMethod.LORA,
        min_gpu_memory_gb=16,
    ),
    SupportedModel(
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        name="Llama 3 8B Instruct",
        size="8B",
        architecture="LlamaForCausalLM",
        recommended_method=FineTuneMethod.LORA,
        min_gpu_memory_gb=16,
    ),
    SupportedModel(
        model_id="mistralai/Mistral-7B-v0.1",
        name="Mistral 7B",
        size="7B",
        architecture="MistralForCausalLM",
        recommended_method=FineTuneMethod.LORA,
        min_gpu_memory_gb=14,
    ),
    SupportedModel(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
        name="Mistral 7B Instruct v0.2",
        size="7B",
        architecture="MistralForCausalLM",
        recommended_method=FineTuneMethod.LORA,
        min_gpu_memory_gb=14,
    ),
    SupportedModel(
        model_id="microsoft/Phi-3-mini-4k-instruct",
        name="Phi-3 Mini 4K Instruct",
        size="3.8B",
        architecture="Phi3ForCausalLM",
        recommended_method=FineTuneMethod.LORA,
        min_gpu_memory_gb=8,
    ),
    SupportedModel(
        model_id="microsoft/Phi-3-medium-4k-instruct",
        name="Phi-3 Medium 4K Instruct",
        size="14B",
        architecture="Phi3ForCausalLM",
        recommended_method=FineTuneMethod.QLORA,
        min_gpu_memory_gb=28,
    ),
    SupportedModel(
        model_id="google/gemma-2b",
        name="Gemma 2B",
        size="2B",
        architecture="GemmaForCausalLM",
        recommended_method=FineTuneMethod.LORA,
        min_gpu_memory_gb=6,
    ),
    SupportedModel(
        model_id="google/gemma-7b",
        name="Gemma 7B",
        size="7B",
        architecture="GemmaForCausalLM",
        recommended_method=FineTuneMethod.LORA,
        min_gpu_memory_gb=14,
    ),
    SupportedModel(
        model_id="google/gemma-2-9b-it",
        name="Gemma 2 9B Instruct",
        size="9B",
        architecture="Gemma2ForCausalLM",
        recommended_method=FineTuneMethod.LORA,
        min_gpu_memory_gb=18,
    ),
    SupportedModel(
        model_id="Qwen/Qwen2-7B",
        name="Qwen2 7B",
        size="7B",
        architecture="Qwen2ForCausalLM",
        recommended_method=FineTuneMethod.LORA,
        min_gpu_memory_gb=14,
    ),
    SupportedModel(
        model_id="Qwen/Qwen2-7B-Instruct",
        name="Qwen2 7B Instruct",
        size="7B",
        architecture="Qwen2ForCausalLM",
        recommended_method=FineTuneMethod.LORA,
        min_gpu_memory_gb=14,
    ),
    SupportedModel(
        model_id="bigscience/bloom-7b1",
        name="BLOOM 7B1",
        size="7.1B",
        architecture="BloomForCausalLM",
        recommended_method=FineTuneMethod.LORA,
        min_gpu_memory_gb=14,
    ),
    SupportedModel(
        model_id="tiiuae/falcon-7b",
        name="Falcon 7B",
        size="7B",
        architecture="FalconForCausalLM",
        recommended_method=FineTuneMethod.LORA,
        min_gpu_memory_gb=14,
    ),
    SupportedModel(
        model_id="tiiuae/falcon-7b-instruct",
        name="Falcon 7B Instruct",
        size="7B",
        architecture="FalconForCausalLM",
        recommended_method=FineTuneMethod.LORA,
        min_gpu_memory_gb=14,
    ),
    SupportedModel(
        model_id="EleutherAI/pythia-6.9b",
        name="Pythia 6.9B",
        size="6.9B",
        architecture="GPTNeoXForCausalLM",
        recommended_method=FineTuneMethod.LORA,
        min_gpu_memory_gb=14,
    ),
]

# Cache for model search results
_models_cache = {
    "data": None,
    "timestamp": None,
}


@models_router.get(
    "/supported",
    response_model=SupportedModelsResponse,
    summary="Get list of supported models",
)
async def get_supported_models(
    search: Optional[str] = Query(default=None, description="Search query for model name"),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum results"),
    api_key: str = Depends(validate_api_key),
) -> SupportedModelsResponse:
    """
    Get a list of recommended models for fine-tuning.
    
    If a search query is provided, searches the Hugging Face Hub for matching models.
    Results are cached to improve performance.
    """
    if search:
        # Search HuggingFace Hub
        try:
            models = []
            api = HfApi()
            
            # Search for text-generation models
            search_results = api.list_models(
                search=search,
                task="text-generation",
                sort="downloads",
                direction=-1,
                limit=limit,
            )
            
            for model in search_results:
                # Estimate memory based on model name
                estimated_memory = estimate_model_memory_gb(model.id, "lora")
                
                # Recommend method based on size
                if estimated_memory > 20:
                    recommended = FineTuneMethod.QLORA
                elif estimated_memory > 10:
                    recommended = FineTuneMethod.LORA
                else:
                    recommended = FineTuneMethod.LORA
                
                models.append(SupportedModel(
                    model_id=model.id,
                    name=model.id.split("/")[-1],
                    size=None,  # Would need to parse from model card
                    architecture=None,
                    recommended_method=recommended,
                    min_gpu_memory_gb=estimated_memory,
                ))
            
            return SupportedModelsResponse(
                models=models,
                total=len(models),
                cached_at=datetime.utcnow(),
            )
            
        except Exception as e:
            logger.warning(f"Failed to search HuggingFace Hub: {e}")
            # Fall back to recommended models
            filtered = [m for m in RECOMMENDED_MODELS if search.lower() in m.model_id.lower()]
            return SupportedModelsResponse(
                models=filtered[:limit],
                total=len(filtered),
                cached_at=None,
            )
    
    # Return recommended models
    return SupportedModelsResponse(
        models=RECOMMENDED_MODELS[:limit],
        total=len(RECOMMENDED_MODELS),
        cached_at=None,
    )


@models_router.get(
    "/device-info",
    summary="Get device information",
)
async def get_device_info_endpoint(
    api_key: str = Depends(validate_api_key),
):
    """
    Get information about available compute devices.
    
    Returns CUDA availability, GPU memory, and supported precision modes.
    """
    return get_device_info()


# Include models router
api_router.include_router(models_router)


# ==================== Health Check (no auth required) ====================

health_router = APIRouter(tags=["Health"])


@health_router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "timestamp": datetime.utcnow().isoformat(),
    }


@health_router.get("/health/storage")
async def storage_health(api_key: str = Depends(validate_api_key)):
    """Get storage statistics."""
    from services.storage_service import storage_service
    return storage_service.get_storage_stats()
