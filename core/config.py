"""
Application configuration settings using Pydantic Settings.
"""
import os
from typing import List, Optional
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, computed_field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    # Application settings
    APP_NAME: str = "LLM Fine-Tuner API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API settings
    API_V1_PREFIX: str = "/api/v1"
    
    # Security - stored as comma-separated string
    API_KEYS_STR: str = Field(default="default-dev-key", alias="API_KEYS")
    
    @computed_field
    @property
    def API_KEYS(self) -> List[str]:
        """Parse API_KEYS from comma-separated string."""
        return [key.strip() for key in self.API_KEYS_STR.split(',') if key.strip()]
    
    # Hugging Face
    HF_TOKEN: Optional[str] = None
    HF_CACHE_DIR: str = "./hf_cache"
    
    # Storage paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = Field(default="")
    UPLOADS_DIR: str = Field(default="")
    JOBS_DIR: str = Field(default="")
    MODELS_DIR: str = Field(default="")
    LOGS_DIR: str = Field(default="")
    
    # Database
    DATABASE_URL: str = "sqlite:///./data/finetuner.db"
    
    # Upload limits
    MAX_UPLOAD_SIZE_GB: float = 10.0
    MAX_UPLOAD_SIZE_BYTES: int = Field(default=0)
    
    # Fine-tuning defaults
    DEFAULT_LEARNING_RATE: float = 2e-4
    DEFAULT_BATCH_SIZE: int = 4
    DEFAULT_EPOCHS: int = 3
    DEFAULT_MAX_SEQ_LENGTH: int = 2048
    DEFAULT_LORA_RANK: int = 16
    DEFAULT_LORA_ALPHA: int = 32
    DEFAULT_LORA_DROPOUT: float = 0.05
    DEFAULT_WARMUP_STEPS: int = 100
    DEFAULT_WEIGHT_DECAY: float = 0.01
    DEFAULT_GRADIENT_ACCUMULATION_STEPS: int = 4
    
    # Supported models cache TTL (seconds)
    MODELS_CACHE_TTL: int = 3600
    
    def model_post_init(self, __context) -> None:
        """Initialize computed paths after model creation."""
        if not self.DATA_DIR:
            self.DATA_DIR = os.path.join(self.BASE_DIR, "data")
        if not self.UPLOADS_DIR:
            self.UPLOADS_DIR = os.path.join(self.DATA_DIR, "uploads")
        if not self.JOBS_DIR:
            self.JOBS_DIR = os.path.join(self.DATA_DIR, "jobs")
        if not self.MODELS_DIR:
            self.MODELS_DIR = os.path.join(self.BASE_DIR, "trained_models")
        if not self.LOGS_DIR:
            self.LOGS_DIR = os.path.join(self.DATA_DIR, "logs")
        if not self.MAX_UPLOAD_SIZE_BYTES:
            self.MAX_UPLOAD_SIZE_BYTES = int(self.MAX_UPLOAD_SIZE_GB * 1024 * 1024 * 1024)
        
        # Ensure directories exist
        for dir_path in [self.DATA_DIR, self.UPLOADS_DIR, self.JOBS_DIR, 
                         self.MODELS_DIR, self.LOGS_DIR]:
            os.makedirs(dir_path, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
