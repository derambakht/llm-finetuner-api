"""
Security utilities for API key validation.
"""
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from typing import Optional

from .config import settings


# API Key header scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def validate_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """
    Validate the API key from request header.
    
    Args:
        api_key: API key from X-API-Key header
        
    Returns:
        The validated API key
        
    Raises:
        HTTPException: If API key is missing or invalid
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key. Please provide X-API-Key header.",
            headers={"WWW-Authenticate": "API-Key"},
        )
    
    if api_key not in settings.API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key.",
        )
    
    return api_key


def verify_api_key(api_key: str) -> bool:
    """
    Verify if an API key is valid.
    
    Args:
        api_key: API key to verify
        
    Returns:
        True if valid, False otherwise
    """
    return api_key in settings.API_KEYS
