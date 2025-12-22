"""
Helper utilities and common functions.
"""
import os
import re
import shutil
import uuid
import zipfile
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
import json

import torch


def generate_job_id() -> str:
    """
    Generate a unique job ID.
    
    Returns:
        UUID string for job identification
    """
    return str(uuid.uuid4())


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": 0,
        "cuda_devices": [],
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "device": "cpu",
        "bf16_supported": False,
        "fp16_supported": False,
    }
    
    if info["cuda_available"]:
        info["cuda_device_count"] = torch.cuda.device_count()
        info["device"] = "cuda"
        
        for i in range(info["cuda_device_count"]):
            props = torch.cuda.get_device_properties(i)
            info["cuda_devices"].append({
                "index": i,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
                "major": props.major,
                "minor": props.minor,
            })
        
        # Check precision support
        if info["cuda_device_count"] > 0:
            major = torch.cuda.get_device_properties(0).major
            info["bf16_supported"] = major >= 8  # Ampere or newer
            info["fp16_supported"] = major >= 7  # Volta or newer
    elif info["mps_available"]:
        info["device"] = "mps"
        info["fp16_supported"] = True
    
    return info


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to be safe for filesystem.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace problematic characters
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
    filename = re.sub(r"\s+", "_", filename)
    filename = filename.strip("._")
    return filename[:255]  # Max filename length


def create_zip_archive(source_dir: str, output_path: str) -> str:
    """
    Create a ZIP archive from a directory.
    
    Args:
        source_dir: Source directory to archive
        output_path: Output ZIP file path
        
    Returns:
        Path to created ZIP file
    """
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)
    
    return output_path


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB
    """
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return 0.0


def estimate_model_memory_gb(model_name: str, method: str = "lora") -> float:
    """
    Estimate GPU memory required for a model.
    
    Args:
        model_name: Model name or path
        method: Fine-tuning method (lora, qlora, full)
        
    Returns:
        Estimated memory in GB
    """
    # Rough estimates based on model size patterns in name
    size_patterns = {
        "1b": 2,
        "3b": 6,
        "7b": 14,
        "8b": 16,
        "13b": 26,
        "30b": 60,
        "65b": 130,
        "70b": 140,
    }
    
    model_lower = model_name.lower()
    base_memory = 8  # Default for unknown models
    
    for pattern, memory in size_patterns.items():
        if pattern in model_lower:
            base_memory = memory
            break
    
    # Adjust based on method
    if method == "qlora":
        return base_memory * 0.25  # 4-bit quantization
    elif method == "lora":
        return base_memory * 0.5  # Half precision + LoRA
    else:  # full
        return base_memory * 2  # Full fine-tuning needs more memory


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def parse_model_name(model_name: str) -> Dict[str, str]:
    """
    Parse model name into components.
    
    Args:
        model_name: Full model name (e.g., "meta-llama/Meta-Llama-3-8B")
        
    Returns:
        Dictionary with parsed components
    """
    parts = model_name.split("/")
    if len(parts) == 2:
        return {
            "organization": parts[0],
            "model": parts[1],
            "full_name": model_name
        }
    return {
        "organization": None,
        "model": model_name,
        "full_name": model_name
    }


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Output file path
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load data from JSON file.
    
    Args:
        file_path: Input file path
        
    Returns:
        Loaded data or None if file doesn't exist
    """
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def cleanup_directory(dir_path: str) -> bool:
    """
    Remove a directory and its contents.
    
    Args:
        dir_path: Directory path to remove
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        return True
    except Exception:
        return False


def get_default_lora_target_modules(model_name: str) -> List[str]:
    """
    Get default LoRA target modules based on model architecture.
    
    Args:
        model_name: Model name or path
        
    Returns:
        List of target module names
    """
    model_lower = model_name.lower()
    
    # Llama-based models
    if any(x in model_lower for x in ["llama", "mistral", "vicuna", "alpaca"]):
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Phi models
    if "phi" in model_lower:
        return ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
    
    # Gemma models
    if "gemma" in model_lower:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Bloom models
    if "bloom" in model_lower:
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    
    # Falcon models
    if "falcon" in model_lower:
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    
    # GPT-NeoX / Pythia
    if any(x in model_lower for x in ["neox", "pythia"]):
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    
    # Default for transformer models
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def validate_hf_token(token: str) -> bool:
    """
    Validate Hugging Face token format.
    
    Args:
        token: HF token to validate
        
    Returns:
        True if valid format, False otherwise
    """
    if not token:
        return False
    # HF tokens start with "hf_" and are alphanumeric
    return token.startswith("hf_") and len(token) > 10
