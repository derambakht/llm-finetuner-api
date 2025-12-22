"""
Dataset processing and validation service.
"""
import os
import json
import csv
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from datasets import Dataset, load_dataset, DatasetDict
import pandas as pd

from core.config import settings
from utils.logging import get_logger
from models.job import DatasetFormat

logger = get_logger(__name__)


class DatasetService:
    """Service for handling dataset operations."""
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {".json", ".jsonl", ".csv", ".parquet"}
    
    # Expected columns for instruction tuning
    INSTRUCTION_COLUMNS = ["instruction", "input", "output"]
    TEXT_COLUMN = "text"
    
    def __init__(self):
        """Initialize dataset service."""
        self.uploads_dir = settings.UPLOADS_DIR
        os.makedirs(self.uploads_dir, exist_ok=True)
    
    def save_uploaded_file(self, job_id: str, file_content: bytes, filename: str) -> str:
        """
        Save uploaded file to disk.
        
        Args:
            job_id: Job identifier
            file_content: File content as bytes
            filename: Original filename
            
        Returns:
            Path to saved file
        """
        # Create job-specific directory
        job_dir = os.path.join(self.uploads_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        # Save file
        file_path = os.path.join(job_dir, filename)
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        logger.info(f"Saved uploaded file: {file_path}")
        return file_path
    
    def detect_format(self, file_path: str) -> DatasetFormat:
        """
        Detect dataset format from file extension.
        
        Args:
            file_path: Path to dataset file
            
        Returns:
            Detected format
        """
        ext = Path(file_path).suffix.lower()
        format_map = {
            ".json": DatasetFormat.JSON,
            ".jsonl": DatasetFormat.JSONL,
            ".csv": DatasetFormat.CSV,
            ".parquet": DatasetFormat.PARQUET,
        }
        return format_map.get(ext, DatasetFormat.JSON)
    
    def validate_file_extension(self, filename: str) -> bool:
        """
        Check if file has a supported extension.
        
        Args:
            filename: Filename to check
            
        Returns:
            True if supported, False otherwise
        """
        ext = Path(filename).suffix.lower()
        return ext in self.SUPPORTED_EXTENSIONS
    
    def load_dataset_from_file(self, file_path: str) -> Dataset:
        """
        Load dataset from a local file.
        
        Args:
            file_path: Path to dataset file
            
        Returns:
            Loaded Dataset object
        """
        format_type = self.detect_format(file_path)
        
        try:
            if format_type == DatasetFormat.JSON:
                # Try loading as JSON array first
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return Dataset.from_list(data)
                elif isinstance(data, dict):
                    return Dataset.from_dict(data)
            
            elif format_type == DatasetFormat.JSONL:
                data = []
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data.append(json.loads(line))
                return Dataset.from_list(data)
            
            elif format_type == DatasetFormat.CSV:
                df = pd.read_csv(file_path)
                return Dataset.from_pandas(df)
            
            elif format_type == DatasetFormat.PARQUET:
                df = pd.read_parquet(file_path)
                return Dataset.from_pandas(df)
            
        except Exception as e:
            logger.error(f"Error loading dataset from {file_path}: {e}")
            raise ValueError(f"Failed to load dataset: {str(e)}")
        
        raise ValueError(f"Unsupported format: {format_type}")
    
    def load_dataset_from_hub(
        self, 
        dataset_name: str, 
        split: str = "train",
        token: Optional[str] = None
    ) -> Dataset:
        """
        Load dataset from Hugging Face Hub.
        
        Args:
            dataset_name: Name of dataset on HF Hub
            split: Dataset split to use
            token: HF token for private datasets
            
        Returns:
            Loaded Dataset object
        """
        try:
            dataset = load_dataset(dataset_name, split=split, token=token)
            if isinstance(dataset, DatasetDict):
                dataset = dataset[split] if split in dataset else dataset["train"]
            logger.info(f"Loaded dataset {dataset_name} with {len(dataset)} examples")
            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset from Hub: {e}")
            raise ValueError(f"Failed to load dataset from Hub: {str(e)}")
    
    def validate_dataset_structure(self, dataset: Dataset) -> Tuple[bool, str, str]:
        """
        Validate dataset structure and detect format type.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            Tuple of (is_valid, format_type, error_message)
        """
        columns = dataset.column_names
        
        # Check for instruction format
        if all(col in columns for col in ["instruction", "output"]):
            return True, "instruction", ""
        
        # Check for text format
        if self.TEXT_COLUMN in columns:
            return True, "text", ""
        
        # Check for conversation format (ShareGPT style)
        if "conversations" in columns:
            return True, "conversation", ""
        
        # Check for messages format
        if "messages" in columns:
            return True, "messages", ""
        
        return False, "", f"Dataset must have either 'instruction'/'output' columns, 'text' column, 'conversations', or 'messages'. Found: {columns}"
    
    def convert_to_training_format(
        self, 
        dataset: Dataset, 
        format_type: str,
        system_prompt: Optional[str] = None
    ) -> Dataset:
        """
        Convert dataset to a unified training format.
        
        Args:
            dataset: Input dataset
            format_type: Detected format type
            system_prompt: Optional system prompt to prepend
            
        Returns:
            Dataset with 'text' column in training format
        """
        def format_instruction_example(example: Dict[str, Any]) -> Dict[str, str]:
            """Format instruction-style example to text."""
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            output = example.get("output", "")
            
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            
            if system_prompt:
                prompt = f"### System:\n{system_prompt}\n\n{prompt}"
            
            return {"text": prompt}
        
        def format_conversation_example(example: Dict[str, Any]) -> Dict[str, str]:
            """Format conversation-style example to text."""
            conversations = example.get("conversations", [])
            text_parts = []
            
            if system_prompt:
                text_parts.append(f"### System:\n{system_prompt}")
            
            for turn in conversations:
                role = turn.get("from", turn.get("role", ""))
                content = turn.get("value", turn.get("content", ""))
                
                if role in ["human", "user"]:
                    text_parts.append(f"### Human:\n{content}")
                elif role in ["gpt", "assistant"]:
                    text_parts.append(f"### Assistant:\n{content}")
                elif role == "system":
                    text_parts.insert(0, f"### System:\n{content}")
            
            return {"text": "\n\n".join(text_parts)}
        
        def format_messages_example(example: Dict[str, Any]) -> Dict[str, str]:
            """Format messages-style example to text."""
            messages = example.get("messages", [])
            text_parts = []
            
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if role == "system":
                    text_parts.append(f"### System:\n{content}")
                elif role == "user":
                    text_parts.append(f"### Human:\n{content}")
                elif role == "assistant":
                    text_parts.append(f"### Assistant:\n{content}")
            
            return {"text": "\n\n".join(text_parts)}
        
        if format_type == "instruction":
            return dataset.map(format_instruction_example, remove_columns=dataset.column_names)
        elif format_type == "conversation":
            return dataset.map(format_conversation_example, remove_columns=dataset.column_names)
        elif format_type == "messages":
            return dataset.map(format_messages_example, remove_columns=dataset.column_names)
        elif format_type == "text":
            # Already in text format, just ensure column exists
            if "text" in dataset.column_names:
                return dataset.select_columns(["text"])
            raise ValueError("Text format expected but 'text' column not found")
        
        raise ValueError(f"Unknown format type: {format_type}")
    
    def prepare_dataset(
        self,
        source: str,
        file_path: Optional[str] = None,
        token: Optional[str] = None,
        system_prompt: Optional[str] = None,
        train_split: float = 0.9,
    ) -> Tuple[Dataset, Optional[Dataset]]:
        """
        Prepare dataset for training.
        
        Args:
            source: Dataset source (HF name or 'upload')
            file_path: Path to uploaded file (if source is 'upload')
            token: HF token for private datasets
            system_prompt: Optional system prompt
            train_split: Fraction for training (rest for validation)
            
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        # Load dataset
        if source.lower() == "upload":
            if not file_path:
                raise ValueError("File path required for upload source")
            dataset = self.load_dataset_from_file(file_path)
        else:
            dataset = self.load_dataset_from_hub(source, token=token)
        
        # Validate structure
        is_valid, format_type, error_msg = self.validate_dataset_structure(dataset)
        if not is_valid:
            raise ValueError(error_msg)
        
        logger.info(f"Dataset format detected: {format_type}")
        
        # Convert to training format
        dataset = self.convert_to_training_format(dataset, format_type, system_prompt)
        
        # Split into train/eval
        if train_split < 1.0 and len(dataset) > 10:
            split_dataset = dataset.train_test_split(test_size=1 - train_split, seed=42)
            return split_dataset["train"], split_dataset["test"]
        
        return dataset, None
    
    def get_dataset_info(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Get information about a dataset.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Dictionary with dataset information
        """
        info = {
            "num_examples": len(dataset),
            "columns": dataset.column_names,
            "features": {k: str(v) for k, v in dataset.features.items()},
        }
        
        # Get sample if available
        if len(dataset) > 0:
            sample = dataset[0]
            info["sample"] = {k: str(v)[:200] + "..." if len(str(v)) > 200 else str(v) 
                            for k, v in sample.items()}
        
        return info
    
    def cleanup_job_files(self, job_id: str) -> bool:
        """
        Clean up files for a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if successful
        """
        job_dir = os.path.join(self.uploads_dir, job_id)
        if os.path.exists(job_dir):
            import shutil
            shutil.rmtree(job_dir)
            logger.info(f"Cleaned up files for job {job_id}")
            return True
        return False


# Singleton instance
dataset_service = DatasetService()
