"""
Storage service for managing job metadata and model files.
"""
import os
import json
import shutil
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

from core.config import settings
from core.dependencies import get_db_context
from models.job import JobStatus, FineTuneMethod, JobInfo, JobSummary
from utils.logging import get_logger
from utils.helpers import save_json, load_json, create_zip_archive

logger = get_logger(__name__)


class StorageService:
    """Service for managing job storage and persistence."""
    
    def __init__(self):
        """Initialize storage service."""
        self.jobs_dir = settings.JOBS_DIR
        self.models_dir = settings.MODELS_DIR
        os.makedirs(self.jobs_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
    
    # ==================== Job CRUD Operations ====================
    
    def create_job(
        self,
        job_id: str,
        model_name: str,
        dataset_source: str,
        method: str,
        hyperparameters: Dict[str, Any],
        dataset_path: Optional[str] = None,
        hf_token: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new job record in the database.
        
        Args:
            job_id: Unique job identifier
            model_name: Model to fine-tune
            dataset_source: Dataset source
            method: Fine-tuning method
            hyperparameters: Training hyperparameters
            dataset_path: Path to dataset file
            hf_token: HF token (encrypted/hashed in production)
            description: Job description
            
        Returns:
            Created job record
        """
        with get_db_context() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO jobs (
                    id, model_name, dataset_source, dataset_path, method,
                    hyperparameters, hf_token, description, status, progress,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job_id,
                model_name,
                dataset_source,
                dataset_path,
                method,
                json.dumps(hyperparameters),
                hf_token,  # In production, encrypt this
                description,
                JobStatus.QUEUED.value,
                0.0,
                datetime.utcnow().isoformat()
            ))
            
            conn.commit()
        
        # Create job directory for logs and metadata
        job_dir = os.path.join(self.jobs_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        logger.info(f"Created job {job_id} in database")
        return self.get_job(job_id)
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job by ID.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job record or None
        """
        with get_db_context() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_dict(row)
            return None
    
    def get_jobs(
        self, 
        limit: int = 20, 
        offset: int = 0,
        status: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get paginated list of jobs.
        
        Args:
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip
            status: Filter by status
            
        Returns:
            Tuple of (jobs list, total count)
        """
        with get_db_context() as conn:
            cursor = conn.cursor()
            
            # Count total
            if status:
                cursor.execute("SELECT COUNT(*) FROM jobs WHERE status = ?", (status,))
            else:
                cursor.execute("SELECT COUNT(*) FROM jobs")
            total = cursor.fetchone()[0]
            
            # Get paginated results
            if status:
                cursor.execute("""
                    SELECT * FROM jobs WHERE status = ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """, (status, limit, offset))
            else:
                cursor.execute("""
                    SELECT * FROM jobs
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))
            
            rows = cursor.fetchall()
            jobs = [self._row_to_dict(row) for row in rows]
            
            return jobs, total
    
    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        progress: Optional[float] = None,
        error_message: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> bool:
        """
        Update job status and related fields.
        
        Args:
            job_id: Job identifier
            status: New status
            progress: Progress percentage
            error_message: Error message if failed
            output_path: Path to output model
            
        Returns:
            True if updated successfully
        """
        with get_db_context() as conn:
            cursor = conn.cursor()
            
            updates = ["status = ?"]
            values = [status.value]
            
            if progress is not None:
                updates.append("progress = ?")
                values.append(progress)
            
            if error_message is not None:
                updates.append("error_message = ?")
                values.append(error_message)
            
            if output_path is not None:
                updates.append("output_path = ?")
                values.append(output_path)
            
            # Update timestamps
            if status == JobStatus.RUNNING:
                updates.append("started_at = ?")
                values.append(datetime.utcnow().isoformat())
            elif status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                updates.append("finished_at = ?")
                values.append(datetime.utcnow().isoformat())
            
            values.append(job_id)
            
            cursor.execute(f"""
                UPDATE jobs SET {', '.join(updates)}
                WHERE id = ?
            """, values)
            
            conn.commit()
            
            logger.info(f"Updated job {job_id} status to {status.value}")
            return cursor.rowcount > 0
    
    def update_job_progress(self, job_id: str, progress: float) -> bool:
        """
        Update job progress percentage.
        
        Args:
            job_id: Job identifier
            progress: Progress percentage (0-100)
            
        Returns:
            True if updated successfully
        """
        with get_db_context() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE jobs SET progress = ? WHERE id = ?",
                (progress, job_id)
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job and its associated files.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if deleted successfully
        """
        # Delete from database
        with get_db_context() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM job_logs WHERE job_id = ?", (job_id,))
            cursor.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            conn.commit()
        
        # Delete job directory
        job_dir = os.path.join(self.jobs_dir, job_id)
        if os.path.exists(job_dir):
            shutil.rmtree(job_dir)
        
        # Delete model directory
        model_dir = os.path.join(self.models_dir, job_id)
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        
        logger.info(f"Deleted job {job_id}")
        return True
    
    # ==================== Log Operations ====================
    
    def add_job_log(self, job_id: str, level: str, message: str) -> None:
        """
        Add a log entry for a job.
        
        Args:
            job_id: Job identifier
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: Log message
        """
        with get_db_context() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO job_logs (job_id, log_level, message, timestamp)
                VALUES (?, ?, ?, ?)
            """, (job_id, level, message, datetime.utcnow().isoformat()))
            conn.commit()
    
    def get_job_logs(self, job_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent logs for a job.
        
        Args:
            job_id: Job identifier
            limit: Maximum number of logs to return
            
        Returns:
            List of log entries
        """
        with get_db_context() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT log_level, message, timestamp
                FROM job_logs
                WHERE job_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (job_id, limit))
            
            logs = []
            for row in cursor.fetchall():
                logs.append({
                    "level": row[0],
                    "message": row[1],
                    "timestamp": row[2]
                })
            
            return list(reversed(logs))
    
    # ==================== Model Storage Operations ====================
    
    def get_model_output_path(self, job_id: str) -> str:
        """
        Get the output path for a job's model.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Path to model output directory
        """
        return os.path.join(self.models_dir, job_id)
    
    def model_exists(self, job_id: str) -> bool:
        """
        Check if a trained model exists for a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if model exists
        """
        model_path = self.get_model_output_path(job_id)
        return os.path.exists(model_path) and os.listdir(model_path)
    
    def create_model_archive(self, job_id: str) -> Optional[str]:
        """
        Create a ZIP archive of the trained model.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Path to ZIP file or None if model doesn't exist
        """
        model_path = self.get_model_output_path(job_id)
        if not self.model_exists(job_id):
            return None
        
        zip_path = os.path.join(self.models_dir, f"{job_id}.zip")
        
        # Remove existing archive
        if os.path.exists(zip_path):
            os.remove(zip_path)
        
        create_zip_archive(model_path, zip_path)
        logger.info(f"Created model archive: {zip_path}")
        
        return zip_path
    
    def get_model_archive_path(self, job_id: str) -> Optional[str]:
        """
        Get path to model archive if it exists.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Path to ZIP file or None
        """
        zip_path = os.path.join(self.models_dir, f"{job_id}.zip")
        if os.path.exists(zip_path):
            return zip_path
        return None
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage stats
        """
        def get_dir_size(path: str) -> int:
            total = 0
            for dirpath, _, filenames in os.walk(path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total += os.path.getsize(fp)
            return total
        
        jobs_size = get_dir_size(self.jobs_dir) if os.path.exists(self.jobs_dir) else 0
        models_size = get_dir_size(self.models_dir) if os.path.exists(self.models_dir) else 0
        uploads_size = get_dir_size(settings.UPLOADS_DIR) if os.path.exists(settings.UPLOADS_DIR) else 0
        
        return {
            "jobs_size_mb": round(jobs_size / (1024 * 1024), 2),
            "models_size_mb": round(models_size / (1024 * 1024), 2),
            "uploads_size_mb": round(uploads_size / (1024 * 1024), 2),
            "total_size_mb": round((jobs_size + models_size + uploads_size) / (1024 * 1024), 2),
        }
    
    # ==================== Helper Methods ====================
    
    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert database row to dictionary."""
        result = dict(row)
        
        # Parse JSON fields
        if result.get("hyperparameters"):
            try:
                result["hyperparameters"] = json.loads(result["hyperparameters"])
            except json.JSONDecodeError:
                result["hyperparameters"] = {}
        
        # Parse datetime fields
        for field in ["created_at", "started_at", "finished_at"]:
            if result.get(field):
                try:
                    result[field] = datetime.fromisoformat(result[field])
                except ValueError:
                    pass
        
        return result


# Singleton instance
storage_service = StorageService()
