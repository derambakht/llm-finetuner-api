"""
FastAPI dependencies for dependency injection.
"""
from typing import Generator
import sqlite3
from contextlib import contextmanager

from .config import settings


def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Get a database connection.
    
    Yields:
        SQLite database connection
    """
    # Extract path from sqlite:/// URL
    db_path = settings.DATABASE_URL.replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def get_db_context() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager for database connections.
    
    Yields:
        SQLite database connection
    """
    db_path = settings.DATABASE_URL.replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_database() -> None:
    """Initialize the database schema."""
    with get_db_context() as conn:
        cursor = conn.cursor()
        
        # Create jobs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                dataset_source TEXT NOT NULL,
                dataset_path TEXT,
                method TEXT NOT NULL DEFAULT 'lora',
                hyperparameters TEXT,
                hf_token TEXT,
                description TEXT,
                status TEXT NOT NULL DEFAULT 'queued',
                progress REAL DEFAULT 0.0,
                error_message TEXT,
                output_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                finished_at TIMESTAMP
            )
        """)
        
        # Create job_logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS job_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                log_level TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES jobs(id)
            )
        """)
        
        conn.commit()
