from pydantic import BaseModel
from enum import Enum
from typing import Optional


class JobStatus(str, Enum):
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


class ProcessingJob(BaseModel):
    id: str
    status: JobStatus
    progress: int = 0
    download_url: Optional[str] = None
    error_message: Optional[str] = None
