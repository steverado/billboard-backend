import uuid
import shutil
import json
import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.status import HTTP_400_BAD_REQUEST
from datetime import datetime, timedelta
from pathlib import Path
import logging
from redis import Redis
from rq import Queue
from rq.job import Job
from app.job_store import job_store
from app.config import USE_PRESIGNED_URLS, PRESIGNED_URL_EXPIRES_SECS, presign_url


# from .compositor import EnhancedQualityCompositor# - Comment for better loading speeds for swagger testing#
from .config import (
    TMP_DIR,
    TMP_MAX_AGE,
    MAX_UPLOAD_MB,
    TEMPLATE_REGISTRY,
    REDIS_URL,
    QUEUE_NAME,
)


redis_conn = Redis.from_url(REDIS_URL)
rq_queue = Queue(QUEUE_NAME, connection=redis_conn)

# Set up logger
logger = logging.getLogger("uvicorn.error")

# FastAPI app
app = FastAPI()

# after: app = FastAPI()
from fastapi.responses import JSONResponse


@app.get("/healthz", include_in_schema=False)
def healthz():
    return JSONResponse({"ok": True, "service": "billboard-backend"}, status_code=200)


# optional alias if you ever point Railway to /health instead
@app.get("/health", include_in_schema=False)
def health_alias():
    return {"ok": True}


# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create tmp dir if it doesn't exist
os.makedirs(TMP_DIR, exist_ok=True)


def preprocess_user_image(upload: UploadFile, input_path: Path) -> None:
    """Validate and save user-uploaded image."""
    try:
        if not upload.content_type or not upload.content_type.startswith("image/"):
            raise ValueError("Invalid file type")

        size_mb = len(upload.file.read()) / (1024 * 1024)
        upload.file.seek(0)
        if size_mb > MAX_UPLOAD_MB:
            raise ValueError("File too large")

        with open(input_path, "wb") as out_file:
            shutil.copyfileobj(upload.file, out_file)

    except Exception as e:
        logger.error(f"[Preprocessing] Failed to preprocess image {input_path}: {e}")
        raise ValueError(
            "Invalid image file. Please upload a valid PNG/JPG under 10MB."
        ) from e


from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class TemplateItem(BaseModel):
    id: str
    name: str


@router.get("/templates", response_model=list[TemplateItem])
def list_templates():
    env_val = os.getenv("ALLOWED_TEMPLATES", "")
    ids = [t.strip() for t in env_val.split(",") if t.strip()]
    if not ids:
        # fallback: scan templates directory (expects template.mp4 & tracking.json)
        try:
            ids = [
                d
                for d in os.listdir("templates")
                if os.path.isdir(os.path.join("templates", d))
            ]
        except FileNotFoundError:
            ids = []

    # Minimal metadata (friendly name == id for now)
    return [{"id": t, "name": t.replace("-", " ").title()} for t in ids]


app.include_router(router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=HTTP_400_BAD_REQUEST,
        content={"detail": exc.errors()},
    )


@app.post("/generate")
async def generate_asset(
    file: UploadFile = File(...),
    template_name: str = Form(...),
):
    # Validate template
    template_info = next(
        (t for t in TEMPLATE_REGISTRY if t["id"] == template_name), None
    )
    if not template_info:
        raise ValueError("Invalid template name")

    try:
        logger.info(f"[Generate] Starting job for template: {template_name}")

        # Create job folder
        job_id = str(uuid.uuid4())
        job_dir = Path(TMP_DIR) / job_id
        os.makedirs(job_dir, exist_ok=True)

        job_store.set(job_id, {"job_id": job_id, "status": "pending"})

        # Save user image
        user_img_path = job_dir / "user.png"
        preprocess_user_image(file, user_img_path)

        # Output path
        output_path = job_dir / "output.mp4"

        # Enqueue background job (RQ job_id == our job_id)
        job = rq_queue.enqueue(
            "app.tasks.process_video_task",  # String reference only
            template_name,
            str(user_img_path),
            job_id,
            str(output_path),
            job_id=job_id,
            result_ttl=86400,
            ttl=86400,
            job_timeout=3600,
        )

        logger.info(f"[Generate] Enqueued job {job_id}")
        return {"job_id": job_id, "status": "queued"}

    except ValueError as ve:
        logger.warning(f"[Generate] Validation failed: {ve}")
        return JSONResponse(
            status_code=HTTP_400_BAD_REQUEST,
            content={"detail": str(ve)},
        )

    except Exception as e:
        logger.error(f"[Generate] Unexpected error: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )


@app.on_event("startup")
def cleanup_tmp_dir():
    """Clean old files in TMP_DIR at startup."""
    # Ensure tmp path exists before any cleanup
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("[Startup] Cleaning old temp files...")
    now = datetime.now()
    for subdir in Path(TMP_DIR).iterdir():
        if subdir.is_dir():
            mtime = datetime.fromtimestamp(subdir.stat().st_mtime)
            if now - mtime > timedelta(seconds=TMP_MAX_AGE):
                try:
                    shutil.rmtree(subdir)
                    logger.info(f"[Cleanup] Deleted {subdir}")
                except Exception as e:
                    logger.warning(f"[Cleanup] Failed to delete {subdir}: {e}")


from fastapi.responses import FileResponse


@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        # If Redis no longer has the job, fall back to the file existing
        output_path = Path(TMP_DIR) / job_id / "output.mp4"
        if output_path.exists():
            return {"job_id": job_id, "status": "finished", "result": str(output_path)}
        raise HTTPException(status_code=404, detail="Job not found")

    status = job.get_status()  # queued | started | finished | failed | deferred
    result = job.result if job.is_finished else None

    # Extra safety: if marked finished but file not yet present, treat as started
    if status == "finished":
        output_path = Path(TMP_DIR) / job_id / "output.mp4"
        if not output_path.exists():
            status = "started"
            result = None

    return {"job_id": job_id, "status": status, "result": result}


# === /output/{job_id} endpoint (presigned URL mode) ===
@app.get("/output/{job_id}")
def get_output(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    status = job.get("status")
    if status != "finished":
        return {"job_id": job_id, "status": status}

    key = job.get("output_key")
    if not key:
        raise HTTPException(status_code=500, detail="Output not available yet")

    if not USE_PRESIGNED_URLS:
        raise HTTPException(
            status_code=500, detail="Server misconfigured: presigned URLs disabled"
        )

    try:
        url = presign_url(key, expires=PRESIGNED_URL_EXPIRES_SECS)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to generate download URL")

    return {
        "job_id": job_id,
        "status": "finished",
        "url": url,
        "expires_in": PRESIGNED_URL_EXPIRES_SECS,
    }
