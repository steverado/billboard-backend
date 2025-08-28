import uuid
import shutil
import json
import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Response
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

        # Create job id
        job_id = str(uuid.uuid4())
        job_store.set(job_id, {"job_id": job_id, "status": "pending"})

        # Upload user image directly to S3
        import boto3, os
        from botocore.exceptions import ClientError

        s3 = boto3.client("s3")
        bucket = os.getenv("AWS_S3_BUCKET")
        key = f"inputs/{job_id}/user.png"

        s3.upload_fileobj(file.file, bucket, key)
        logger.info(f"[Generate] Uploaded input to s3://{bucket}/{key}")

        # Enqueue background job (RQ job_id == our job_id)
        from app.tasks import process_video_task  # local import avoids circulars

        job = rq_queue.enqueue(
            process_video_task,
            template_name,
            key,            # pass the S3 key instead of local path
            job_id,
            f"outputs/{job_id}/output.mp4",  # output key in S3
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


@app.get("/status/{job_id}")
def status(job_id: str):
    """
    Canonical status endpoint backed by RQ job.meta in Redis.
    Returns: job_id, rq_status, status, progress, stage, url, error
    """
    try:
        job = Job.fetch(job_id, connection=redis_conn)  # public id == RQ id (your enqueue uses job_id=job_id)
    except Exception:
        # If not found in Redis, report as 404
        return JSONResponse(status_code=404, content={"detail": "job not found"})

    rq_status = job.get_status(refresh=True)  # queued | started | finished | failed | deferred
    meta = job.meta or {}
    result = job.result if rq_status == "finished" else None
    url = meta.get("url") or (result.get("url") if isinstance(result, dict) else None)

    payload = {
        "job_id": job_id,
        "rq_status": rq_status,
        "status": meta.get("status", rq_status),   # your worker sets 'status' in meta
        "progress": meta.get("progress", 0),
        "stage": meta.get("stage"),
        "url": url,
        "error": meta.get("error") or (str(job.exc_info) if rq_status == "failed" else None),
    }
    return Response(
        content=json.dumps(payload),
        media_type="application/json",
        headers={"Cache-Control": "no-store"},
    )

# Back-compat alias for any clients already calling /job-status/{job_id}
@app.get("/job-status/{job_id}")
def job_status_alias(job_id: str):
    return status(job_id)



# === /output/{job_id} endpoint (presigned URL mode) ===
@app.get("/output/{job_id}")
def get_output(job_id: str):
    """
    Return a presigned URL for the output if finished.
    Reads from RQ job.meta written by the worker task.
    """
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail="Job not found")

    rq_status = job.get_status(refresh=True)
    meta = job.meta or {}

    if rq_status != "finished" and meta.get("status") != "finished":
        # Still running (or failed)
        return {
            "job_id": job_id,
            "status": meta.get("status", rq_status),
            "progress": meta.get("progress", 0),
            "stage": meta.get("stage"),
            "error": meta.get("error"),
        }

    # Prefer URL the worker already stored (best case)
    url = meta.get("url")

    # If worker didn't store url for some reason, reconstruct key and presign here
    if not url:
        if not USE_PRESIGNED_URLS:
            raise HTTPException(status_code=500, detail="Presigned URLs disabled")
        # Your task writes outputs at: outputs/{job_id}/output.mp4
        output_key = f"outputs/{job_id}/output.mp4"
        try:
            url = presign_url(output_key, expires=PRESIGNED_URL_EXPIRES_SECS)
        except Exception:
            raise HTTPException(status_code=500, detail="Failed to generate download URL")

    return {
        "job_id": job_id,
        "status": "finished",
        "url": url,
        "expires_in": PRESIGNED_URL_EXPIRES_SECS,
    }

