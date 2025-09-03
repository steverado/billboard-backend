# app/main.py — DROP-IN (canonical presign + unified status endpoints)

import uuid
import shutil
import json
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from os.path import basename

import boto3
from botocore.exceptions import ClientError

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.exceptions import RequestValidationError
from starlette.status import HTTP_400_BAD_REQUEST

from redis import Redis
from rq import Queue
from rq.job import Job

from app.job_store import job_store
from app.config import USE_PRESIGNED_URLS, PRESIGNED_URL_EXPIRES_SECS  # <-- no presign import here
from .config import (
    TMP_DIR,
    TMP_MAX_AGE,
    MAX_UPLOAD_MB,
    TEMPLATE_REGISTRY,
    REDIS_URL,
    QUEUE_NAME,
)

# -------------------- AWS / S3 (canonical) --------------------

AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
_s3 = boto3.client("s3", region_name=AWS_DEFAULT_REGION)

def presign_url(key: str, expires: int) -> str:
    """
    Generate a presigned URL that forces a real file download in browsers.
    """
    return _s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={
            "Bucket": AWS_S3_BUCKET,
            "Key": key,
            "ResponseContentDisposition": f'attachment; filename="{basename(key)}"',
            "ResponseContentType": "video/mp4",
        },
        ExpiresIn=expires,
    )

# -------------------- Redis / RQ --------------------

redis_conn = Redis.from_url(REDIS_URL)
rq_queue = Queue(QUEUE_NAME, connection=redis_conn)

# -------------------- App / Logger --------------------

logger = logging.getLogger("uvicorn.error")
app = FastAPI()

# -------------------- Health / Root --------------------

@app.get("/", include_in_schema=False)
def root():
    return {"ok": True, "service": "billboard-backend", "docs": "/docs"}

@app.get("/healthz", include_in_schema=False)
def healthz():
    return JSONResponse({"ok": True, "service": "billboard-backend"}, status_code=200)

@app.get("/health", include_in_schema=False)
def health_alias():
    return {"ok": True}

# -------------------- CORS --------------------

ALLOWED_ORIGINS = [
    "https://nycbillboardgenerator.com",
    "https://www.nycbillboardgenerator.com",
    "http://localhost:5173",  # Vite dev
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.lovable\.app",
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["ETag", "Accept-Ranges", "Content-Range", "Content-Length"],
    allow_credentials=False,
)

# Ensure tmp dir
os.makedirs(TMP_DIR, exist_ok=True)

# -------------------- Upload preprocessing (kept) --------------------

def preprocess_user_image(upload: UploadFile, input_path: Path) -> None:
    """Validate and save user-uploaded image (unused in S3 path but kept for local flow)."""
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
        raise ValueError("Invalid image file. Please upload a valid PNG/JPG under 10MB.") from e

# -------------------- Templates listing (kept) --------------------

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
        try:
            ids = [
                d for d in os.listdir("templates")
                if os.path.isdir(os.path.join("templates", d))
            ]
        except FileNotFoundError:
            ids = []
    return [{"id": t, "name": t.replace("-", " ").title()} for t in ids]

app.include_router(router)

# -------------------- Validation errors --------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=HTTP_400_BAD_REQUEST, content={"detail": exc.errors()})

# -------------------- Generate --------------------

@app.post("/generate")
async def generate_asset(
    file: UploadFile = File(...),
    # Accept both for backwards-compat: prefer `template`, fall back to `template_name`
    template: str | None = Form(None),
    template_name: str | None = Form(None),
):
    chosen_template = template or template_name
    if not chosen_template:
        return JSONResponse(status_code=400, content={"detail": "Missing form field: 'template'"})

    # Validate template against registry if you keep TEMPLATE_REGISTRY
    try:
        template_info = next((t for t in TEMPLATE_REGISTRY if t["id"] == chosen_template), None)
    except NameError:
        template_info = {"id": chosen_template}
    if not template_info:
        return JSONResponse(status_code=400, content={"detail": "Invalid template name"})

    # Create a public job id
    job_id = str(uuid.uuid4())
    try:
        job_store.set(job_id, {"job_id": job_id, "status": "queued", "progress": 5})
    except Exception:
        pass

    # --- Upload input directly to S3 ---
    if not AWS_S3_BUCKET:
        logger.error("[Generate] AWS_S3_BUCKET not set in BACKEND service")
        return JSONResponse(status_code=500, content={"detail": "AWS_S3_BUCKET not configured"})

    input_key = f"inputs/{job_id}/user.png"
    extra_args = {"ContentType": file.content_type or "application/octet-stream"}

    try:
        try:
            file.file.seek(0)
        except Exception:
            pass

        _s3.upload_fileobj(file.file, AWS_S3_BUCKET, input_key, ExtraArgs=extra_args)

        head = _s3.head_object(Bucket=AWS_S3_BUCKET, Key=input_key)
        size = head.get("ContentLength", 0)
        logger.info(f"[Generate] Uploaded input -> s3://{AWS_S3_BUCKET}/{input_key} ({size} bytes) region={AWS_DEFAULT_REGION}")

    except ClientError as e:
        err = e.response.get("Error", {})
        code = err.get("Code", "ClientError")
        msg = err.get("Message", str(e))
        logger.error(f"[Generate] S3 client error: code={code} message={msg}")
        return JSONResponse(status_code=500, content={"detail": f"S3 error: {code}: {msg}"})
    except Exception as e:
        logger.exception("[Generate] Unexpected error uploading to S3")
        return JSONResponse(status_code=500, content={"detail": "Unexpected error uploading to S3"})

    # Enqueue worker job (RQ job_id == public id)
    try:
        from app.tasks import process_video_task  # lazy import to avoid circulars
        rq_queue.enqueue(
            process_video_task,
            chosen_template,
            input_key,                           # S3 input key
            job_id,
            f"outputs/{job_id}/output.mp4",      # S3 output key
            job_id=job_id,
            result_ttl=86400,
            ttl=86400,
            job_timeout=3600,
        )
        logger.info(f"[Generate] Enqueued job {job_id}")
    except Exception:
        logger.exception("[Generate] Failed to enqueue job")
        return JSONResponse(status_code=500, content={"detail": "Failed to enqueue processing job"})

    return {"job_id": job_id, "status": "queued"}

# -------------------- Startup cleanup (kept) --------------------

@app.on_event("startup")
def cleanup_tmp_dir():
    """Clean old files in TMP_DIR at startup."""
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

# -------------------- Unified Status / Output / Download --------------------

# Serve BOTH endpoints with the same handler (per Lovable)
@app.get("/job-status/{job_id}")
@app.get("/status/{job_id}")
def job_status_alias(job_id: str, request: Request):
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        return JSONResponse({"job_id": job_id, "status": "not_found"}, status_code=404)

    rq_status = job.get_status(refresh=True)  # queued | started | finished | failed | deferred
    meta = job.meta or {}

    status_str = meta.get("status") or rq_status
    progress = int(meta.get("progress", 0))
    stage = meta.get("stage")

    # Prefer worker-provided presigned; otherwise presign here if finished
    key = meta.get("s3_key") or f"outputs/{job_id}/output.mp4"
    presigned = meta.get("url")
    if (not presigned) and USE_PRESIGNED_URLS and (status_str == "finished" or rq_status == "finished"):
        try:
            presigned = presign_url(key, expires=PRESIGNED_URL_EXPIRES_SECS)
        except Exception:
            presigned = None

    # Use direct S3 presigned for download (avoid redirect handling quirks)
    public_base = os.getenv("PUBLIC_API_BASE_URL")
    if public_base:
        backend_download = f"{public_base.rstrip('/')}/download/{job_id}"
    else:
        base = str(request.base_url).rstrip('/')
        if base.startswith("http://"):
            base = "https://" + base[len("http://"):]
        backend_download = f"{base}/download/{job_id}"
    download_url = presigned or backend_download

    normalized = {
        "job_id": job_id,
        "status": status_str,                 # queued | processing | finished | error
        "progress": progress,
        "stage": stage,
        "url": presigned,                     # Lovable consumes this for the player

        # Helpful extras (harmless if unused)
        "output_url": presigned,
        "download_url": download_url,
        "mime_type": "video/mp4" if presigned else None,
        "filename": f"{job_id}.mp4" if presigned else None,

        # Debug / fallback data
        "s3_bucket": meta.get("s3_bucket"),
        "s3_key": key,
        "size": meta.get("size"),
        "raw": meta,

        # Nested result (some templates read this)
        "result": {
            "url": presigned,
            "download_url": download_url,
        },
    }
    return JSONResponse(normalized, status_code=200, headers={"Cache-Control": "no-store"})

@app.get("/output/{job_id}")
def get_output(job_id: str):
    """
    Return a presigned URL for the output if finished.
    """
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail="Job not found")

    rq_status = job.get_status(refresh=True)
    meta = job.meta or {}

    if rq_status != "finished" and meta.get("status") != "finished":
        return {
            "job_id": job_id,
            "status": meta.get("status", rq_status),
            "progress": meta.get("progress", 0),
            "stage": meta.get("stage"),
            "error": meta.get("error"),
        }

    url = meta.get("url")
    if not url:
        if not USE_PRESIGNED_URLS:
            raise HTTPException(status_code=500, detail="Presigned URLs disabled")
        output_key = f"outputs/{job_id}/output.mp4"
        try:
            url = presign_url(output_key, expires=PRESIGNED_URL_EXPIRES_SECS)
        except Exception:
            raise HTTPException(status_code=500, detail="Failed to generate download URL")

    return {"job_id": job_id, "status": "finished", "url": url, "expires_in": PRESIGNED_URL_EXPIRES_SECS}

@app.get("/download/{job_id}")
def download_output(job_id: str):
    """
    Browser-friendly download: redirect directly to S3 (attachment).
    """
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        return JSONResponse({"job_id": job_id, "error": "not_found"}, status_code=404)

    status = job.get_status(refresh=True)
    meta = job.meta or {}
    status = meta.get("status") or status
    if status != "finished":
        return JSONResponse({"job_id": job_id, "error": "not_ready", "status": status}, status_code=409)

    url = meta.get("url")
    key = meta.get("s3_key") or f"outputs/{job_id}/output.mp4"

    if (not url) and USE_PRESIGNED_URLS:
        try:
            url = presign_url(key, expires=PRESIGNED_URL_EXPIRES_SECS)
        except Exception as e:
            return JSONResponse({"job_id": job_id, "error": "presign_failed", "detail": str(e)}, status_code=500)

    if not url:
        return JSONResponse({"job_id": job_id, "error": "no_output_url"}, status_code=500)

    return RedirectResponse(url, status_code=307)

# -------------------- EOF --------------------




