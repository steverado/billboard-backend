# app/main.py  — DROP-IN

import uuid
import shutil
import json
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.exceptions import RequestValidationError
from starlette.status import HTTP_400_BAD_REQUEST

from redis import Redis
from rq import Queue
from rq.job import Job

from app.job_store import job_store
from app.config import USE_PRESIGNED_URLS, PRESIGNED_URL_EXPIRES_SECS, presign_url
from .config import (
    TMP_DIR,
    TMP_MAX_AGE,
    MAX_UPLOAD_MB,
    TEMPLATE_REGISTRY,
    REDIS_URL,
    QUEUE_NAME,
)

# Redis/RQ
redis_conn = Redis.from_url(REDIS_URL)
rq_queue = Queue(QUEUE_NAME, connection=redis_conn)

# Logger
logger = logging.getLogger("uvicorn.error")

# FastAPI
app = FastAPI()


# --------- Health / Root ---------

@app.get("/", include_in_schema=False)
def root():
    return {"ok": True, "service": "billboard-backend", "docs": "/docs"}

@app.get("/healthz", include_in_schema=False)
def healthz():
    return JSONResponse({"ok": True, "service": "billboard-backend"}, status_code=200)

@app.get("/health", include_in_schema=False)
def health_alias():
    return {"ok": True}


# --------- CORS (safe prod config) ---------
# Exact production origins + regex for Lovable preview domains; no credentials.
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


# --------- Upload preprocessing (kept) ---------

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


# --------- Templates listing (kept) ---------

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
        # Fallback: scan templates dir (expects template.mp4 & tracking.json)
        try:
            ids = [
                d for d in os.listdir("templates")
                if os.path.isdir(os.path.join("templates", d))
            ]
        except FileNotFoundError:
            ids = []
    return [{"id": t, "name": t.replace("-", " ").title()} for t in ids]

app.include_router(router)


# --------- Validation errors ---------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=HTTP_400_BAD_REQUEST, content={"detail": exc.errors()})


# --------- Generate ---------

@app.post("/generate")
async def generate_asset(
    file: UploadFile = File(...),
    # Accept both for backwards-compat: prefer `template`, fall back to `template_name`
    template: str | None = Form(None),
    template_name: str | None = Form(None),
):
    import uuid
    import os
    import logging
    import boto3
    from botocore.exceptions import ClientError
    from fastapi.responses import JSONResponse

    logger = logging.getLogger("uvicorn.error")

    chosen_template = template or template_name
    if not chosen_template:
        return JSONResponse(status_code=400, content={"detail": "Missing form field: 'template'"})

    # Validate template against registry if you keep TEMPLATE_REGISTRY
    try:
        template_info = next((t for t in TEMPLATE_REGISTRY if t["id"] == chosen_template), None)
    except NameError:
        template_info = {"id": chosen_template}  # if TEMPLATE_REGISTRY not present, skip strict validation
    if not template_info:
        return JSONResponse(status_code=400, content={"detail": "Invalid template name"})

    # Create a public job id
    job_id = str(uuid.uuid4())
    try:
        job_store.set(job_id, {"job_id": job_id, "status": "queued", "progress": 5})
    except Exception:
        pass

    # --- Upload input directly to S3 (minimal & robust) ---
    bucket = os.getenv("AWS_S3_BUCKET")
    if not bucket:
        logger.error("[Generate] AWS_S3_BUCKET not set in BACKEND service")
        return JSONResponse(status_code=500, content={"detail": "AWS_S3_BUCKET not configured"})

    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    s3 = boto3.client("s3", region_name=region)

    input_key = f"inputs/{job_id}/user.png"
    extra_args = {"ContentType": file.content_type or "application/octet-stream"}

    try:
        # Ensure the file pointer is at start (some middlewares may have read from it)
        try:
            file.file.seek(0)
        except Exception:
            pass

        s3.upload_fileobj(file.file, bucket, input_key, ExtraArgs=extra_args)

        # Verify and log size
        head = s3.head_object(Bucket=bucket, Key=input_key)
        size = head.get("ContentLength", 0)
        logger.info(f"[Generate] Uploaded input -> s3://{bucket}/{input_key} ({size} bytes) region={region}")

    except ClientError as e:
        err = e.response.get("Error", {})
        code = err.get("Code", "ClientError")
        msg = err.get("Message", str(e))
        logger.error(f"[Generate] S3 client error: code={code} message={msg}")
        return JSONResponse(status_code=500, content={"detail": f"S3 error: {code}: {msg}"})
    except Exception as e:
        logger.exception("[Generate] Unexpected error uploading to S3")
        return JSONResponse(status_code=500, content={"detail": "Unexpected error uploading to S3"})
    # --- end upload ---

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
    except Exception as e:
        logger.exception("[Generate] Failed to enqueue job")
        return JSONResponse(status_code=500, content={"detail": "Failed to enqueue processing job"})

    return {"job_id": job_id, "status": "queued"}



# --------- Startup cleanup (kept) ---------

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


# --------- Status / Output / Download ---------

@app.get("/status/{job_id}")
def status(job_id: str):
    """
    Canonical status endpoint backed by RQ job.meta in Redis.
    Returns: job_id, rq_status, status, progress, stage, url, error
    """
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        return JSONResponse(status_code=404, content={"detail": "job not found"})

    rq_status = job.get_status(refresh=True)  # queued | started | finished | failed | deferred
    meta = job.meta or {}
    result = job.result if rq_status == "finished" else None
    url = meta.get("url") or (result.get("url") if isinstance(result, dict) else None)

    payload = {
        "job_id": job_id,
        "rq_status": rq_status,
        "status": meta.get("status", rq_status),
        "progress": meta.get("progress", 0),
        "stage": meta.get("stage"),
        "url": url,
        "error": meta.get("error") or (str(job.exc_info) if rq_status == "failed" else None),
        # If your worker set them:
        "s3_bucket": meta.get("s3_bucket"),
        "s3_key": meta.get("s3_key"),
        "size": meta.get("size"),
    }
    return Response(content=json.dumps(payload), media_type="application/json", headers={"Cache-Control": "no-store"})

from fastapi import Request

@app.get("/job-status/{job_id}")
def job_status_alias(job_id: str, request: Request):
    # Normalized alias for frontends (adds output_url, download_url, result.url)
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        return JSONResponse({"job_id": job_id, "status": "not_found"}, status_code=404)

    rq_status = job.get_status(refresh=True)
    meta = job.meta or {}

    status_str = meta.get("status", rq_status)
    progress = int(meta.get("progress", 0))

    # Prefer the worker-written presigned URL; otherwise presign on demand if finished
    presigned = meta.get("url")
    key = meta.get("s3_key") or f"outputs/{job_id}/output.mp4"

    if not presigned and USE_PRESIGNED_URLS and (status_str == "finished" or rq_status == "finished"):
        try:
            presigned = presign_url(key, expires=PRESIGNED_URL_EXPIRES_SECS)
        except Exception:
            presigned = None

    # Same-origin download alias avoids S3 CORS weirdness
    download_url = f"{str(request.base_url).rstrip('/')}/download/{job_id}"

    normalized = {
        "job_id": job_id,
        "status": status_str,                 # "queued" | "processing" | "finished" | "failed"
        "progress": progress,
        "url": presigned,                     # original key
        "output_url": presigned,              # common alt key
        "download_url": download_url,         # same-origin alias
        "mime_type": "video/mp4" if presigned else None,
        "filename": f"{job_id}.mp4" if presigned else None,
        "raw": meta,

        # Many frontends read a nested result object
        "result": {
            "url": presigned,
            "output_url": presigned,
            "download_url": download_url,
        },
    }
    return JSONResponse(normalized, status_code=200)

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
def download(job_id: str):
    """
    Redirect straight to the presigned MP4 for easy download/play.
    """
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail="Job not found")

    rq_status = job.get_status(refresh=True)
    meta = job.meta or {}
    if rq_status != "finished" and meta.get("status") != "finished":
        raise HTTPException(status_code=409, detail="Job not finished yet")

    url = meta.get("url")
    if not url:
        if not USE_PRESIGNED_URLS:
            raise HTTPException(status_code=500, detail="Presigned URLs disabled")
        output_key = f"outputs/{job_id}/output.mp4"
        url = presign_url(output_key, expires=PRESIGNED_URL_EXPIRES_SECS)

    return RedirectResponse(url, status_code=302)


