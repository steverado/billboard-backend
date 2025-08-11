import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

log = logging.getLogger(__name__)

# --- Load .env from project root (../.env relative to this file) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# ---------------------------
# Directories / limits
# ---------------------------
TMP_DIR = Path(os.getenv("TMP_DIR", PROJECT_ROOT / "tmp"))
TMP_MAX_AGE = int(os.getenv("TMP_MAX_AGE", "3600"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))

# ---------------------------
# Template registry
# ---------------------------
TEMPLATE_REGISTRY_PATH = Path(__file__).resolve().parent / "template_registry.json"
TEMPLATE_REGISTRY = []
if TEMPLATE_REGISTRY_PATH.exists():
    try:
        TEMPLATE_REGISTRY = json.loads(TEMPLATE_REGISTRY_PATH.read_text())
    except Exception as e:
        log.warning(
            f"Failed to load template registry: {e}; continuing with empty list"
        )

ALLOWED_TEMPLATES = [t.get("id") for t in TEMPLATE_REGISTRY if t.get("id")]
TEMPLATE_CANVAS = {
    t["id"]: (t["canvas_size"]["width"], t["canvas_size"]["height"])
    for t in TEMPLATE_REGISTRY
    if t.get("id") and t.get("canvas_size")
}

# ---------------------------
# Redis / RQ
# ---------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
QUEUE_NAME = os.getenv("QUEUE_NAME", "billboard")


# ---------------------------
# S3 / Presign helpers
# ---------------------------
def get_region() -> str:
    # support both AWS_REGION and legacy AWS_DEFAULT_REGION
    return os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "us-east-1")


def get_bucket() -> str:
    # prefer AWS_S3_BUCKET, fall back to S3_BUCKET
    return os.getenv("AWS_S3_BUCKET") or os.getenv("S3_BUCKET") or ""


USE_PRESIGNED_URLS = os.getenv("USE_PRESIGNED_URLS", "true").lower() == "true"
PRESIGNED_URL_EXPIRES_SECS = int(os.getenv("PRESIGNED_URL_EXPIRES_SECS", "3600"))


def create_s3_client():
    # Lazy import so module import stays fast
    import boto3

    return boto3.client("s3", region_name=get_region())


def s3_key_for_job(job_id: str, ext: str = "mp4") -> str:
    return f"outputs/{job_id}.{ext}"


def presign_url(key: str, expires: int | None = None) -> str:
    if not USE_PRESIGNED_URLS:
        raise RuntimeError("presign_url called but USE_PRESIGNED_URLS is false.")
    bucket = get_bucket()
    if not bucket:
        raise RuntimeError("AWS_S3_BUCKET (or S3_BUCKET) is not set; cannot presign.")
    s3 = create_s3_client()
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=int(expires or PRESIGNED_URL_EXPIRES_SECS),
    )
