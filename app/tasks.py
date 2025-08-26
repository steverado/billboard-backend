import os
import asyncio
import mimetypes
import logging
from datetime import datetime, timezone
from typing import Dict, Any
from pathlib import Path

from dotenv import load_dotenv
from botocore.exceptions import ClientError

from .job_store import job_store
from .config import create_s3_client, s3_key_for_job, get_bucket
from .compositor import EnhancedQualityCompositor

# --- Load .env from project root (../.env relative to this file) ---
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)

def _guess_content_type(path: str, default: str = "application/octet-stream") -> str:
    ctype, _ = mimetypes.guess_type(path)
    if not ctype:
        if path.lower().endswith(".mp4"):
            return "video/mp4"
        if path.lower().endswith(".mov"):
            return "video/quicktime"
        return default
    return ctype


def _upload_to_s3_no_acl(local_path: str, key: str) -> str:
    """Upload file to S3 without ACLs. Returns s3://bucket/key."""
    bucket = get_bucket()
    if not bucket:
        raise RuntimeError(
            "AWS_S3_BUCKET (or S3_BUCKET) is not set. Add it to your .env in the project root and restart API & worker."
        )
    extra_args = {
        "ContentType": _guess_content_type(local_path),
        "CacheControl": "public, max-age=31536000, immutable",
    }
    s3 = create_s3_client()
    s3.upload_file(local_path, bucket, key, ExtraArgs=extra_args)
    return f"s3://{bucket}/{key}"


def process_video_task(template_name: str, s3_input_key: str, job_id: str, s3_output_key: str) -> Dict[str, Any]:
    """
    Worker task: download input from S3, run compositor, upload result back to S3.
    """
    import boto3, os, tempfile, asyncio

    logger.info(f"[Task] Starting job {job_id} for template {template_name}")

    s3 = boto3.client("s3")
    bucket = os.getenv("AWS_S3_BUCKET")

    # Create local temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        local_input = os.path.join(tmpdir, "user.png")
        local_output = os.path.join(tmpdir, "output.mp4")

        # Download input file from S3
        try:
            s3.download_file(bucket, s3_input_key, local_input)
            logger.info(f"[Task] Downloaded input from s3://{bucket}/{s3_input_key}")
        except Exception as e:
            logger.error(f"[Task] Failed to download input: {e}")
            return {"job_id": job_id, "status": "failed", "error": "input_download_failed"}

        # Run compositor
        try:
            compositor = EnhancedQualityCompositor()
            final_path = asyncio.run(
                compositor.process_video(
                    template_name,
                    local_input,
                    local_output,
                )
            )
            logger.info(f"[Task] Video processing complete: {final_path}")
        except Exception as e:
            logger.error(f"[Task] Processing failed: {e}")
            return {"job_id": job_id, "status": "failed", "error": "processing_failed"}

        # Upload output to S3
        try:
            s3.upload_file(local_output, bucket, s3_output_key)
            logger.info(f"[Task] Uploaded result to s3://{bucket}/{s3_output_key}")
        except Exception as e:
            logger.error(f"[Task] Upload failed: {e}")
            return {"job_id": job_id, "status": "failed", "error": "output_upload_failed"}

    # Return presigned URL for download
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": s3_output_key},
        ExpiresIn=int(os.getenv("PRESIGNED_URL_EXPIRES_SECS", "3600")),
    )

    return {
        "job_id": job_id,
        "status": "finished",
        "url": url,
        "expires_in": int(os.getenv("PRESIGNED_URL_EXPIRES_SECS", "3600")),
    }

