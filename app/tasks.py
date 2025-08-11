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


def process_video_task(
    template_name: str, user_image_path: str, job_id: str, output_path: str
) -> Dict[str, Any]:
    """
    Background task executed by RQ worker.

    Args:
      - template_name: template id (e.g. 'subway-entrance')
      - user_image_path: path to the saved user image (PNG/JPG)
      - job_id: UUID string
      - output_path: where compositor should write the final mp4
    """
    logger.info(f"[Worker] Start job {job_id} template={template_name}")
    job_store.update(
        job_id, {"status": "processing", "template": template_name, "progress": 1}
    )

    def _progress_cb(pct: int) -> None:
        try:
            job_store.update(job_id, {"status": "processing", "progress": int(pct)})
        except Exception:
            pass

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Optional: allow bypassing color-matching if it's flaky today
        # if os.getenv("SKIP_COLOR_MATCH", "0") == "1":
        #     EnhancedQualityCompositor.DISABLE_COLOR_MATCH = True

        comp = EnhancedQualityCompositor(quality_preset="high")
        final_path = asyncio.run(
            comp.process_video(
                template_id=template_name,
                user_file=user_image_path,
                job_id=job_id,
                progress_callback=_progress_cb,
                output_path=output_path,
            )
        )

        if not final_path or not os.path.exists(final_path):
            raise FileNotFoundError(f"Expected output not found at {final_path}")
        size = os.path.getsize(final_path)
        if size < 1024:
            raise ValueError(f"Output too small at {final_path} ({size} bytes)")
        logger.info(f"[Worker] Rendered file {final_path} ({size} bytes)")

        ext = final_path.rsplit(".", 1)[-1].lower() if "." in final_path else "mp4"
        key = s3_key_for_job(job_id, ext)
        s3_uri = _upload_to_s3_no_acl(final_path, key)
        logger.info(f"[Worker] Uploaded to {s3_uri}")

        job_store.update(
            job_id,
            {
                "status": "finished",
                "output_key": key,
                "progress": 100,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        logger.info(f"[Worker] Job {job_id} finished")
        return {"job_id": job_id, "status": "finished", "output_key": key}

    except ClientError as ce:
        logger.error(f"[Worker] S3 upload failed for job {job_id}: {ce}")
        job_store.update(job_id, {"status": "error", "error": "s3_upload_failed"})
        raise
    except Exception as e:
        logger.err
