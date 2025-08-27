# app/tasks.py
import os
import asyncio
import tempfile
import mimetypes
import logging
from typing import Dict, Any

import boto3
from botocore.exceptions import ClientError

from .job_store import job_store
from .compositor import EnhancedQualityCompositor

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


def process_video_task(
    template_name: str,
    s3_input_key: str,
    job_id: str,
    s3_output_key: str,
) -> Dict[str, Any]:
    """
    Download input from S3 -> process locally -> upload output to S3 -> return URL.
    Expected keys:
      - inputs/<job_id>/user.png
      - outputs/<job_id>/output.mp4
    """
    logger.info(f"[Task] Start job={job_id} template={template_name}")
    job_store.set(job_id, {"job_id": job_id, "status": "processing", "progress": 5})

    bucket = os.getenv("AWS_S3_BUCKET")
    if not bucket:
        logger.error("AWS_S3_BUCKET is not set")
        job_store.set(job_id, {"job_id": job_id, "status": "error", "error": "bucket_not_configured"})
        return {"job_id": job_id, "status": "failed", "error": "bucket_not_configured"}

    # Ensure templates exist *in the worker image*
    template_mp4 = os.path.join("templates", template_name, "template.mp4")
    tracking_json = os.path.join("templates", template_name, "tracking.json")
    for p in (template_mp4, tracking_json):
        if not os.path.exists(p):
            logger.error(f"[Task] Missing template asset in worker: {p}")
            job_store.set(job_id, {"job_id": job_id, "status": "error", "error": "missing_template_assets"})
            return {"job_id": job_id, "status": "failed", "error": "missing_template_assets"}

    # Probe codec ability to open the template
    try:
        import cv2
        cap = cv2.VideoCapture(template_mp4)
        if not cap.isOpened():
            logger.error(f"[Probe] OpenCV failed to open template video: {template_mp4}")
            job_store.set(job_id, {"job_id": job_id, "status": "error", "error": "template_open_failed"})
            return {"job_id": job_id, "status": "failed", "error": "template_open_failed"}
        cap.release()
    except Exception as e:
        logger.error(f"[Probe] OpenCV probe error: {e}")

    # S3 client (explicit creds/region for clarity)
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_input = os.path.join(tmpdir, "user.png")
            # Pass a STEM (no extension) to compositor to avoid double .mp4
            local_output_stem = os.path.join(tmpdir, "final_output")

            # --- 1) Download input (strict) ---
            try:
                logger.info(f"[Probe] bucket={bucket!r} key={s3_input_key!r}")
                head = s3.head_object(Bucket=bucket, Key=s3_input_key)
                logger.info(f"[Probe] head_object OK, ContentLength={head.get('ContentLength')}")

                resp = s3.get_object(Bucket=bucket, Key=s3_input_key)
                body = resp["Body"].read()

                os.makedirs(os.path.dirname(local_input), exist_ok=True)
                with open(local_input, "wb") as f:
                    f.write(body)

                if not os.path.exists(local_input):
                    raise FileNotFoundError(f"Post-write missing: {local_input}")
                in_size = os.path.getsize(local_input)
                if in_size == 0:
                    raise IOError(f"Downloaded input is empty: {local_input}")
                logger.info(f"[Task] Downloaded {in_size} bytes -> {local_input}")
            except Exception as e:
                logger.error(f"[Task] Input download failed: {e}")
                job_store.set(job_id, {"job_id": job_id, "status": "error", "error": "input_download_failed"})
                return {"job_id": job_id, "status": "failed", "error": "input_download_failed"}

            job_store.set(job_id, {"job_id": job_id, "status": "processing", "progress": 40})

            # --- 2) Render ---
            try:
                compositor = EnhancedQualityCompositor(quality_preset="high")
                returned_path = asyncio.run(
                    compositor.process_video(
                        template_id=template_name,
                        user_file=local_input,
                        output_path=local_output_stem,  # STEM (no extension)
                        job_id=job_id,
                    )
                )

                # Normalize final output (handle both stem / full path returns)
                candidates = []
                if returned_path:
                    candidates.append(returned_path)
                    if not returned_path.lower().endswith(".mp4"):
                        candidates.append(returned_path + ".mp4")
                candidates.append(local_output_stem + ".mp4")

                final_path = None
                for p in candidates:
                    if p and os.path.exists(p):
                        final_path = p
                        break

                if not final_path:
                    logger.error(f"[Task] Output not found. candidates={candidates}")
                    job_store.set(job_id, {"job_id": job_id, "status": "error", "error": "no_output_file"})
                    return {"job_id": job_id, "status": "failed", "error": "no_output_file"}

                out_size = os.path.getsize(final_path)
                if out_size < 1024:
                    logger.error(f"[Task] Output too small: {final_path} ({out_size} bytes)")
                    job_store.set(job_id, {"job_id": job_id, "status": "error", "error": "tiny_output"})
                    return {"job_id": job_id, "status": "failed", "error": "tiny_output"}

                logger.info(f"[Task] Render complete: {final_path} ({out_size} bytes)")
            except Exception as e:
                logger.error(f"[Task] Processing failed: {e}")
                job_store.set(job_id, {"job_id": job_id, "status": "error", "error": "processing_failed"})
                return {"job_id": job_id, "status": "failed", "error": "processing_failed"}

            job_store.set(job_id, {"job_id": job_id, "status": "processing", "progress": 90})

            # --- 3) Upload output ---
            try:
                s3.upload_file(
                    final_path,
                    bucket,
                    s3_output_key,
                    ExtraArgs={
                        "ContentType": _guess_content_type(final_path),
                        "CacheControl": "public, max-age=31536000, immutable",
                    },
                )
                logger.info(f"[Task] Uploaded result -> s3://{bucket}/{s3_output_key}")
            except Exception as e:
                logger.error(f"[Task] Output upload failed: {e}")
                job_store.set(job_id, {"job_id": job_id, "status": "error", "error": "output_upload_failed"})
                return {"job_id": job_id, "status": "failed", "error": "output_upload_failed"}

    except Exception as e:
        logger.error(f"[Task] Unexpected worker error: {e}")
        job_store.set(job_id, {"job_id": job_id, "status": "error", "error": "worker_exception"})
        return {"job_id": job_id, "status": "failed", "error": "worker_exception"}

    # --- 4) Presigned URL & final status ---
    try:
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": s3_output_key},
            ExpiresIn=int(os.getenv("PRESIGNED_URL_EXPIRES_SECS", "3600")),
        )
    except ClientError as e:
        logger.error(f"[Task] Could not sign URL: {e}")
        job_store.set(job_id, {"job_id": job_id, "status": "finished", "url": None})
        return {"job_id": job_id, "status": "finished", "url": None, "expires_in": 0}

    job_store.set(job_id, {
        "job_id": job_id,
        "status": "finished",
        "progress": 100,
        "url": url,
    })
    return {
        "job_id": job_id,
        "status": "finished",
        "url": url,
        "expires_in": int(os.getenv("PRESIGNED_URL_EXPIRES_SECS", "3600")),
    }

