# app/tasks.py
import os
import asyncio
import tempfile
import mimetypes
import logging
from typing import Dict, Any, Tuple, Optional

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


def _resolve_template_paths(template_name: str) -> Tuple[Optional[str], Optional[str], str]:
    """
    Resolve absolute paths to template.mp4 and tracking.json in the worker container.
    Supports both repo layouts:
      /app/templates/<name>/...
      /app/app/templates/<name>/...
    Returns: (template_mp4_abs, tracking_json_abs, base_dir_used)
    """
    # /app/app/tasks.py -> /app
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    app_dir   = os.path.abspath(os.path.dirname(__file__))  # /app/app

    candidates = [
        (os.path.join(repo_root, "templates", template_name, "template.mp4"),
         os.path.join(repo_root, "templates", template_name, "tracking.json"),
         repo_root),
        (os.path.join(app_dir, "templates", template_name, "template.mp4"),
         os.path.join(app_dir, "templates", template_name, "tracking.json"),
         app_dir),
    ]

    for mp4, json_path, base in candidates:
        if os.path.exists(mp4) and os.path.exists(json_path):
            return mp4, json_path, base

    # Not found
    return None, None, repo_root


def process_video_task(
    template_name: str,
    s3_input_key: str,
    job_id: str,
    s3_output_key: str,
) -> Dict[str, Any]:
    """
    Download input from S3 -> process locally -> upload output to S3 -> return URL.
    Keys:
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

    # Resolve template asset absolute paths
    template_mp4, tracking_json, base_dir = _resolve_template_paths(template_name)
    if not template_mp4 or not tracking_json:
        logger.error(f"[Task] Missing template assets for '{template_name}' in worker image")
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
        logger.info(f"[Probe] Template open OK: {template_mp4}")
    except Exception as e:
        logger.error(f"[Probe] OpenCV probe error: {e}")
        job_store.set(job_id, {"job_id": job_id, "status": "error", "error": "opencv_probe_failed"})
        return {"job_id": job_id, "status": "failed", "error": "opencv_probe_failed"}

    # S3 client (explicit creds/region for clarity)
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )

    # Process in a temp working dir; also chdir to base_dir so relative paths in compositor resolve
    cwd_before = os.getcwd()
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

            # Mirror the expected relative layout inside the temp dir:
            #   ./templates/<template_name>/template.mp4
            #   ./templates/<template_name>/tracking.json
            import shutil

            rel_tpl_dir = os.path.join(tmpdir, "templates", template_name)
            os.makedirs(rel_tpl_dir, exist_ok=True)
            shutil.copy2(template_mp4, os.path.join(rel_tpl_dir, "template.mp4"))
            shutil.copy2(tracking_json, os.path.join(rel_tpl_dir, "tracking.json"))

            # Set CWD to the temp dir so all compositor-relative paths resolve here
            os.chdir(tmpdir)

            # Optional hints via env (harmless if compositor ignores them)
            os.environ["TEMPLATE_VIDEO_PATH"] = os.path.join(rel_tpl_dir, "template.mp4")
            os.environ["TRACKING_JSON_PATH"] = os.path.join(rel_tpl_dir, "tracking.json")

            # --- 2) Render ---
            try:
                compositor = EnhancedQualityCompositor(quality_preset="high")
                returned_path = asyncio.run(
                    compositor.process_video(
                        template_id=template_name,
                        user_file=local_input,
                        output_path=local_output_stem,  # STEM (no extension)
                        job_id=job_id,
                        # If your compositor supports explicit paths, you can also pass:
                        # template_video_path=os.environ["TEMPLATE_VIDEO_PATH"],
                        # tracking_path=os.environ["TRACKING_JSON_PATH"],
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
    finally:
        try:
            os.chdir(cwd_before)
        except Exception:
            pass

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
