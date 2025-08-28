# app/tasks.py
import os
import asyncio
import tempfile
import mimetypes
import logging
from typing import Dict, Any, Tuple, Optional

import boto3
from botocore.exceptions import ClientError
from rq import get_current_job
from redis.exceptions import RedisError

from .job_store import job_store
from .compositor import EnhancedQualityCompositor

logger = logging.getLogger(__name__)
if os.getenv("LOG_LEVEL", "").upper() in {"DEBUG","INFO","WARNING","ERROR"}:
    logger.setLevel(getattr(logging, os.getenv("LOG_LEVEL").upper()))
else:
    logger.setLevel(logging.INFO)


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
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # /app
    app_dir   = os.path.abspath(os.path.dirname(__file__))                      # /app/app

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


def _meta_update(**kvs):
    """Mirror status/progress into RQ job.meta so backend can read it from Redis."""
    try:
        job = get_current_job()
        if not job:
            return
        job.meta.update(kvs)
        job.save_meta()
    except RedisError:
        pass


def _fail(job_id: str, code: str, detail: str) -> None:
    """Set shared status then raise to mark RQ job as FAILED."""
    logger.error(f"[Task] {code}: {detail}")
    _meta_update(status="error", error=code, detail=detail, progress=0)
    # Keep your in-process store in sync (optional)
    try:
        job_store.set(job_id, {"job_id": job_id, "status": "error", "error": code})
    except Exception:
        pass
    raise RuntimeError(f"{code}: {detail}")


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
    _meta_update(status="queued", progress=5)
    try:
        job_store.set(job_id, {"job_id": job_id, "status": "queued", "progress": 5})
    except Exception:
        pass

    bucket = os.getenv("AWS_S3_BUCKET")
    if not bucket:
        _fail(job_id, "bucket_not_configured", "AWS_S3_BUCKET is not set")

    # Resolve template asset absolute paths
    template_mp4, tracking_json, base_dir = _resolve_template_paths(template_name)
    if not template_mp4 or not tracking_json:
        _fail(job_id, "missing_template_assets", f"template '{template_name}' not found in container")

    # Preflight: show exactly what the worker sees
    try:
        logger.info(f"[Preflight] base_dir={os.path.abspath(base_dir)}")
        logger.info(f"[Preflight] template_mp4={os.path.abspath(template_mp4)} exists={os.path.exists(template_mp4)}")
        logger.info(f"[Preflight] tracking_json={os.path.abspath(tracking_json)} exists={os.path.exists(tracking_json)}")
        logger.info(f"[Preflight] listdir(templates/{template_name}) -> {os.listdir(os.path.dirname(template_mp4))}")
    except Exception as e:
        logger.warning(f"[Preflight] listing failed: {e}")

    # Probe that OpenCV can open the template (clear error if not)
    try:
        import cv2
        cap = cv2.VideoCapture(template_mp4)
        if not cap.isOpened():
            _fail(job_id, "template_open_failed", f"OpenCV failed to open {template_mp4}")
        cap.release()
        logger.info(f"[Probe] Template open OK: {template_mp4}")
    except Exception as e:
        _fail(job_id, "opencv_probe_failed", str(e))

    # S3 client
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
            # IMPORTANT: give a real filename with an extension
            local_output = os.path.join(tmpdir, "final_output.mp4")

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
                    _fail(job_id, "input_missing_after_write", local_input)
                in_size = os.path.getsize(local_input)
                if in_size == 0:
                    _fail(job_id, "input_zero_bytes", local_input)
                logger.info(f"[Task] Downloaded {in_size} bytes -> {local_input}")
            except ClientError as e:
                _fail(job_id, "input_download_failed", f"S3 get {bucket}/{s3_input_key}: {e}")
            except Exception as e:
                _fail(job_id, "input_download_failed", str(e))

            _meta_update(status="processing", progress=40)
            try:
                job_store.set(job_id, {"job_id": job_id, "status": "processing", "progress": 40})
            except Exception:
                pass

            # Mirror expected relative layout inside the temp dir:
            #   ./templates/<template_name>/template.mp4
            #   ./templates/<template_name>/tracking.json
            import shutil

            rel_tpl_dir = os.path.join(tmpdir, "templates", template_name)
            os.makedirs(rel_tpl_dir, exist_ok=True)
            shutil.copy2(template_mp4, os.path.join(rel_tpl_dir, "template.mp4"))
            shutil.copy2(tracking_json, os.path.join(rel_tpl_dir, "tracking.json"))

            # Set CWD to the temp dir so compositor-relative paths resolve here
            os.chdir(tmpdir)

            os.environ["TEMPLATE_VIDEO_PATH"] = os.path.join(rel_tpl_dir, "template.mp4")
            os.environ["TRACKING_JSON_PATH"] = os.path.join(rel_tpl_dir, "tracking.json")

            # --- 2) Render ---
            try:
                _meta_update(status="processing", stage="render", progress=60)
                compositor = EnhancedQualityCompositor(quality_preset="high")
                returned_path = asyncio.run(
                    compositor.process_video(
                        template_id=template_name,
                        user_file=local_input,
                        job_id=job_id,
                        output_path=local_output,
                    )
                )

                candidates = []
                if returned_path:
                    candidates.append(returned_path)
                    if not returned_path.lower().endswith(".mp4"):
                        candidates.append(returned_path + ".mp4")
                candidates.append(local_output)

                final_path = next((p for p in candidates if p and os.path.exists(p)), None)
                if not final_path:
                    _fail(job_id, "no_output_file", f"candidates={candidates}")

                out_size = os.path.getsize(final_path)
                if out_size < 2048:  # slightly stricter than before
                    _fail(job_id, "tiny_output", f"{final_path} ({out_size} bytes)")

                logger.info(f"[Task] Render complete: {final_path} ({out_size} bytes)")
            except Exception as e:
                _fail(job_id, "processing_failed", str(e))

            _meta_update(status="uploading", progress=90)
            try:
                job_store.set(job_id, {"job_id": job_id, "status": "uploading", "progress": 90})
            except Exception:
                pass

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
                # HEAD to verify it’s really there
                head2 = s3.head_object(Bucket=bucket, Key=s3_output_key)
                logger.info(f"[Task] Uploaded result -> s3://{bucket}/{s3_output_key} (ContentLength={head2.get('ContentLength')})")
            except ClientError as e:
                _fail(job_id, "output_upload_failed", f"S3 put {bucket}/{s3_output_key}: {e}")
            except Exception as e:
                _fail(job_id, "output_upload_failed", str(e))

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
        _fail(job_id, "presign_failed", str(e))

    _meta_update(status="finished", progress=100, url=url)
    try:
        job_store.set(job_id, {"job_id": job_id, "status": "finished", "progress": 100, "url": url})
    except Exception:
        pass

    logger.info(f"[Task] Success: {url}")
    return {
        "job_id": job_id,
        "status": "finished",
        "url": url,
        "expires_in": int(os.getenv("PRESIGNED_URL_EXPIRES_SECS", "3600")),
    }


