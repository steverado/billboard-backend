# app/tasks.py

import os
import io
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

AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
AWS_BUCKET = os.getenv("AWS_S3_BUCKET")
USE_PRESIGNED_URLS = os.getenv("USE_PRESIGNED_URLS", "true").lower() == "true"
PRESIGN_EXPIRES = int(os.getenv("PRESIGNED_URL_EXPIRES_SECS", "3600"))
PUBLIC_S3_URL_PREFIX = os.getenv("PUBLIC_S3_URL_PREFIX", "").rstrip("/")  # e.g., https://cdn.example.com/bucket

# Single S3 client (worker process)
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=AWS_REGION,
)


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
    """Mirror status/progress (and more) into RQ job.meta so backend can read it from Redis."""
    try:
        job = get_current_job()
        if not job:
            return
        job.meta = job.meta or {}
        job.meta.update(kvs)
        job.save_meta()
    except RedisError:
        pass
    except Exception:
        # Do not crash task if meta write fails
        logger.exception("[Task] job.meta update failed")


def _job_store_update(job_id: str, payload: Dict[str, Any]):
    try:
        job_store.set(job_id, payload)
    except Exception:
        # Non-fatal; keep going
        pass


def _stage(job_id: str, stage: str, status: str, progress: int, extra: Optional[Dict[str, Any]] = None):
    data = {"job_id": job_id, "stage": stage, "status": status, "progress": progress}
    if extra:
        data.update(extra)
    _meta_update(**data)
    _job_store_update(job_id, data)


def _fail(job_id: str, code: str, detail: str) -> None:
    """Set shared status then raise to mark RQ job as FAILED."""
    logger.error(f"[Task:{job_id}] {code}: {detail}")
    _stage(job_id, stage="error", status="error", progress=0, extra={"error": code, "detail": detail})
    raise RuntimeError(f"{code}: {detail}")


def _public_url_or_none(bucket: str, key: str) -> Optional[str]:
    """
    If PUBLIC_S3_URL_PREFIX is configured for a CDN/public bucket, build a stable URL
    (no presign). Otherwise return None.
    Example: PUBLIC_S3_URL_PREFIX=https://cdn.example.com
             -> https://cdn.example.com/<bucket>/<key>
    If your CDN already maps the bucket root, set PUBLIC_S3_URL_PREFIX to that and
    omit the bucket part in the f-string below as needed.
    """
    if not PUBLIC_S3_URL_PREFIX:
        return None
    # Common case: prefix points to domain (optionally to bucket root).
    # If your prefix already includes the bucket segment, you can drop "/{bucket}".
    return f"{PUBLIC_S3_URL_PREFIX}/{bucket}/{key}"


# --- presign helper (tasks.py) ---
import os

def _presign_or_none(bucket: str, key: str, expires: int) -> str | None:
    try:
        return s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={
                "Bucket": bucket,
                "Key": key,
                "ResponseContentDisposition": f'attachment; filename="{os.path.basename(key)}"',
                "ResponseContentType": "video/mp4",
            },
            ExpiresIn=expires,
        )
    except Exception:
        logger.exception("[Task] Failed to presign output")
        return None



def process_video_task(
    template_name: str,
    s3_input_key: str,
    job_id: str,
    s3_output_key: str,
) -> Dict[str, Any]:
    """
    Download input from S3 -> process locally -> upload output to S3 -> return URL.
    Keys (convention):
      - inputs/<job_id>/user.png
      - outputs/<job_id>/output.mp4
    """
    if not AWS_BUCKET:
        _fail(job_id, "bucket_not_configured", "AWS_S3_BUCKET is not set")

    logger.info(f"[Task] Start job={job_id} template={template_name} bucket={AWS_BUCKET}")
    _stage(job_id, stage="queued", status="queued", progress=5)

    # Resolve template assets present in the container
    template_mp4, tracking_json, base_dir = _resolve_template_paths(template_name)
    if not template_mp4 or not tracking_json:
        _fail(job_id, "missing_template_assets", f"template '{template_name}' not found in container")

    # Preflight visibility
    try:
        logger.info(f"[Preflight] base_dir={os.path.abspath(base_dir)}")
        logger.info(f"[Preflight] template_mp4={os.path.abspath(template_mp4)} exists={os.path.exists(template_mp4)}")
        logger.info(f"[Preflight] tracking_json={os.path.abspath(tracking_json)} exists={os.path.exists(tracking_json)}")
        logger.info(f"[Preflight] listdir(templates/{template_name}) -> {os.listdir(os.path.dirname(template_mp4))}")
    except Exception as e:
        logger.warning(f"[Preflight] listing failed: {e}")

    # Probe OpenCV can open the template
    try:
        import cv2
        cap = cv2.VideoCapture(template_mp4)
        if not cap.isOpened():
            _fail(job_id, "template_open_failed", f"OpenCV failed to open {template_mp4}")
        cap.release()
        logger.info(f"[Probe] Template open OK: {template_mp4}")
    except Exception as e:
        _fail(job_id, "opencv_probe_failed", str(e))

    cwd_before = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_input = os.path.join(tmpdir, "user.png")           # keep .png extension
            local_output = os.path.join(tmpdir, "final_output.mp4")  # explicit .mp4

            # --- 1) Download input ---
            _stage(job_id, stage="fetch_input", status="processing", progress=20)
            try:
                logger.info(f"[Probe] head_object bucket={AWS_BUCKET!r} key={s3_input_key!r}")
                head = s3.head_object(Bucket=AWS_BUCKET, Key=s3_input_key)
                logger.info(f"[Probe] head_object OK, ContentLength={head.get('ContentLength')}")

                resp = s3.get_object(Bucket=AWS_BUCKET, Key=s3_input_key)
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
                _fail(job_id, "input_download_failed", f"S3 get {AWS_BUCKET}/{s3_input_key}: {e}")
            except Exception as e:
                _fail(job_id, "input_download_failed", str(e))

            _stage(job_id, stage="prepare_template", status="processing", progress=35)

            # Mirror expected relative layout inside temp dir
            import shutil
            rel_tpl_dir = os.path.join(tmpdir, "templates", template_name)
            os.makedirs(rel_tpl_dir, exist_ok=True)
            shutil.copy2(template_mp4, os.path.join(rel_tpl_dir, "template.mp4"))
            shutil.copy2(tracking_json, os.path.join(rel_tpl_dir, "tracking.json"))

            # Make compositor's relative paths resolve against tmpdir
            os.chdir(tmpdir)
            os.environ["TEMPLATE_VIDEO_PATH"] = os.path.join(rel_tpl_dir, "template.mp4")
            os.environ["TRACKING_JSON_PATH"] = os.path.join(rel_tpl_dir, "tracking.json")

            # --- 2) Render ---
            _stage(job_id, stage="render", status="processing", progress=60)
            try:
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
                if out_size < 2048:
                    _fail(job_id, "tiny_output", f"{final_path} ({out_size} bytes)")

                logger.info(f"[Task] Render complete: {final_path} ({out_size} bytes)")
            except Exception as e:
                _fail(job_id, "processing_failed", str(e))

            # --- 3) Upload output ---
            _stage(job_id, stage="upload_output", status="uploading", progress=90)
            try:
                s3.upload_file(
                    final_path,
                    AWS_BUCKET,
                    s3_output_key,
                    ExtraArgs={
                        "ContentType": _guess_content_type(final_path),
                        "CacheControl": "public, max-age=31536000, immutable",
                    },
                )
                head2 = s3.head_object(Bucket=AWS_BUCKET, Key=s3_output_key)
                out_len = head2.get("ContentLength")
                logger.info(f"[Task] Uploaded -> s3://{AWS_BUCKET}/{s3_output_key} (ContentLength={out_len})")

            except ClientError as e:
                _fail(job_id, "output_upload_failed", f"S3 put {AWS_BUCKET}/{s3_output_key}: {e}")
            except Exception as e:
                _fail(job_id, "output_upload_failed", str(e))

    finally:
        try:
            os.chdir(cwd_before)
        except Exception:
            pass

    # --- 4) URL selection & final status ---
    # Prefer public URL if configured; else presign (default); else leave None (backend can presign on GET).
    direct_url = _public_url_or_none(AWS_BUCKET, s3_output_key)
    presigned_url = None
    if USE_PRESIGNED_URLS and not direct_url:
        presigned_url = _presign_or_none(AWS_BUCKET, s3_output_key, PRESIGN_EXPIRES)

    final_url = direct_url or presigned_url  # may be None if both unavailable

    # Also record bucket/key/size for backend aliases to presign on demand
    try:
        head3 = s3.head_object(Bucket=AWS_BUCKET, Key=s3_output_key)
        out_len = head3.get("ContentLength", None)
    except Exception:
        out_len = None

    _stage(
        job_id,
        stage="done",
        status="finished",
        progress=100,
        extra={
            "url": final_url,
            "s3_bucket": AWS_BUCKET,
            "s3_key": s3_output_key,
            "size": out_len,
            "use_presigned": USE_PRESIGNED_URLS,
            "public_prefix": bool(PUBLIC_S3_URL_PREFIX),
        },
    )

    logger.info(f"[Task:{job_id}] Success url={'<none>' if not final_url else final_url}")
    return {
        "job_id": job_id,
        "status": "finished",
        "url": final_url,
        "s3_bucket": AWS_BUCKET,
        "s3_key": s3_output_key,
        "size": out_len,
        "expires_in": PRESIGN_EXPIRES if presigned_url else None,
        "presigned": bool(presigned_url),
        "public_url": direct_url if direct_url else None,
    }



