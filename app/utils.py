import os
import shutil
import time
import asyncio
import logging

logger = logging.getLogger(__name__)


async def cleanup_temp_files(file_paths: list[str], delay_hours: int = 24):
    """
    Legacy cleanup: remove individual temp files after a delay.
    Keeping this in case it's useful for one-off file deletions.
    """
    await asyncio.sleep(delay_hours * 60 * 60)

    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup {file_path}: {e}")


def create_job_folder(job_id: str, base_dir: str) -> str:
    """
    Create a temp folder for a given job_id inside base_dir.
    Returns the full path to the folder.
    """
    job_dir = os.path.join(base_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)
    logger.info(f"Created job folder: {job_dir}")
    return job_dir


def cleanup_job_folder(job_dir: str):
    """
    Delete a job folder and all its contents.
    """
    if os.path.exists(job_dir):
        shutil.rmtree(job_dir, ignore_errors=True)
        logger.info(f"Cleaned up job folder: {job_dir}")


def cleanup_old_jobs(base_dir: str, max_age: int):
    """
    Delete job folders older than max_age seconds.
    Useful as a safety net for crashed/incomplete jobs.
    """
    now = time.time()
    if not os.path.exists(base_dir):
        return

    for folder in os.listdir(base_dir):
        path = os.path.join(base_dir, folder)
        if os.path.isdir(path):
            folder_age = now - os.path.getmtime(path)
            if folder_age > max_age:
                shutil.rmtree(path, ignore_errors=True)
                logger.info(f"Removed old job folder: {path}")


# Schedule periodic cleanup of old jobs

from PIL import Image
import os
from .config import TEMPLATE_REGISTRY
import logging

logger = logging.getLogger(__name__)


def preprocess_user_image(input_path: str, output_path: str, template_id: str):
    try:
        # Load image
        image = Image.open(input_path).convert("RGB")
        original_width, original_height = image.size
        logger.info(
            f"[Preprocessing] Original image size: {original_width}x{original_height}"
        )

        # Get target canvas size from template registry
        matching_templates = [t for t in TEMPLATE_REGISTRY if t["id"] == template_id]
        if not matching_templates:
            raise ValueError(f"Invalid template_id: {template_id}")
        canvas_size = matching_templates[0]["canvas_size"]
        target_width = canvas_size["width"]
        target_height = canvas_size["height"]
        logger.info(
            f"[Preprocessing] Target canvas size: {target_width}x{target_height}"
        )

        # Calculate aspect ratios
        target_aspect = target_width / target_height
        original_aspect = original_width / original_height

        # Crop to match target aspect ratio (center-crop)
        if original_aspect > target_aspect:
            # Image is wider than target → crop sides
            new_width = int(original_height * target_aspect)
            left = (original_width - new_width) // 2
            right = left + new_width
            top = 0
            bottom = original_height
        else:
            # Image is taller than target → crop top/bottom
            new_height = int(original_width / target_aspect)
            top = (original_height - new_height) // 2
            bottom = top + new_height
            left = 0
            right = original_width

        cropped = image.crop((left, top, right, bottom))
        logger.info(f"[Preprocessing] Cropped to {cropped.size[0]}x{cropped.size[1]}")

        # Resize to fit canvas
        resized = cropped.resize((target_width, target_height), Image.LANCZOS)
        resized.save(output_path)
        logger.info(f"[Preprocessing] Final image saved to {output_path}")

    except Exception as e:
        logger.error(f"[Preprocessing] Failed to preprocess image {input_path}: {e}")
        raise ValueError(
            "Invalid image file. Please upload a valid PNG/JPG under 10MB."
        ) from e
