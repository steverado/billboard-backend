import os
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# Hardcoded template billboard canvas sizes (width x height in pixels)
TEMPLATE_CANVAS = {
    "subway-entrance": (445, 214),  # confirmed measurement
    "times-square": (500, 250),  # placeholder until measured
    "subway-underground": (400, 200),  # placeholder until measured
}


def preprocess_user_image(input_path: str, template_id: str, output_dir: str) -> str:
    """
    Preprocess uploaded user image:
    - Convert to PNG
    - Crop/resize to billboard canvas size for template
    - Save normalized image into job folder
    Returns: Path to normalized PNG
    """
    if template_id not in TEMPLATE_CANVAS:
        raise ValueError(f"Unknown template_id: {template_id}")

    target_w, target_h = TEMPLATE_CANVAS[template_id]

    try:
        with Image.open(input_path) as img:
            logger.info(f"[Preprocess] Opened user image: {img.size}, mode={img.mode}")

            # Convert to RGBA for consistency (handles transparency)
            img = img.convert("RGBA")

            # Resize with aspect ratio preserved, then center-crop
            img_ratio = img.width / img.height
            target_ratio = target_w / target_h

            if img_ratio > target_ratio:
                # Wider than target → fit height, crop width
                new_h = target_h
                new_w = int(new_h * img_ratio)
            else:
                # Taller than target → fit width, crop height
                new_w = target_w
                new_h = int(new_w / img_ratio)

            resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Center crop
            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
            right = left + target_w
            bottom = top + target_h
            cropped = resized.crop((left, top, right, bottom))

            # Save normalized file
            normalized_path = os.path.join(output_dir, "normalized.png")
            cropped.save(normalized_path, format="PNG")
            logger.info(f"[Preprocess] Saved normalized image at {normalized_path}")

            return normalized_path

    except Exception as e:
        logger.error(f"[Preprocess] Failed: {e}")
        raise
