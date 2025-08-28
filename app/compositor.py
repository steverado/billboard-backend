# compositor.py
# Drop-in replacement with robust reader/writer, codec fallback, and explicit failure points.
# Public API preserved: class EnhancedQualityCompositor with async process_video(...)

import cv2
import numpy as np
import json
import time
import os
import asyncio
from typing import Dict, List, Tuple, Optional, Callable, Iterable
from PIL import Image
from scipy.ndimage import gaussian_filter
import logging
import shutil, subprocess  


logger = logging.getLogger(__name__)


# ---------------------------
# Low-level I/O helpers
# ---------------------------

def _safe_mkdir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

def _safe_video_writer(
    out_path: str,
    frame_size: Tuple[int, int],
    fps: float,
    preferred_codecs: Tuple[str, ...] = ("mp4v", "avc1", "MJPG", "XVID"),
) -> cv2.VideoWriter:
    w, h = frame_size
    _safe_mkdir(os.path.dirname(out_path))
    tried = []
    vw: Optional[cv2.VideoWriter] = None

    for c in preferred_codecs:
        fourcc = cv2.VideoWriter_fourcc(*c)
        vw = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
        opened = bool(vw.isOpened())
        tried.append((c, opened))
        logger.info(f"[VideoWriter] Try codec={c} fps={fps} size={w}x{h} -> opened={opened}")
        if opened:
            break
        try:
            vw.release()
        except Exception:
            pass
        vw = None

    if vw is None:
        raise RuntimeError(f"VideoWriter failed to open. Tried={tried} out_path={out_path}")
    return vw

def _read_frames_with_meta(path: str, max_frames: Optional[int] = None):
    """Generator that yields frames and logs source metadata once."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Template video not found: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open template video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    logger.info(f"[Reader] src={path} fps={fps} size={w}x{h} frames={n}")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame, (fps, w, h, count)
        count += 1
        if max_frames is not None and count >= max_frames:
            break

    cap.release()
    if count == 0:
        raise RuntimeError(f"Source yielded 0 frames: {path}")
    
def ffmpeg_fallback(frames: Iterable[np.ndarray], out_mp4: str, fps: float, size: Optional[tuple[int,int]] = None) -> int:
    """
    Assemble frames into an .mp4 using ffmpeg (must be available in PATH).
    Auto-detects (w,h) from the first frame if size is None or invalid.
    Returns the number of frames written.
    """
    it = iter(frames)
    try:
        first = next(it)
    except StopIteration:
        raise RuntimeError("ffmpeg_fallback: no frames to write")

    # Normalize first frame to BGR and get size
    if first.ndim == 2:
        first = cv2.cvtColor(first, cv2.COLOR_GRAY2BGR)
    elif first.ndim == 3 and first.shape[2] == 4:
        first = cv2.cvtColor(first, cv2.COLOR_BGRA2BGR)

    fh, fw = first.shape[:2]
    if not size or size[0] <= 0 or size[1] <= 0:
        w, h = fw, fh
    else:
        w, h = size

    # Make frames dir and dump PNGs
    tmp_dir = os.path.join(os.path.dirname(out_mp4), "frames")
    os.makedirs(tmp_dir, exist_ok=True)

    count = 0
    # write the first frame
    if (fw, fh) != (w, h):
        first = cv2.resize(first, (w, h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(tmp_dir, f"{count:06d}.png"), first)
    count += 1

    # write the rest
    for frame in it:
        if frame is None:
            continue
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        fh2, fw2 = frame.shape[:2]
        if (fw2, fh2) != (w, h):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(tmp_dir, f"{count:06d}.png"), frame)
        count += 1

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg binary not found in PATH")

    cmd = [
        ffmpeg, "-y",
        "-framerate", str(int(round(fps or 24))),
        "-i", os.path.join(tmp_dir, "%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        out_mp4
    ]
    logger.info(f"[FFmpeg] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    if not os.path.exists(out_mp4) or os.path.getsize(out_mp4) <= 0:
        raise RuntimeError("ffmpeg_fallback: ffmpeg produced empty file")

    logger.info(f"[FFmpeg] Wrote {count} frames -> {out_mp4}")
    return count


def _write_frames(
    frames_iter: Iterable[np.ndarray],
    out_path: str,
    fps: float,
    expect_size: Tuple[int, int],
) -> int:
    """Write frames, enforce size, and log bytes. Returns frames written."""
    it = iter(frames_iter)
    try:
        first = next(it)
    except StopIteration:
        raise RuntimeError("No frames provided by upstream generator")

    # Normalize first frame to BGR
    if first.ndim == 2:
        first = cv2.cvtColor(first, cv2.COLOR_GRAY2BGR)
    elif first.ndim == 3 and first.shape[2] == 4:
        first = cv2.cvtColor(first, cv2.COLOR_BGRA2BGR)

    fh, fw = first.shape[:2]
    w, h = expect_size
    if w <= 0 or h <= 0:
        w, h = fw, fh  # auto-derive from first frame

    # Try OpenCV writer first; if no codecs, fall back to ffmpeg (with the first frame + rest).
    try:
        writer = _safe_video_writer(out_path, (w, h), fps)
        used_ffmpeg = False
    except RuntimeError as e:
        logger.warning(f"[Writer] Falling back to ffmpeg immediately: {e}")
        # re-chain the first frame + the rest into ffmpeg fallback
        def chain():
            yield first
            for f in it:
                yield f
        return ffmpeg_fallback(chain(), out_path, fps, (w, h))

    written = 0

    # Write first frame (resizing if needed)
    if (fw, fh) != (w, h):
        first_resized = cv2.resize(first, (w, h), interpolation=cv2.INTER_AREA)
    else:
        first_resized = first
    writer.write(first_resized)
    written += 1

    # Write remaining frames
    for idx, frame in enumerate(it, start=1):
        if frame is None:
            logger.error(f"[Writer] frame {idx} is None (skipping)")
            continue
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        fh2, fw2 = frame.shape[:2]
        if (fw2, fh2) != (w, h):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

        writer.write(frame)
        written += 1

    writer.release()
    time.sleep(0.1)

    exists = os.path.exists(out_path)
    size = os.path.getsize(out_path) if exists else -1
    logger.info(f"[Writer] wrote_frames={written} out_path={out_path} exists={exists} bytes={size}")

    if not exists or size <= 0 or written == 0:
        # Note: at this point the generator is consumed; a second-pass fallback would require buffering.
        # This situation is rare; it's almost always a codec-open issue handled above.
        raise RuntimeError("Writer produced an invalid file after writing frames")

    return written


class EnhancedQualityCompositor:
    """
    Professional-grade compositor adapted for FastAPI backend.
    Maintains all VFX quality improvements from Colab version.
    """

    def __init__(self, quality_preset: str = "high"):
        self.quality_preset = quality_preset
        self.frame_lookup: Dict[int, np.ndarray] = {}
        self.previous_corners: Optional[np.ndarray] = None
        self.motion_vectors: List[np.ndarray] = []

        # Quality settings based on preset
        if quality_preset == "ultra":
            self.edge_feather_radius = 4
            self.anti_alias_samples = 4
            self.motion_blur_strength = 0.8
            self.color_matching_strength = 0.7
        elif quality_preset == "high":
            self.edge_feather_radius = 3
            self.anti_alias_samples = 2
            self.motion_blur_strength = 0.6
            self.color_matching_strength = 0.5
        else:  # "medium"
            self.edge_feather_radius = 2
            self.anti_alias_samples = 1
            self.motion_blur_strength = 0.4
            self.color_matching_strength = 0.3

        logger.info(f"Enhanced Quality Compositor ({quality_preset} preset)")

    # ---------------------------
    # Data loading / utilities
    # ---------------------------

    def load_tracking_data(self, tracking_file: str):
        """Load preprocessed tracking data from file path"""
        logger.info(f"Loading tracking data from {tracking_file} ...")
        with open(tracking_file, "r") as f:
            data = json.load(f)

        tracking_frames = data["trackingData"]
        self.frame_lookup.clear()

        for frame_data in tracking_frames:
            frame_num = int(frame_data["frame"])
            corners = np.array(frame_data["corners"], dtype=np.float32)
            if corners.shape != (4, 2):
                raise ValueError(f"Tracking corners must be 4x2, got {corners.shape} at frame {frame_num}")
            self.frame_lookup[frame_num] = corners

        logger.info(f"Loaded {len(tracking_frames)} tracking frames")

    # ---------------------------
    # Compositing helpers (unchanged quality logic)
    # ---------------------------

    def create_advanced_mask(
        self,
        warped_overlay: np.ndarray,
        corners: np.ndarray,
        frame_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Create advanced alpha mask with edge feathering and natural boundaries."""
        height, width = frame_shape[:2]

        # Base mask from non-zero pixels
        if warped_overlay.ndim == 3:
            base_mask = np.any(warped_overlay > 0, axis=2).astype(np.float32)
        else:
            base_mask = (warped_overlay > 0).astype(np.float32)

        # Edge feathering
        feathered_mask = (
            gaussian_filter(base_mask, sigma=self.edge_feather_radius)
            if self.edge_feather_radius > 0
            else base_mask
        )

        # Perspective-aware softening
        corner_distances = np.linalg.norm(corners - np.mean(corners, axis=0), axis=1)
        max_distance = float(np.max(corner_distances)) if corner_distances.size else 0.0
        if max_distance > 0:
            y_coords, x_coords = np.ogrid[:height, :width]
            center_y, center_x = np.mean(corners, axis=0)[::-1]

            distance_from_center = np.sqrt(
                (y_coords - center_y) ** 2 + (x_coords - center_x) ** 2
            )
            max_billboard_distance = float(
                np.max(distance_from_center * feathered_mask)
            )
            if max_billboard_distance > 0:
                distance_factor = (
                    1.0 - (distance_from_center / max_billboard_distance) * 0.2
                )
                distance_factor = np.clip(distance_factor, 0.8, 1.0)
                feathered_mask *= distance_factor

        # Final smoothing
        final_mask = gaussian_filter(feathered_mask, sigma=0.5)
        return np.clip(final_mask, 0.0, 1.0)

    def apply_motion_blur(
        self, overlay: np.ndarray, motion_vector: np.ndarray
    ) -> np.ndarray:
        """Apply directional motion blur based on tracking movement."""
        if self.motion_blur_strength == 0 or np.linalg.norm(motion_vector) < 1.0:
            return overlay

        motion_magnitude = (
            float(np.linalg.norm(motion_vector)) * self.motion_blur_strength
        )
        if motion_magnitude < 0.5:
            return overlay

        kernel_size = int(min(motion_magnitude * 2, 15)) // 2 * 2 + 1
        if kernel_size < 3:
            return overlay

        if np.linalg.norm(motion_vector) <= 0:
            return overlay
        direction = motion_vector / np.linalg.norm(motion_vector)

        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        center = kernel_size // 2

        for i in range(kernel_size):
            t = (i - center) / center
            y = int(center + direction[1] * t * center)
            x = int(center + direction[0] * t * center)
            if 0 <= y < kernel_size and 0 <= x < kernel_size:
                kernel[y, x] = 1.0

        s = float(kernel.sum())
        if s <= 0:
            return overlay
        kernel /= s

        if overlay.ndim == 3:
            blurred = np.zeros_like(overlay)
            for c in range(overlay.shape[2]):
                blurred[:, :, c] = cv2.filter2D(overlay[:, :, c], -1, kernel)
            return blurred
        else:
            return cv2.filter2D(overlay, -1, kernel)

    def match_scene_lighting(self, overlay: np.ndarray, background_region: np.ndarray) -> np.ndarray:
    """Light/color match with scalar stats (robust to odd shapes)."""
    if self.color_matching_strength == 0 or background_region.size == 0:
        return overlay
    try:
        # Flatten background to [N, C] and compute scalar targets
        bg = background_region.reshape(-1, background_region.shape[-1] if overlay.ndim == 3 else 1)
        target_mean = float(np.mean(bg))
        target_std  = float(np.std(bg))

        ov = overlay.astype(np.float32)
        ov_mean = float(np.mean(ov))
        ov_std  = float(max(np.std(ov), 1.0))

        adj = self.color_matching_strength

        normalized = (ov - ov_mean) / ov_std
        adjusted   = normalized * (adj * target_std + (1 - adj) * ov_std) + (adj * target_mean + (1 - adj) * ov_mean)
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    except Exception as e:
        logger.warning(f"Color matching failed (robust): {e}")
        return overlay


    def apply_anti_aliasing_transform(
        self,
        overlay: np.ndarray,
        transform_matrix: np.ndarray,
        output_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Apply perspective transform with advanced anti-aliasing."""
        if self.anti_alias_samples <= 1:
            return cv2.warpPerspective(
                overlay,
                transform_matrix,
                output_shape,
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

        scale_factor = self.anti_alias_samples
        overlay_scaled = cv2.resize(
            overlay,
            (overlay.shape[1] * scale_factor, overlay.shape[0] * scale_factor),
            interpolation=cv2.INTER_LANCZOS4,
        )

        scale_matrix = np.array(
            [[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]], dtype=np.float32
        )

        scaled_transform = scale_matrix @ transform_matrix @ np.linalg.inv(scale_matrix)

        warped_scaled = cv2.warpPerspective(
            overlay_scaled,
            scaled_transform,
            (output_shape[0] * scale_factor, output_shape[1] * scale_factor),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        warped_final = cv2.resize(
            warped_scaled, output_shape, interpolation=cv2.INTER_AREA
        )

        return warped_final

    def calculate_motion_vector(self, current_corners: np.ndarray) -> np.ndarray:
        """Calculate motion vector from previous frame"""
        if self.previous_corners is None:
            self.previous_corners = current_corners.copy()
            return np.array([0.0, 0.0], dtype=np.float32)

        current_center = np.mean(current_corners, axis=0)
        previous_center = np.mean(self.previous_corners, axis=0)

        motion_vector = current_center - previous_center
        self.previous_corners = current_corners.copy()

        self.motion_vectors.append(motion_vector)
        if len(self.motion_vectors) > 5:
            self.motion_vectors.pop(0)

        if len(self.motion_vectors) > 1:
            return np.mean(self.motion_vectors, axis=0)

        return motion_vector

    def extract_background_region(
        self, frame: np.ndarray, corners: np.ndarray, expansion_factor: float = 1.3
    ) -> np.ndarray:
        """Extract background region around billboard for lighting analysis."""
        try:
            x_min, y_min = np.min(corners, axis=0).astype(int)
            x_max, y_max = np.max(corners, axis=0).astype(int)

            width = x_max - x_min
            height = y_max - y_min

            expand_w = int(width * (expansion_factor - 1) / 2)
            expand_h = int(height * (expansion_factor - 1) / 2)

            x_min_expanded = max(0, x_min - expand_w)
            y_min_expanded = max(0, y_min - expand_h)
            x_max_expanded = min(frame.shape[1], x_max + expand_w)
            y_max_expanded = min(frame.shape[0], y_max + expand_h)

            region = frame[y_min_expanded:y_max_expanded, x_min_expanded:x_max_expanded]

            billboard_mask = np.zeros(region.shape[:2], dtype=np.uint8)

            adjusted_corners = corners.copy()
            adjusted_corners[:, 0] -= x_min_expanded
            adjusted_corners[:, 1] -= y_min_expanded

            cv2.fillPoly(billboard_mask, [adjusted_corners.astype(int)], 255)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            billboard_mask = cv2.dilate(billboard_mask, kernel)

            background_pixels = region[billboard_mask == 0]

            if background_pixels.size > 0:
                return background_pixels.reshape(
                    -1, region.shape[2] if len(region.shape) == 3 else 1
                )
            else:
                return region

        except Exception:
            margin = 50
            y_start = max(0, int(np.min(corners[:, 1])) - margin)
            y_end = min(frame.shape[0], int(np.max(corners[:, 1])) + margin)
            x_start = max(0, int(np.min(corners[:, 0])) - margin)
            x_end = min(frame.shape[1], int(np.max(corners[:, 0])) + margin)
            return frame[y_start:y_end, x_start:x_end]

    def apply_enhanced_composite(
        self,
        frame: np.ndarray,
        overlay: np.ndarray,
        corners: np.ndarray,
        frame_idx: int,
    ) -> np.ndarray:
        """Apply enhanced compositing with all quality improvements."""
        try:
            motion_vector = self.calculate_motion_vector(corners)
            background_region = self.extract_background_region(frame, corners)
            lighting_corrected_overlay = self.match_scene_lighting(
                overlay, background_region
            )
            motion_blurred_overlay = self.apply_motion_blur(
                lighting_corrected_overlay, motion_vector
            )

            overlay_h, overlay_w = motion_blurred_overlay.shape[:2]
            src_corners = np.array(
                [
                    [0, 0],
                    [overlay_w - 1, 0],
                    [overlay_w - 1, overlay_h - 1],
                    [0, overlay_h - 1],
                ],
                dtype=np.float32,
            )

            transform_matrix = cv2.getPerspectiveTransform(
                src_corners, corners.astype(np.float32)
            )

            warped_overlay = self.apply_anti_aliasing_transform(
                motion_blurred_overlay,
                transform_matrix,
                (frame.shape[1], frame.shape[0]),
            )

            alpha_mask = self.create_advanced_mask(
                warped_overlay, corners, frame.shape[:2]
            )

            frame_float = frame.astype(np.float32)
            warped_float = warped_overlay.astype(np.float32)

            if warped_overlay.ndim == 3:
                alpha_3d = np.stack([alpha_mask] * 3, axis=2)
            else:
                alpha_3d = alpha_mask

            result = (1 - alpha_3d) * frame_float + alpha_3d * warped_float
            return result.astype(np.uint8)

        except Exception as e:
            logger.error(f"Enhanced composite error at frame {frame_idx}: {e}")
            return frame

    # ---------------------------
    # Public API
    # ---------------------------

    async def process_video(
        self,
        template_id: str,
        user_file: str,
        job_id: str,
        progress_callback: Optional[Callable[[int], None]] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Process video with enhanced quality - main API method.
        """
        # Resolve template assets
        template_dir = os.path.join("templates", template_id)
        template_video = os.path.join(template_dir, "template.mp4")
        tracking_data = os.path.join(template_dir, "tracking.json")

        # Validate required files
        for file_path in [template_video, tracking_data]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Template file missing: {file_path}")

        if progress_callback:
            progress_callback(30)

        # Load tracking + overlay
        self.load_tracking_data(tracking_data)
        overlay_img = Image.open(user_file).convert("RGB")
        overlay = np.array(overlay_img)

        if progress_callback:
            progress_callback(40)

        # Default output path if not provided
        if output_path is None:
            output_path = f"/tmp/output_{job_id}.mp4"

        # Render
        final_path = await self._process_enhanced_video(
            template_video,
            overlay,
            output_path,
            progress_callback,
        )

        if not final_path:
            raise RuntimeError("Enhanced video processing failed (no frames written)")

        if progress_callback:
            progress_callback(100)

        logger.info(f"Final video written to {final_path}")
        return final_path

    # ---------------------------
    # Internal rendering
    # ---------------------------

    async def _process_enhanced_video(
        self,
        video_file: str,
        overlay: np.ndarray,
        output_path: str,
        progress_callback: Optional[Callable[[int], None]] = None,
        max_frames: int = 150,
    ) -> Optional[str]:
        """
        Process video with enhanced quality, ensure file output is complete, and return the path.
        Raises RuntimeError with actionable messages on failure; returns path on success.
        """
        cap = None
        try:
            # Read frames & source meta (validates existence/open)
            frames_meta_iter = _read_frames_with_meta(video_file, max_frames=max_frames)

            fps = None
            width = None
            height = None
            processed_count = 0

            def composited_frames():
                nonlocal fps, width, height, processed_count
                for frame, meta in frames_meta_iter:
                    _fps, w, h, idx = meta
                    if fps is None:
                        fps, width, height = _fps, w, h

                    # Convert to RGB for your compositor
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    if idx in self.frame_lookup:
                        corners = self.frame_lookup[idx]
                        result_frame_rgb = self.apply_enhanced_composite(
                            frame_rgb, overlay, corners, idx
                        )
                        processed_count += 1
                    else:
                        result_frame_rgb = frame_rgb  # passthrough when no tracking

                    # Back to BGR for writer
                    yield cv2.cvtColor(result_frame_rgb, cv2.COLOR_RGB2BGR)

                    # Cooperative yield for asyncio
                    if idx % 30 == 0:
                        # allow event loop to breathe
                        # (no await inside generator; caller awaits after write)
                        pass

            # Write all frames
            written = _write_frames(
                frames_iter=composited_frames(),
                out_path=output_path,
                fps=float(fps) if fps is not None else 24.0,
                expect_size=(int(width) if width else 0, int(height) if height else 0),
            )

            # Final fs flush window for some filesystems
            await asyncio.sleep(0)

            size = os.path.getsize(output_path)
            logger.info(
                f"✅ Enhanced processing complete: processed={processed_count} / written={written} frames ({size} bytes) → {output_path}"
            )

            # Sanity: ensure at least one composited frame occurred; if not, still OK as passthrough.
            return output_path

        except Exception as e:
            logger.error(f"Enhanced video processing failed: {e}")
            return None
