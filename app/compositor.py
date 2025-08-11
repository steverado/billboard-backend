import cv2
import numpy as np
import json
import time
import os
import asyncio
from typing import Dict, List, Tuple, Optional, Callable
from PIL import Image
from scipy.ndimage import gaussian_filter
import logging

logger = logging.getLogger(__name__)


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

        for frame_data in tracking_frames:
            frame_num = frame_data["frame"]
            corners = np.array(frame_data["corners"], dtype=np.float32)
            self.frame_lookup[frame_num] = corners

        logger.info(f"Loaded {len(tracking_frames)} tracking frames")

    # ---------------------------
    # Compositing helpers
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
        max_distance = float(np.max(corner_distances))
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

    def match_scene_lighting(
        self, overlay: np.ndarray, background_region: np.ndarray
    ) -> np.ndarray:
        """Adjust overlay lighting and color to match the background scene."""
        if self.color_matching_strength == 0 or background_region.size == 0:
            return overlay

        try:
            bg_mean = np.mean(background_region, axis=(0, 1))
            bg_std = np.std(background_region, axis=(0, 1))

            overlay_mean = np.mean(overlay, axis=(0, 1))
            overlay_std = np.maximum(np.std(overlay, axis=(0, 1)), 1.0)

            corrected = overlay.astype(np.float32)

            channels = overlay.shape[2] if overlay.ndim == 3 else 1
            for c in range(min(3, channels)):
                if overlay.ndim == 3:
                    channel = corrected[:, :, c]
                    target_mean = (
                        bg_mean[c] if c < len(bg_mean) else float(np.mean(bg_mean))
                    )
                    target_std = (
                        bg_std[c] if c < len(bg_std) else float(np.mean(bg_std))
                    )
                else:
                    channel = corrected
                    target_mean = float(np.mean(bg_mean))
                    target_std = float(np.mean(bg_std))

                adjustment_factor = self.color_matching_strength

                normalized = (channel - overlay_mean[c]) / overlay_std[c]
                adjusted = (
                    normalized * target_std * (1 - adjustment_factor)
                    + normalized * overlay_std[c] * adjustment_factor
                )
                adjusted = (
                    adjusted
                    + target_mean * adjustment_factor
                    + overlay_mean[c] * (1 - adjustment_factor)
                )

                if overlay.ndim == 3:
                    corrected[:, :, c] = adjusted
                else:
                    corrected = adjusted

            return np.clip(corrected, 0, 255).astype(np.uint8)

        except Exception as e:
            logger.warning(f"Color matching failed: {e}")
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
        """Process video with enhanced quality, ensure file output is complete, and return the path."""
        cap = None
        out = None
        try:
            if not os.path.exists(video_file):
                logger.error(f"❌ Template video does not exist: {video_file}")
                return None

            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                logger.error(f"❌ Could not open video file: {video_file}")
                return None

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # fallback if FPS is 0

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            processed_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or max_frames)

            for frame_idx in range(min(total_frames, max_frames)):
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.info(f"End of video or failed to read frame {frame_idx}")
                    break

                if frame_idx in self.frame_lookup:
                    corners = self.frame_lookup[frame_idx]
                    result_frame = self.apply_enhanced_composite(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        overlay,
                        corners,
                        frame_idx,
                    )
                    result_frame = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
                    processed_count += 1
                else:
                    result_frame = frame

                out.write(result_frame)

                # Update progress
                if progress_callback and (frame_idx + 1) % 10 == 0:
                    progress = int(40 + (frame_idx / max_frames) * 50)  # 40-90%
                    progress_callback(progress)

                if frame_idx % 30 == 0:
                    await asyncio.sleep(0)

            # Ensure containers are finalized
            if out:
                out.release()
            if cap:
                cap.release()

            # Small delay to ensure fs flush before size check
            time.sleep(0.2)

            if not os.path.exists(output_path):
                logger.error(f"❌ Output file not found: {output_path}")
                return None

            size = os.path.getsize(output_path)
            if size <= 0:
                logger.error(f"❌ Output file is empty: {output_path}")
                return None

            logger.info(
                f"✅ Enhanced processing complete: {processed_count} frames written ({size} bytes) → {output_path}"
            )
            return output_path if processed_count > 0 else None

        except Exception as e:
            logger.error(f"Enhanced video processing failed: {e}")
            return None

        finally:
            try:
                if cap:
                    cap.release()
            except Exception:
                pass
            try:
                if out:
                    out.release()
            except Exception:
                pass
