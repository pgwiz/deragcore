"""Video scene detection and smart chunking.

Detects scene/shot boundaries in video and extracts representative keyframes.
Uses PySceneDetect for accurate scene boundary detection.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class VideoSceneChunker:
    """Chunk video by detecting scene/shot boundaries.

    Uses multiple detection strategies:
    1. Content-aware detection (pixel difference-based)
    2. Color histogram changes
    3. Manual keyframe detection
    """

    def __init__(
        self,
        detection_method: str = "content",
        threshold: float = 27.0,
        min_scene_length_s: float = 1.0,
        keyframes_per_scene: int = 2,
    ):
        """Initialize video scene chunker.

        Args:
            detection_method: "content" (pixel-based), "histogram" (color-based), or "manual"
            threshold: Sensitivity threshold for scene detection (higher = fewer scenes)
            min_scene_length_s: Minimum scene duration in seconds
            keyframes_per_scene: Number of keyframes to extract per scene (1-3)
        """
        self.detection_method = detection_method
        self.threshold = threshold
        self.min_scene_length_s = min_scene_length_s
        self.keyframes_per_scene = max(1, min(3, keyframes_per_scene))

    async def chunk_by_scenes(
        self,
        video_path: str,
        transcript: Optional[str] = None,
        narration_map: Optional[Dict[float, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Chunk video by detected scene boundaries.

        Args:
            video_path: Path to video file
            transcript: Optional full transcript for reference
            narration_map: Optional mapping of timestamps to narration text

        Returns:
            List of chunks: [{start_sec, end_sec, keyframe_indices, narrative, scene_id}, ...]
        """
        if not video_path:
            logger.warning("No video path provided")
            return []

        try:
            logger.debug(f"Analyzing video: {video_path}")

            # Detect scene boundaries
            scene_boundaries = await self._detect_scene_boundaries(video_path)

            if not scene_boundaries:
                logger.warning("No scene boundaries detected, using default chunking")
                return await self._chunk_by_duration(video_path)

            # Extract keyframes for each scene
            chunks = await self._create_scene_chunks(
                video_path,
                scene_boundaries,
                narration_map,
            )

            logger.info(f"Created {len(chunks)} scene chunks from {len(scene_boundaries)} boundaries")
            return chunks

        except ImportError:
            logger.error(
                "scenedetect not installed. Install with: pip install scenedetect"
            )
            raise
        except Exception as e:
            logger.error(f"Error analyzing video: {e}")
            raise

    async def _detect_scene_boundaries(
        self,
        video_path: str,
    ) -> List[Tuple[float, float]]:
        """Detect scene/shot boundaries in video.

        Returns list of (start_sec, end_sec) tuples for each scene.
        """
        try:
            from scenedetect import detect, AdaptiveDetector, ContentDetector
            import cv2

            logger.debug(f"Detecting scenes with method: {self.detection_method}")

            if self.detection_method == "content":
                detector = ContentDetector(threshold=self.threshold)
            elif self.detection_method == "histogram":
                detector = ContentDetector(threshold=self.threshold)
            else:
                # Manual/default
                detector = AdaptiveDetector(threshold=self.threshold)

            # Detect scenes
            scenes = detect(video_path, detector)

            if not scenes:
                logger.warning("SceneDetect found no scenes, analyzing frame by frame")
                return await self._manual_scene_detection(video_path)

            # Convert scenes to boundaries
            boundaries = []
            for i, scene in enumerate(scenes):
                start_time = scene[0].get_seconds()
                end_time = scene[1].get_seconds() if i + 1 < len(scenes) else float('inf')
                boundaries.append((start_time, end_time))

            logger.debug(f"Detected {len(boundaries)} scenes")
            return boundaries

        except ImportError:
            logger.error("scenedetect library not installed")
            raise

    async def _manual_scene_detection(
        self,
        video_path: str,
    ) -> List[Tuple[float, float]]:
        """Fallback: manual scene detection via frame comparison.

        Uses frame-by-frame analysis when scenedetect is unavailable.
        """
        try:
            import cv2

            logger.debug("Running manual frame-by-frame scene detection")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return []

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.debug(f"Video: {total_frames} frames at {fps} FPS")

            frame_interval = max(1, int(fps * self.min_scene_length_s))
            boundaries = []
            prev_frame = None
            scene_start_frame = 0

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Check at intervals
                if frame_idx % frame_interval == 0:
                    if prev_frame is not None:
                        # Compute frame difference (simple approach)
                        diff = cv2.absdiff(frame, prev_frame)
                        mean_diff = diff.mean()

                        # If difference exceeds threshold, scene boundary
                        if mean_diff > (self.threshold * 2):  # Adjusted threshold
                            scene_start_sec = (scene_start_frame / fps)
                            scene_end_sec = (frame_idx / fps)
                            boundaries.append((scene_start_sec, scene_end_sec))
                            scene_start_frame = frame_idx
                            logger.debug(f"Scene boundary at {scene_end_sec:.1f}s (diff: {mean_diff:.1f})")

                    prev_frame = frame.copy()

                frame_idx += 1

            # Add final scene
            if scene_start_frame < total_frames:
                final_duration_sec = total_frames / fps
                boundaries.append((scene_start_frame / fps, final_duration_sec))

            cap.release()
            logger.info(f"Manual detection found {len(boundaries)} scenes")
            return boundaries

        except ImportError:
            logger.error("cv2 (OpenCV) not installed. Install with: pip install opencv-python")
            raise

    async def _create_scene_chunks(
        self,
        video_path: str,
        scene_boundaries: List[Tuple[float, float]],
        narration_map: Optional[Dict[float, str]],
    ) -> List[Dict[str, Any]]:
        """Create chunks from scene boundaries with keyframes.

        Args:
            video_path: Path to video
            scene_boundaries: List of (start_sec, end_sec) tuples
            narration_map: Optional narration text per timestamp

        Returns:
            List of scene chunks with keyframes
        """
        try:
            import cv2

            logger.debug(f"Extracting keyframes from {len(scene_boundaries)} scenes")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return []

            fps = cap.get(cv2.CAP_PROP_FPS)
            chunks = []

            for scene_idx, (start_sec, end_sec) in enumerate(scene_boundaries):
                # Extract keyframe indices for this scene
                keyframe_indices = self._select_keyframes(
                    start_sec,
                    end_sec,
                    fps,
                    self.keyframes_per_scene,
                )

                # Get narration for this scene (if available)
                narrative = ""
                if narration_map:
                    # Find narration that overlaps with this scene
                    scene_narrations = [
                        text for time, text in narration_map.items()
                        if start_sec <= time <= end_sec
                    ]
                    if scene_narrations:
                        narrative = " ".join(scene_narrations)

                chunks.append({
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "scene_id": scene_idx,
                    "keyframe_indices": keyframe_indices,
                    "narrative": narrative,
                    "confidence": 0.88,  # Moderate confidence from scene detection
                    "boundary_type": "scene",
                    "duration_sec": end_sec - start_sec,
                })

            cap.release()
            logger.info(f"Created {len(chunks)} chunks with keyframes")
            return chunks

        except ImportError:
            logger.error("cv2 (OpenCV) not installed")
            raise

    def _select_keyframes(
        self,
        start_sec: float,
        end_sec: float,
        fps: float,
        num_keyframes: int,
    ) -> List[int]:
        """Select representative keyframe indices for a scene.

        Uses uniform distribution with preference for the middle and edges.

        Args:
            start_sec: Scene start time in seconds
            end_sec: Scene end time in seconds
            fps: Video framerate
            num_keyframes: Number of keyframes to select

        Returns:
            List of frame indices
        """
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        total_frames = end_frame - start_frame

        if total_frames <= 0:
            return [start_frame]

        if num_keyframes >= total_frames:
            # All frames are keyframes
            return list(range(start_frame, end_frame))

        # Select keyframes using weighted distribution
        keyframes = []

        if num_keyframes == 1:
            # Middle frame
            keyframes = [start_frame + total_frames // 2]
        elif num_keyframes == 2:
            # Start and end
            keyframes = [start_frame, end_frame - 1]
        else:  # num_keyframes >= 3
            # Beginning, middle, end
            keyframes = [
                start_frame,  # First frame
                start_frame + total_frames // 2,  # Middle frame
                end_frame - 1,  # Last frame
            ]

        return sorted(list(set(keyframes)))

    async def _chunk_by_duration(
        self,
        video_path: str,
        chunk_duration_s: float = 10.0,
    ) -> List[Dict[str, Any]]:
        """Fallback: chunk video by fixed duration.

        Used when scene detection fails or is unavailable.
        """
        try:
            import cv2

            logger.debug(f"Chunking video by {chunk_duration_s}s duration (fallback)")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return []

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_duration_s = total_frames / fps

            cap.release()

            chunks = []
            time_sec = 0.0
            chunk_idx = 0

            while time_sec < total_duration_s:
                end_sec = min(time_sec + chunk_duration_s, total_duration_s)

                chunks.append({
                    "start_sec": time_sec,
                    "end_sec": end_sec,
                    "scene_id": chunk_idx,
                    "keyframe_indices": [int((time_sec + (end_sec - time_sec) / 2) * fps)],
                    "narrative": "",
                    "confidence": 0.6,  # Low confidence for fallback
                    "boundary_type": "duration",
                    "duration_sec": end_sec - time_sec,
                })

                time_sec = end_sec
                chunk_idx += 1

            logger.info(f"Created {len(chunks)} duration-based chunks")
            return chunks

        except ImportError:
            logger.error("cv2 (OpenCV) not installed")
            raise

    def estimate_chunk_count(
        self,
        video_duration_s: float,
        avg_scene_duration_s: float = 5.0,
    ) -> int:
        """Estimate number of chunks for given video duration.

        Args:
            video_duration_s: Total video duration in seconds
            avg_scene_duration_s: Average scene duration (typically 5-10s)

        Returns:
            Estimated chunk count
        """
        estimated_chunks = max(1, int(video_duration_s / avg_scene_duration_s))
        return estimated_chunks
