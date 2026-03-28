"""Audio chunking with silence detection and speaker diarization support."""

import logging
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class AudioSilenceChunker:
    """Chunk audio by detecting silence intervals.

    Uses energy-based silence detection to identify natural breaks
    in audio that correspond to pauses in speech.
    """

    def __init__(
        self,
        energy_percentile: int = 20,
        min_chunk_duration_s: float = 2.0,
        silence_duration_s: float = 0.5,
    ):
        """Initialize audio silence chunker.

        Args:
            energy_percentile: Percentile threshold for silence detection (0-100)
                             Lower values = more aggressive silence detection
                             Default 20 = bottom 20% energy is silence
            min_chunk_duration_s: Minimum chunk duration in seconds
            silence_duration_s: Minimum silence duration to create break
        """
        self.energy_percentile = energy_percentile
        self.min_chunk_duration_s = min_chunk_duration_s
        self.silence_duration_s = silence_duration_s

    async def chunk_by_silence(
        self,
        transcript: str,
        timestamps: Optional[List[Tuple[float, float]]] = None,
        audio_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Chunk transcript by detected silence in audio.

        Args:
            transcript: Full transcript text
            timestamps: Optional word-level timestamps [(start_sec, end_sec), ...]
            audio_path: Optional path to audio file for silence analysis

        Returns:
            List of chunks: [{start_sec, end_sec, content, confidence}, ...]
        """
        if not transcript or not transcript.strip():
            logger.warning("Empty transcript provided")
            return []

        # If we have timestamps, use them for chunking
        if timestamps:
            return self._chunk_by_timestamps(transcript, timestamps)

        # Otherwise, do simple heuristic-based chunking
        # (Would need audio_path for real silence detection)
        if audio_path:
            try:
                return await self._chunk_by_audio_silence(transcript, audio_path)
            except ImportError:
                logger.warning("librosa not available, falling back to text-based chunking")
            except Exception as e:
                logger.error(f"Error analyzing audio: {e}")

        # Fallback: Simple sentence-based chunking
        return self._chunk_by_sentences(transcript)

    def _chunk_by_timestamps(
        self,
        transcript: str,
        timestamps: List[Tuple[float, float]],
    ) -> List[Dict[str, Any]]:
        """Chunk based on word-level timestamps.

        Groups words by speakers or natural pauses.
        """
        logger.debug(f"Chunking {len(timestamps)} words by timestamps")

        chunks = []
        current_chunk_text = []
        current_chunk_start = None
        current_chunk_end = None

        for i, (start_sec, end_sec) in enumerate(timestamps):
            if current_chunk_start is None:
                current_chunk_start = start_sec

            # Check if there's a gap (pause) between this word and next
            next_start = timestamps[i + 1][0] if i + 1 < len(timestamps) else None
            gap = (next_start - end_sec) if next_start else 0

            # Add word to current chunk
            words = transcript.split()
            if i < len(words):
                current_chunk_text.append(words[i])

            current_chunk_end = end_sec

            # Create chunk if gap is large enough or we're at the end
            if gap > self.silence_duration_s or i == len(timestamps) - 1:
                content = " ".join(current_chunk_text)
                if len(content.strip()) > 0:
                    chunks.append({
                        "start_sec": current_chunk_start,
                        "end_sec": current_chunk_end,
                        "content": content,
                        "confidence": 0.95,  # High confidence with timestamps
                        "boundary_type": "silence",
                    })

                current_chunk_text = []
                current_chunk_start = None

        logger.info(f"Created {len(chunks)} chunks from {len(timestamps)} timestamps")
        return chunks

    async def _chunk_by_audio_silence(
        self,
        transcript: str,
        audio_path: str,
    ) -> List[Dict[str, Any]]:
        """Analyze audio file to detect silence and chunk accordingly.

        Args:
            transcript: Full transcript
            audio_path: Path to audio file

        Returns:
            Chunks aligned with silence boundaries
        """
        try:
            import librosa

            logger.debug(f"Analyzing audio file: {audio_path}")

            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)
            logger.debug(f"Loaded audio: {len(y)} samples at {sr}Hz")

            # Compute mel-spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

            # Convert to dB scale
            S_db = librosa.power_to_db(S, ref=np.max)

            # Compute energy per frame
            energy = S_db.mean(axis=0)

            # Find silence threshold (below percentile)
            silence_threshold = np.percentile(energy, self.energy_percentile)
            logger.debug(f"Silence threshold: {silence_threshold:.2f} dB (below {self.energy_percentile}th percentile)")

            # Detect silent frames
            silent_frames = energy < silence_threshold

            # Convert frames to time
            frame_times = librosa.frames_to_time(np.arange(len(silent_frames)), sr=sr)

            # Find silence boundaries (transitions from speech to silence)
            silence_starts = []
            silence_ends = []

            in_silence = silent_frames[0]
            for i, is_silent in enumerate(silent_frames):
                if is_silent and not in_silence:
                    # Silence starts
                    silence_starts.append(frame_times[i])
                elif not is_silent and in_silence:
                    # Silence ends
                    silence_ends.append(frame_times[i])
                in_silence = is_silent

            logger.debug(f"Detected {len(silence_starts)} silence regions")

            # Map transcript to silence boundaries (simple approach: uniform distribution)
            chunks = self._map_transcript_to_silence(
                transcript,
                silence_starts,
                silence_ends,
                y.shape[0] / sr  # Total duration
            )

            return chunks

        except ImportError:
            logger.error("librosa not installed. Install with: pip install librosa")
            raise

    def _map_transcript_to_silence(
        self,
        transcript: str,
        silence_starts: List[float],
        silence_ends: List[float],
        total_duration_s: float,
    ) -> List[Dict[str, Any]]:
        """Map transcript words to silence boundaries.

        Simple heuristic: distribute words uniformly across time,
        then group by silence boundaries.
        """
        logger.debug(f"Mapping {len(transcript.split())} words to {len(silence_starts)} silence regions")

        words = transcript.split()
        if not words:
            return []

        # Estimate word timing (words per second)
        words_per_sec = len(words) / max(total_duration_s, 1.0)

        chunks = []
        current_start_sec = 0.0
        current_words = []

        for silence_start in silence_starts:
            # All words up to this silence point go in one chunk
            words_in_chunk = int(words_per_sec * silence_start)

            if words_in_chunk > len(current_words):
                chunk_words = words[len(current_words):words_in_chunk]
            else:
                chunk_words = []

            if chunk_words:
                content = " ".join(chunk_words)
                chunks.append({
                    "start_sec": current_start_sec,
                    "end_sec": silence_start,
                    "content": content,
                    "confidence": 0.85,  # Moderate confidence with audio analysis
                    "boundary_type": "silence",
                })
                current_words.extend(chunk_words)
                current_start_sec = silence_start

        # Add remaining words as final chunk
        if len(current_words) < len(words):
            remaining = words[len(current_words):]
            if remaining:
                chunks.append({
                    "start_sec": current_start_sec,
                    "end_sec": total_duration_s,
                    "content": " ".join(remaining),
                    "confidence": 0.85,
                    "boundary_type": "end",
                })

        logger.info(f"Created {len(chunks)} chunks from silence analysis")
        return chunks

    def _chunk_by_sentences(self, transcript: str) -> List[Dict[str, Any]]:
        """Fallback: chunk by sentence boundaries.

        Simple approach for when audio analysis is not available.
        """
        logger.debug("Chunking by sentences (fallback method)")

        # Split by periods, question marks, exclamation marks
        import re
        sentences = re.split(r'[.!?]+', transcript)

        chunks = []
        current_time_sec = 0.0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Estimate duration (assume ~150 words per minute = 2.5 words per second)
            word_count = len(sentence.split())
            estimated_duration_s = word_count / 2.5

            chunks.append({
                "start_sec": current_time_sec,
                "end_sec": current_time_sec + estimated_duration_s,
                "content": sentence,
                "confidence": 0.7,  # Lower confidence for heuristic
                "boundary_type": "sentence",
            })

            current_time_sec += estimated_duration_s

        logger.info(f"Created {len(chunks)} sentence-based chunks")
        return chunks

    def estimate_chunk_count(
        self,
        transcript: str,
        audio_duration_s: Optional[float] = None,
    ) -> int:
        """Estimate number of chunks for given transcript.

        Args:
            transcript: Full transcript
            audio_duration_s: Audio duration (optional, for estimation)

        Returns:
            Estimated chunk count
        """
        # Heuristic: ~30-60 seconds per chunk = ~75-150 words per chunk
        word_count = len(transcript.split())
        avg_words_per_chunk = 100
        estimated_chunks = max(1, word_count // avg_words_per_chunk)

        return estimated_chunks
