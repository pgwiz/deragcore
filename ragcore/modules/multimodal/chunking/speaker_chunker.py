"""Audio chunking with speaker diarization.

Uses pyannote.audio for speaker identification and segmentation.
Requires Hugging Face token for model access.
"""

import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class SpeakerDiarizationChunker:
    """Chunk audio by speaker boundaries using speaker diarization.

    Identifies who is speaking when in the audio and chunks accordingly.
    Requires: pyannote.audio (pip install pyannote.audio)
    """

    def __init__(self, huggingface_token: Optional[str] = None):
        """Initialize speaker diarization chunker.

        Args:
            huggingface_token: Optional Hugging Face API token for model access
                             If not provided, will attempt to use .cache/huggingface
        """
        self.huggingface_token = huggingface_token
        self.pipeline = None
        self._initialized = False

    async def chunk_by_speakers(
        self,
        transcript: str,
        audio_path: str,
        timestamps: Optional[List[tuple]] = None,
    ) -> List[Dict[str, Any]]:
        """Chunk transcript by speaker changes.

        Args:
            transcript: Full transcript text
            audio_path: Path to audio file
            timestamps: Optional word-level timestamps for alignment

        Returns:
            List of chunks: [{start_sec, end_sec, speaker_id, content, confidence}, ...]
        """
        if not transcript or not transcript.strip():
            logger.warning("Empty transcript provided")
            return []

        try:
            await self._ensure_initialized()

            logger.debug(f"Running speaker diarization on: {audio_path}")

            # Run diarization pipeline
            diarization = self.pipeline(audio_path)

            # Extract speaker segments
            segments = self._extract_speaker_segments(diarization)
            logger.debug(f"Detected {len(segments)} speaker segments")

            # Map transcript to speaker segments
            chunks = self._map_transcript_to_speakers(
                transcript,
                segments,
                timestamps
            )

            logger.info(f"Created {len(chunks)} chunks from {len(set(s['speaker'] for s in segments))} speakers")
            return chunks

        except ImportError:
            logger.error(
                "pyannote.audio not installed. Install with: pip install pyannote-audio"
            )
            raise
        except Exception as e:
            logger.error(f"Error in speaker diarization: {e}")
            raise

    def _extract_speaker_segments(self, diarization) -> List[Dict[str, Any]]:
        """Extract speaker segments from diarization output.

        Args:
            diarization: Output from pyannote pipeline

        Returns:
            List of segments: [{start_sec, end_sec, speaker}, ...]
        """
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start_sec": turn.start,
                "end_sec": turn.end,
                "speaker": speaker,
                "duration_sec": turn.end - turn.start,
            })
        return segments

    def _map_transcript_to_speakers(
        self,
        transcript: str,
        segments: List[Dict[str, Any]],
        timestamps: Optional[List[tuple]] = None,
    ) -> List[Dict[str, Any]]:
        """Map transcript words to speaker segments.

        Groups words by speaker to create logical chunks.

        Args:
            transcript: Full transcript
            segments: Speaker segments from diarization
            timestamps: Optional word-level timestamps

        Returns:
            Chunks with speaker ID and content
        """
        words = transcript.split()
        if not timestamps:
            # Without timestamps, distribute words uniformly across segments
            total_duration = segments[-1]["end_sec"] if segments else 1.0
            words_per_sec = len(words) / max(total_duration, 1.0)

        chunks = []

        for segment in segments:
            start_sec = segment["start_sec"]
            end_sec = segment["end_sec"]
            speaker = segment["speaker"]

            # Extract words for this speaker's segment
            if timestamps:
                # Use timestamps for accurate word mapping
                segment_words = [
                    words[i] for i, (w_start, w_end) in enumerate(timestamps)
                    if w_start >= start_sec and w_end <= end_sec
                ]
            else:
                # Estimate word positions
                start_word_idx = int(start_sec * words_per_sec)
                end_word_idx = int(end_sec * words_per_sec)
                segment_words = words[start_word_idx:end_word_idx]

            if not segment_words:
                continue

            content = " ".join(segment_words)

            chunks.append({
                "start_sec": start_sec,
                "end_sec": end_sec,
                "speaker": speaker,
                "content": content,
                "confidence": 0.90,  # High confidence from ML model
                "boundary_type": "speaker_change",
                "duration_sec": end_sec - start_sec,
            })

        logger.debug(f"Mapped {len(words)} words to {len(chunks)} speaker segments")
        return chunks

    async def _ensure_initialized(self):
        """Lazily initialize the diarization pipeline.

        Loads the pretrained model on first use.
        """
        if self._initialized:
            return

        try:
            from pyannote.audio import Pipeline

            logger.debug("Initializing speaker diarization pipeline")

            # Load pretrained diarization pipeline
            # Note: Requires Hugging Face account and token
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.0",
                use_auth_token=self.huggingface_token,
            )

            self._initialized = True
            logger.info("Speaker diarization pipeline loaded successfully")

        except ImportError:
            logger.error(
                "pyannote.audio not installed. Install with: pip install pyannote-audio"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize speaker diarization pipeline: {e}")
            raise

    def get_speaker_count(self, segments: List[Dict[str, Any]]) -> int:
        """Get count of unique speakers in segments.

        Args:
            segments: List of speaker segments

        Returns:
            Number of unique speakers
        """
        speakers = set(s["speaker"] for s in segments)
        return len(speakers)

    def merge_adjacent_speakers(
        self,
        chunks: List[Dict[str, Any]],
        merge_silence_threshold_s: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Merge chunks from same speaker if separated by brief silence.

        Args:
            chunks: Chunks from diarization
            merge_silence_threshold_s: Max gap to merge (seconds)

        Returns:
            Merged chunks
        """
        if not chunks:
            return []

        merged = []
        current_chunk = chunks[0].copy()

        for next_chunk in chunks[1:]:
            gap = next_chunk["start_sec"] - current_chunk["end_sec"]

            # Merge if same speaker and gap is small
            if (next_chunk["speaker"] == current_chunk["speaker"] and
                gap <= merge_silence_threshold_s):

                # Merge content
                current_chunk["content"] += " " + next_chunk["content"]
                current_chunk["end_sec"] = next_chunk["end_sec"]
                current_chunk["duration_sec"] = (
                    current_chunk["end_sec"] - current_chunk["start_sec"]
                )

            else:
                # Different speaker or large gap - keep current, start new
                merged.append(current_chunk)
                current_chunk = next_chunk.copy()

        merged.append(current_chunk)

        logger.debug(f"Merged chunks: {len(chunks)} → {len(merged)}")
        return merged
