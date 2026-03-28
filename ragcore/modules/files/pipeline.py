"""File processing pipeline - Orchestrates parse → chunk → embed → store workflow."""

import logging
from typing import Optional
from uuid import UUID
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ragcore.config import settings
from ragcore.db.database import get_db_session
from ragcore.models import File, Chunk
from ragcore.core.ai_controller import AIController
from ragcore.modules.files.parser import FileParser
from ragcore.modules.files.chunker import TextChunker

logger = logging.getLogger(__name__)


class FileProcessingPipeline:
    """Orchestrates complete file processing: parse → chunk → embed → store."""

    def __init__(self):
        """Initialize pipeline with configured chunker and embedding settings."""
        self.chunker = TextChunker(
            chunk_size_tokens=settings.chunk_size_tokens,
            chunk_overlap_tokens=settings.chunk_overlap_tokens,
        )
        self.embedding_batch_size = settings.embedding_batch_size
        self.embedding_provider = settings.embedding_provider
        self.embedding_model = settings.embedding_model

    async def process(
        self,
        file_id: UUID,
        file_bytes: bytes,
        content_type: str,
        session_id: Optional[UUID] = None,
    ) -> None:
        """
        Process uploaded file through complete pipeline.

        Stages:
        1. parsing: Extract text from PDF/DOCX
        2. chunking: Split text into overlapping chunks
        3. embedding: Generate and store vectors for each chunk
        4. ready: Mark file as complete

        On error at any stage, sets status='failed' and stores error message.

        Args:
            file_id: UUID of File record to process
            file_bytes: Raw file content
            content_type: MIME type (application/pdf or application/vnd.openxmlformats-...)
            session_id: Optional session scope
        """
        async with get_db_session() as session:
            try:
                # Fetch File record
                file_record = await session.get(File, file_id)
                if not file_record:
                    logger.error(f"File {file_id} not found in database")
                    return

                # =====================================================================
                # Stage 1: PARSING
                # =====================================================================
                logger.info(f"PARSING: Starting parse for {file_record.filename}")
                await self._update_file_status(
                    session, file_id, "parsing", error_message=None
                )

                try:
                    text, parse_metadata = FileParser.parse(file_bytes, content_type)
                    logger.info(
                        f"PARSING: Extracted {len(text)} chars from {file_record.filename}"
                    )
                except ValueError as e:
                    error_msg = f"Parse error: {str(e)}"
                    logger.error(f"PARSING: {error_msg}")
                    await self._update_file_status(session, file_id, "failed", error_msg)
                    return
                except Exception as e:
                    error_msg = f"Unexpected parse error: {str(e)}"
                    logger.error(f"PARSING: {error_msg}")
                    await self._update_file_status(session, file_id, "failed", error_msg)
                    return

                # =====================================================================
                # Stage 2: CHUNKING
                # =====================================================================
                logger.info(f"CHUNKING: Starting chunking for {file_record.filename}")
                await self._update_file_status(session, file_id, "chunking")

                try:
                    # Merge parse metadata into chunk metadata
                    chunk_metadata = {
                        "source": file_record.filename,
                        "file_type": parse_metadata.get("file_type"),
                        "parse_metadata": parse_metadata,
                    }

                    chunks = self.chunker.chunk(text, metadata=chunk_metadata)

                    if not chunks:
                        error_msg = "No chunks created after splitting"
                        logger.error(f"CHUNKING: {error_msg}")
                        await self._update_file_status(session, file_id, "failed", error_msg)
                        return

                    logger.info(f"CHUNKING: Created {len(chunks)} chunks")
                except Exception as e:
                    error_msg = f"Chunking error: {str(e)}"
                    logger.error(f"CHUNKING: {error_msg}")
                    await self._update_file_status(session, file_id, "failed", error_msg)
                    return

                # =====================================================================
                # Stage 3: EMBEDDING
                # =====================================================================
                logger.info(f"EMBEDDING: Starting embedding for {len(chunks)} chunks")
                await self._update_file_status(session, file_id, "embedding")

                try:
                    chunk_records = []

                    # Process in batches
                    for batch_idx in range(0, len(chunks), self.embedding_batch_size):
                        batch = chunks[
                            batch_idx : batch_idx + self.embedding_batch_size
                        ]
                        logger.debug(
                            f"EMBEDDING: Batch {batch_idx // self.embedding_batch_size + 1} "
                            f"({len(batch)} chunks)"
                        )

                        for chunk_data in batch:
                            try:
                                # Generate embedding for this chunk
                                embedding_vector = AIController.embed(
                                    provider_name=self.embedding_provider,
                                    model_id=self.embedding_model,
                                    text=chunk_data["text"],
                                )

                                # Create Chunk ORM object
                                chunk_record = Chunk(
                                    file_id=file_id,
                                    chunk_index=chunk_data["chunk_index"],
                                    text=chunk_data["text"],
                                    embedding=embedding_vector,
                                    tokens=chunk_data["tokens"],
                                    metadata_=chunk_data.get("metadata", {}),
                                )
                                chunk_records.append(chunk_record)

                            except Exception as e:
                                logger.error(
                                    f"EMBEDDING: Failed to embed chunk "
                                    f"{chunk_data['chunk_index']}: {str(e)}"
                                )
                                raise

                    logger.info(
                        f"EMBEDDING: Successfully embedded {len(chunk_records)} chunks"
                    )

                except Exception as e:
                    error_msg = f"Embedding error: {str(e)}"
                    logger.error(f"EMBEDDING: {error_msg}")
                    await self._update_file_status(session, file_id, "failed", error_msg)
                    return

                # =====================================================================
                # Stage 4: STORAGE
                # =====================================================================
                logger.info(f"STORAGE: Storing {len(chunk_records)} chunks to database")

                try:
                    # Add all chunk records to session
                    for chunk_record in chunk_records:
                        session.add(chunk_record)

                    # Update File with final status
                    file_record.status = "ready"
                    file_record.chunks_count = len(chunk_records)
                    file_record.error_message = None
                    file_record.updated_at = datetime.utcnow()

                    # Commit all changes
                    await session.commit()
                    logger.info(
                        f"STORAGE: File {file_id} ready - {len(chunk_records)} chunks stored"
                    )

                except Exception as e:
                    await session.rollback()
                    error_msg = f"Storage error: {str(e)}"
                    logger.error(f"STORAGE: {error_msg}")

                    # Try to update status without transaction
                    async with get_db_session() as rollback_session:
                        await self._update_file_status(
                            rollback_session, file_id, "failed", error_msg
                        )
                    return

            except Exception as e:
                # Fall-safe: Catch any unhandled exceptions and mark as failed
                error_msg = f"Unexpected pipeline error: {str(e)}"
                logger.error(f"PIPELINE: {error_msg}", exc_info=True)

                try:
                    async with get_db_session() as error_session:
                        await self._update_file_status(
                            error_session, file_id, "failed", error_msg
                        )
                except Exception as db_error:
                    logger.error(
                        f"PIPELINE: Failed to update error status: {str(db_error)}"
                    )

    async def _update_file_status(
        self,
        session: AsyncSession,
        file_id: UUID,
        status: str,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Update File.status atomically.

        Args:
            session: Database session
            file_id: File ID to update
            status: New status string
            error_message: Optional error description (set if status='failed')
        """
        stmt = (
            update(File)
            .where(File.id == file_id)
            .values(
                status=status,
                error_message=error_message,
                updated_at=datetime.utcnow(),
            )
        )
        await session.execute(stmt)
        await session.commit()
        logger.debug(f"File {file_id} status updated to '{status}'")
