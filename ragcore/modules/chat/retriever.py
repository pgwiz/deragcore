"""Vector retrieval - Search chunks by semantic similarity using pgvector."""

import logging
from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from ragcore.db.database import get_db_session
from ragcore.models import Chunk, File
from ragcore.core.ai_controller import AIController
from ragcore.config import settings

logger = logging.getLogger(__name__)


class RetrievedChunk:
    """Result of vector similarity search."""

    def __init__(
        self,
        chunk_id: UUID,
        file_id: UUID,
        filename: str,
        text: str,
        tokens: int,
        similarity_score: float,
        metadata: dict,
    ):
        self.chunk_id = chunk_id
        self.file_id = file_id
        self.filename = filename
        self.text = text
        self.tokens = tokens
        self.similarity_score = similarity_score
        self.metadata = metadata

    def __repr__(self) -> str:
        return (
            f"<RetrievedChunk {self.chunk_id[:8]} - "
            f"score={self.similarity_score:.3f}, "
            f"tokens={self.tokens}>"
        )


class VectorRetriever:
    """Semantic search via pgvector cosine similarity."""

    def __init__(
        self,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """Initialize retriever with embedding config.

        Args:
            embedding_provider: Provider for query embedding (default: config)
            embedding_model: Model for query embedding (default: config)
        """
        self.embedding_provider = embedding_provider or settings.embedding_provider
        self.embedding_model = embedding_model or settings.embedding_model

    async def retrieve(
        self,
        query: str,
        file_ids: Optional[List[UUID]] = None,
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        """
        Search chunks by semantic similarity to query.

        Strategy:
        1. Embed query using AIController.embed()
        2. Search chunks via pgvector cosine similarity (<-> operator)
        3. Return top_k results ordered by score

        Args:
            query: Search query
            file_ids: Optional list of file IDs to scope search
            top_k: Number of results to return

        Returns:
            List of RetrievedChunk ordered by similarity score (descending)

        Raises:
            ValueError: If query is empty
            RuntimeError: If embedding fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        logger.debug(
            f"Retrieve: query='{query[:50]}...', "
            f"files={len(file_ids) if file_ids else 'all'}, "
            f"top_k={top_k}"
        )

        # =====================================================================
        # Embed Query
        # =====================================================================
        try:
            query_embedding = AIController.embed(
                provider_name=self.embedding_provider,
                model_id=self.embedding_model,
                text=query,
            )
            logger.debug(f"Embedded query: {len(query_embedding)} dims")
        except Exception as e:
            error_msg = f"Failed to embed query: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        # =====================================================================
        # Vector Similarity Search
        # =====================================================================
        async with get_db_session() as session:
            try:
                # Build query filters
                filters = [Chunk.embedding != None]  # noqa: E712
                if file_ids:
                    filters.append(Chunk.file_id.in_(file_ids))

                # Calculate similarity score using pgvector cosine distance
                # The <-> operator computes cosine distance (0 = identical, 2 = opposite)
                # Similarity = 1 - distance / 2
                similarity = 1 - (Chunk.embedding.cosine_distance(query_embedding))

                # Build and execute query
                stmt = (
                    select(
                        Chunk.id,
                        Chunk.file_id,
                        File.filename,
                        Chunk.text,
                        Chunk.tokens,
                        Chunk.metadata_,
                        similarity.label("similarity_score"),
                    )
                    .select_from(Chunk)
                    .join(File, Chunk.file_id == File.id)
                    .where(and_(*filters))
                    .order_by(similarity.desc())
                    .limit(top_k)
                )

                result = await session.execute(stmt)
                rows = result.all()

                logger.debug(f"Retrieved {len(rows)} chunks")

                # Convert rows to RetrievedChunk objects
                chunks = [
                    RetrievedChunk(
                        chunk_id=row.id,
                        file_id=row.file_id,
                        filename=row.filename,
                        text=row.text,
                        tokens=row.tokens,
                        similarity_score=max(0.0, float(row.similarity_score)),  # Clamp to 0-1
                        metadata=row.metadata_ or {},
                    )
                    for row in rows
                ]

                logger.info(
                    f"Retrieved {len(chunks)} chunks (top_k={top_k}), "
                    f"best score={chunks[0].similarity_score:.3f if chunks else 0:.3f}"
                )

                return chunks

            except Exception as e:
                logger.error(f"Vector search error: {str(e)}", exc_info=True)
                raise RuntimeError(f"Vector search failed: {str(e)}") from e

    async def retrieve_by_text(
        self,
        query: str,
        file_ids: Optional[List[UUID]] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.5,
    ) -> List[RetrievedChunk]:
        """
        Search chunks with similarity threshold filtering.

        Args:
            query: Search query
            file_ids: Optional file ID filter
            top_k: Max results to return
            similarity_threshold: Only return results above this score (0-1)

        Returns:
            Filtered list of retrieved chunks
        """
        all_results = await self.retrieve(query, file_ids, top_k * 2)  # Fetch more to filter

        filtered = [
            chunk
            for chunk in all_results
            if chunk.similarity_score >= similarity_threshold
        ]

        return filtered[:top_k]  # Return at most top_k

    async def retrieve_by_file(
        self,
        query: str,
        file_id: UUID,
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        """
        Search within a single file.

        Args:
            query: Search query
            file_id: File to search
            top_k: Number of results

        Returns:
            Retrieved chunks from file
        """
        return await self.retrieve(query, file_ids=[file_id], top_k=top_k)
