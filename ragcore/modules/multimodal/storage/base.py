"""Abstract storage backend interface."""

from abc import ABC, abstractmethod
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract interface for file storage backends.

    Implementations can store files in S3, Azure Blob, local filesystem, etc.
    """

    @abstractmethod
    async def save_file(self, file_id: str, content: bytes) -> str:
        """Save file to storage.

        Args:
            file_id: Unique file identifier (usually UUID as string)
            content: Raw file bytes

        Returns:
            Storage path/URI (e.g., s3://bucket/uuid.bin, file:///data/uuid.bin)

        Raises:
            IOError: If save fails (permission, disk full, network error)
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    async def get_file(self, storage_path: str) -> Optional[bytes]:
        """Retrieve file from storage.

        Args:
            storage_path: Storage path/URI returned by save_file()

        Returns:
            File bytes, or None if not found

        Raises:
            IOError: If retrieval fails
        """
        pass

    @abstractmethod
    async def delete_file(self, storage_path: str) -> bool:
        """Delete file from storage.

        Args:
            storage_path: Storage path/URI to delete

        Returns:
            True if deleted, False if not found

        Raises:
            IOError: If deletion fails (permission denied, etc)
        """
        pass

    @abstractmethod
    async def exists(self, storage_path: str) -> bool:
        """Check if file exists at storage path.

        Args:
            storage_path: Storage path/URI to check

        Returns:
            True if file exists, False otherwise
        """
        pass

    @abstractmethod
    def get_backend_name(self) -> str:
        """Get human-readable backend name (e.g., 's3', 'local', 'azure_blob').

        Returns:
            Backend name
        """
        pass

    async def health_check(self) -> bool:
        """Check if backend is healthy and accessible.

        Returns:
            True if backend is operational
        """
        return True
