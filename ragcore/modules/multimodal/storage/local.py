"""Local filesystem storage backend."""

import os
import logging
from pathlib import Path
from typing import Optional

from ragcore.modules.multimodal.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class LocalStorage(StorageBackend):
    """Store files on local filesystem.

    Simple implementation for development/testing. Production should use S3/Blob.
    """

    def __init__(self, base_path: str = "/data/multimodal"):
        """Initialize local storage.

        Args:
            base_path: Root directory for file storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"LocalStorage initialized: {self.base_path}")

    async def save_file(self, file_id: str, content: bytes) -> str:
        """Save file to local filesystem.

        Args:
            file_id: File identifier
            content: File bytes

        Returns:
            File path (e.g., file:///data/multimodal/uuid.bin)
        """
        try:
            file_path = self.base_path / f"{file_id}.bin"

            # Write file
            file_path.write_bytes(content)

            # Return URI
            return f"file://{file_path.absolute()}"

        except Exception as e:
            logger.error(f"Failed to save file {file_id}: {e}")
            raise IOError(f"Local storage save failed: {e}")

    async def get_file(self, storage_path: str) -> Optional[bytes]:
        """Retrieve file from local filesystem.

        Args:
            storage_path: File path (from save_file)

        Returns:
            File bytes or None if not found
        """
        try:
            # Parse file:// URI
            if storage_path.startswith("file://"):
                file_path = Path(storage_path[7:])  # Remove 'file://'
            else:
                file_path = Path(storage_path)

            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                return None

            return file_path.read_bytes()

        except Exception as e:
            logger.error(f"Failed to retrieve file {storage_path}: {e}")
            raise IOError(f"Local storage retrieval failed: {e}")

    async def delete_file(self, storage_path: str) -> bool:
        """Delete file from local filesystem.

        Args:
            storage_path: File path to delete

        Returns:
            True if deleted, False if not found
        """
        try:
            if storage_path.startswith("file://"):
                file_path = Path(storage_path[7:])
            else:
                file_path = Path(storage_path)

            if not file_path.exists():
                logger.warning(f"File not found for deletion: {file_path}")
                return False

            file_path.unlink()  # Delete file
            logger.info(f"Deleted file: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete file {storage_path}: {e}")
            raise IOError(f"Local storage deletion failed: {e}")

    async def exists(self, storage_path: str) -> bool:
        """Check if file exists locally.

        Args:
            storage_path: File path to check

        Returns:
            True if file exists
        """
        try:
            if storage_path.startswith("file://"):
                file_path = Path(storage_path[7:])
            else:
                file_path = Path(storage_path)

            return file_path.exists()

        except Exception as e:
            logger.error(f"Failed to check file existence {storage_path}: {e}")
            return False

    def get_backend_name(self) -> str:
        """Get backend name."""
        return "local"

    async def health_check(self) -> bool:
        """Check if base path is writable."""
        try:
            test_file = self.base_path / ".health_check"
            test_file.write_text("ok")
            test_file.unlink()
            return True
        except Exception as e:
            logger.error(f"Local storage health check failed: {e}")
            return False
