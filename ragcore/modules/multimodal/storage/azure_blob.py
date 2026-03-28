"""Azure Blob Storage backend."""

import logging
from typing import Optional

from ragcore.modules.multimodal.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class AzureBlobStorage(StorageBackend):
    """Store files in Azure Blob Storage.

    Requires azure-storage-blob and connection string/credentials.
    """

    def __init__(self, connection_string: str, container_name: str, prefix: str = "multimodal/"):
        """Initialize Azure Blob storage.

        Args:
            connection_string: Azure storage connection string
            container_name: Blob container name
            prefix: Blob prefix for all files (e.g., multimodal/)
        """
        self.connection_string = connection_string
        self.container_name = container_name
        self.prefix = prefix

        try:
            from azure.storage.blob import BlobServiceClient

            self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            self.container_client = self.blob_service_client.get_container_client(container_name)
            logger.info(f"AzureBlobStorage initialized: {container_name}/{prefix}")
        except ImportError:
            logger.error("azure-storage-blob not installed. Install with: pip install azure-storage-blob")
            raise ImportError("AzureBlobStorage requires azure-storage-blob")
        except Exception as e:
            logger.error(f"Failed to initialize Azure Blob client: {e}")
            raise

    async def save_file(self, file_id: str, content: bytes) -> str:
        """Save file to Azure Blob Storage.

        Args:
            file_id: File identifier
            content: File bytes

        Returns:
            Blob URI (e.g., https://{account}.blob.core.windows.net/{container}/{prefix}/{file_id}.bin)
        """
        try:
            blob_name = f"{self.prefix}{file_id}.bin"

            # Upload to blob
            blob_client = self.container_client.get_blob_client(blob_name)
            blob_client.upload_blob(content, overwrite=True)

            blob_uri = blob_client.url
            logger.debug(f"Saved to Azure Blob: {blob_uri}")
            return blob_uri

        except Exception as e:
            logger.error(f"Failed to save file to Azure Blob {file_id}: {e}")
            raise IOError(f"Azure Blob storage save failed: {e}")

    async def get_file(self, storage_path: str) -> Optional[bytes]:
        """Retrieve file from Azure Blob Storage.

        Args:
            storage_path: Blob URI or name

        Returns:
            File bytes or None if not found
        """
        try:
            # Parse blob name from URI if needed
            if storage_path.startswith("https://"):
                # Extract blob name from URI
                # Format: https://{account}.blob.core.windows.net/{container}/{blob_name}
                parts = storage_path.split("/", 3)
                if len(parts) >= 4:
                    blob_name = parts[3]
                else:
                    raise ValueError(f"Invalid blob URI: {storage_path}")
            else:
                blob_name = storage_path

            # Retrieve from blob
            blob_client = self.container_client.get_blob_client(blob_name)
            return blob_client.download_blob().readall()

        except Exception as e:
            if "BlobNotFound" in str(type(e).__name__):
                logger.warning(f"Blob not found: {storage_path}")
                return None
            logger.error(f"Failed to retrieve blob {storage_path}: {e}")
            raise IOError(f"Azure Blob storage retrieval failed: {e}")

    async def delete_file(self, storage_path: str) -> bool:
        """Delete file from Azure Blob Storage.

        Args:
            storage_path: Blob URI or name

        Returns:
            True if deleted, False if not found
        """
        try:
            # Parse blob name from URI if needed
            if storage_path.startswith("https://"):
                parts = storage_path.split("/", 3)
                if len(parts) >= 4:
                    blob_name = parts[3]
                else:
                    raise ValueError(f"Invalid blob URI: {storage_path}")
            else:
                blob_name = storage_path

            blob_client = self.container_client.get_blob_client(blob_name)

            # Check if exists
            try:
                blob_client.get_blob_properties()
            except Exception as e:
                if "BlobNotFound" in str(type(e).__name__):
                    logger.warning(f"Blob not found for deletion: {storage_path}")
                    return False
                raise

            # Delete blob
            blob_client.delete_blob()
            logger.info(f"Deleted blob: {storage_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete blob {storage_path}: {e}")
            raise IOError(f"Azure Blob storage deletion failed: {e}")

    async def exists(self, storage_path: str) -> bool:
        """Check if blob exists in Azure Blob Storage.

        Args:
            storage_path: Blob URI or name

        Returns:
            True if blob exists
        """
        try:
            if storage_path.startswith("https://"):
                parts = storage_path.split("/", 3)
                if len(parts) >= 4:
                    blob_name = parts[3]
                else:
                    return False
            else:
                blob_name = storage_path

            blob_client = self.container_client.get_blob_client(blob_name)
            blob_client.get_blob_properties()
            return True

        except Exception as e:
            if "BlobNotFound" in str(type(e).__name__):
                return False
            logger.error(f"Failed to check blob existence {storage_path}: {e}")
            return False

    def get_backend_name(self) -> str:
        """Get backend name."""
        return "azure_blob"

    async def health_check(self) -> bool:
        """Check if container is accessible."""
        try:
            self.container_client.get_container_properties()
            return True
        except Exception as e:
            logger.error(f"Azure Blob health check failed: {e}")
            return False
