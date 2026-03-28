"""AWS S3 storage backend."""

import logging
from typing import Optional

from ragcore.modules.multimodal.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class S3Storage(StorageBackend):
    """Store files in AWS S3.

    Requires boto3 and AWS credentials configured.
    """

    def __init__(self, bucket_name: str, region: str = "us-east-1", prefix: str = "multimodal/"):
        """Initialize S3 storage.

        Args:
            bucket_name: S3 bucket name
            region: AWS region
            prefix: Key prefix for all files (e.g., multimodal/)
        """
        self.bucket_name = bucket_name
        self.region = region
        self.prefix = prefix

        try:
            import boto3

            self.s3_client = boto3.client("s3", region_name=region)
            logger.info(f"S3Storage initialized: s3://{bucket_name}/{prefix}")
        except ImportError:
            logger.error("boto3 not installed. Install with: pip install boto3")
            raise ImportError("S3Storage requires boto3")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise

    async def save_file(self, file_id: str, content: bytes) -> str:
        """Save file to S3.

        Args:
            file_id: File identifier (usually UUID)
            content: File bytes

        Returns:
            S3 path (e.g., s3://bucket/multimodal/uuid.bin)
        """
        try:
            key = f"{self.prefix}{file_id}.bin"

            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=content,
            )

            s3_path = f"s3://{self.bucket_name}/{key}"
            logger.debug(f"Saved to S3: {s3_path}")
            return s3_path

        except Exception as e:
            logger.error(f"Failed to save file to S3 {file_id}: {e}")
            raise IOError(f"S3 storage save failed: {e}")

    async def get_file(self, storage_path: str) -> Optional[bytes]:
        """Retrieve file from S3.

        Args:
            storage_path: S3 path

        Returns:
            File bytes or None if not found
        """
        try:
            # Parse s3://bucket/key
            if not storage_path.startswith("s3://"):
                raise ValueError(f"Invalid S3 path: {storage_path}")

            path_parts = storage_path[5:].split("/", 1)  # Remove s3://
            if len(path_parts) != 2:
                raise ValueError(f"Invalid S3 path format: {storage_path}")

            bucket, key = path_parts

            # Retrieve from S3
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            return response["Body"].read()

        except self.s3_client.exceptions.NoSuchKey:
            logger.warning(f"File not found in S3: {storage_path}")
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve file from S3 {storage_path}: {e}")
            raise IOError(f"S3 storage retrieval failed: {e}")

    async def delete_file(self, storage_path: str) -> bool:
        """Delete file from S3.

        Args:
            storage_path: S3 path to delete

        Returns:
            True if deleted, False if not found
        """
        try:
            if not storage_path.startswith("s3://"):
                raise ValueError(f"Invalid S3 path: {storage_path}")

            path_parts = storage_path[5:].split("/", 1)
            if len(path_parts) != 2:
                raise ValueError(f"Invalid S3 path format: {storage_path}")

            bucket, key = path_parts

            # Check if exists first
            try:
                self.s3_client.head_object(Bucket=bucket, Key=key)
            except self.s3_client.exceptions.NoSuchKey:
                logger.warning(f"File not found for deletion: {storage_path}")
                return False

            # Delete from S3
            self.s3_client.delete_object(Bucket=bucket, Key=key)
            logger.info(f"Deleted from S3: {storage_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete file from S3 {storage_path}: {e}")
            raise IOError(f"S3 storage deletion failed: {e}")

    async def exists(self, storage_path: str) -> bool:
        """Check if file exists in S3.

        Args:
            storage_path: S3 path to check

        Returns:
            True if file exists
        """
        try:
            if not storage_path.startswith("s3://"):
                return False

            path_parts = storage_path[5:].split("/", 1)
            if len(path_parts) != 2:
                return False

            bucket, key = path_parts

            try:
                self.s3_client.head_object(Bucket=bucket, Key=key)
                return True
            except self.s3_client.exceptions.NoSuchKey:
                return False

        except Exception as e:
            logger.error(f"Failed to check S3 file existence {storage_path}: {e}")
            return False

    def get_backend_name(self) -> str:
        """Get backend name."""
        return "s3"

    async def health_check(self) -> bool:
        """Check if S3 bucket is accessible."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            return True
        except Exception as e:
            logger.error(f"S3 health check failed: {e}")
            return False
