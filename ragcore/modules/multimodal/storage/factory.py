"""Storage backend factory."""

import logging
from typing import Optional

from ragcore.modules.multimodal.storage.base import StorageBackend
from ragcore.modules.multimodal.storage.local import LocalStorage
from ragcore.modules.multimodal.storage.s3 import S3Storage
from ragcore.modules.multimodal.storage.azure_blob import AzureBlobStorage

logger = logging.getLogger(__name__)


def get_storage_backend(
    backend_type: str,
    **kwargs,
) -> StorageBackend:
    """Get storage backend instance based on configuration.

    Args:
        backend_type: Storage backend type (local, s3, azure_blob)
        **kwargs: Backend-specific configuration

    Returns:
        StorageBackend instance

    Raises:
        ValueError: If backend_type is unknown
    """
    if backend_type == "local":
        base_path = kwargs.get("path", "/data/multimodal")
        return LocalStorage(base_path=base_path)

    elif backend_type == "s3":
        bucket_name = kwargs.get("bucket")
        if not bucket_name:
            raise ValueError("S3 storage requires 'bucket' parameter")
        region = kwargs.get("region", "us-east-1")
        prefix = kwargs.get("prefix", "multimodal/")
        return S3Storage(bucket_name=bucket_name, region=region, prefix=prefix)

    elif backend_type == "azure_blob":
        connection_string = kwargs.get("connection_string")
        container_name = kwargs.get("container")
        if not connection_string or not container_name:
            raise ValueError("Azure Blob storage requires 'connection_string' and 'container' parameters")
        prefix = kwargs.get("prefix", "multimodal/")
        return AzureBlobStorage(
            connection_string=connection_string,
            container_name=container_name,
            prefix=prefix,
        )

    else:
        raise ValueError(f"Unknown storage backend: {backend_type}")


def get_storage_backend_from_config(config) -> StorageBackend:
    """Get storage backend from application config.

    Args:
        config: Application configuration object

    Returns:
        StorageBackend instance
    """
    backend_type = getattr(config, "multimodal_storage_backend", "local")

    if backend_type == "local":
        path = getattr(config, "multimodal_storage_path", "/data/multimodal")
        return get_storage_backend("local", path=path)

    elif backend_type == "s3":
        bucket = getattr(config, "multimodal_s3_bucket", None)
        region = getattr(config, "multimodal_s3_region", "us-east-1")
        if not bucket:
            raise ValueError("S3 storage configured but multimodal_s3_bucket not set")
        return get_storage_backend("s3", bucket=bucket, region=region)

    elif backend_type == "azure_blob":
        connection_string = getattr(config, "multimodal_azure_connection_string", None)
        container = getattr(config, "multimodal_azure_container", None)
        if not connection_string or not container:
            raise ValueError(
                "Azure Blob storage configured but missing multimodal_azure_connection_string or multimodal_azure_container"
            )
        return get_storage_backend("azure_blob", connection_string=connection_string, container=container)

    else:
        logger.warning(f"Unknown storage backend {backend_type}, defaulting to local")
        path = getattr(config, "multimodal_storage_path", "/data/multimodal")
        return get_storage_backend("local", path=path)
