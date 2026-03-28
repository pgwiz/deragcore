"""Storage backend implementations for multi-modal content.

Abstracts file storage across local, S3, and Azure Blob Storage.
"""

from ragcore.modules.multimodal.storage.base import StorageBackend
from ragcore.modules.multimodal.storage.local import LocalStorage
from ragcore.modules.multimodal.storage.s3 import S3Storage
from ragcore.modules.multimodal.storage.azure_blob import AzureBlobStorage
from ragcore.modules.multimodal.storage.factory import get_storage_backend, get_storage_backend_from_config

__all__ = [
    "StorageBackend",
    "LocalStorage",
    "S3Storage",
    "AzureBlobStorage",
    "get_storage_backend",
    "get_storage_backend_from_config",
]

