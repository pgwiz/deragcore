"""ChromaDB integration module - Client, collections, sync, and performance routing."""

from ragcore.modules.memory.chroma.client import ChromaClientManager, ChromaConfig
from ragcore.modules.memory.chroma.collection_manager import ChromaCollectionManager
from ragcore.modules.memory.chroma.sync_manager import ChromaMemorySyncManager
from ragcore.modules.memory.chroma.performance_router import SmartSearchRouter

__all__ = [
    "ChromaClientManager",
    "ChromaConfig",
    "ChromaCollectionManager",
    "ChromaMemorySyncManager",
    "SmartSearchRouter",
]
