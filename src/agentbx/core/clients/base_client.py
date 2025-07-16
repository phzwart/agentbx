"""
Base client class for Redis operations.
"""

from abc import ABC
from typing import Any
from typing import Dict
from typing import Optional

from .redis_manager import RedisManager


class BaseClient(ABC):
    """
    Base class for clients that interact with Redis.

    Provides common Redis operations and connection management.
    """

    def __init__(self, redis_manager: RedisManager, client_id: str):
        """
        Initialize base client.

        Args:
            redis_manager: Redis manager instance
            client_id: Unique identifier for this client
        """
        self.redis_manager = redis_manager
        self.client_id = client_id

    def store_bundle(self, bundle: Any, bundle_id: Optional[str] = None) -> str:
        """Store a bundle in Redis."""
        return self.redis_manager.store_bundle(bundle, bundle_id)

    def get_bundle(self, bundle_id: str) -> Any:
        """Retrieve a bundle from Redis."""
        return self.redis_manager.get_bundle(bundle_id)

    def delete_bundle(self, bundle_id: str) -> bool:
        """Delete a bundle from Redis."""
        return self.redis_manager.delete_bundle(bundle_id)

    def list_bundles(self, bundle_type: Optional[str] = None) -> list[str]:
        """List all bundles, optionally filtered by type."""
        return self.redis_manager.list_bundles(bundle_type)

    def cache_get(self, key: str) -> Any:
        """Get value from cache."""
        return self.redis_manager.cache_get(key)

    def cache_set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        return self.redis_manager.cache_set(key, value, ttl)

    def get_client_info(self) -> Dict[str, Any]:
        """Get information about this client."""
        return {
            "client_id": self.client_id,
            "client_type": self.__class__.__name__,
            "redis_healthy": self.redis_manager.is_healthy(),
        }
