"""
Redis manager for agentbx - handles connections, serialization, and caching.
"""

import json
import pickle
import hashlib
from typing import Any, Dict, Optional, Union
from datetime import datetime, timedelta
import redis
from redis.exceptions import RedisError, ConnectionError
import logging

logger = logging.getLogger(__name__)


class RedisManager:
    """
    Manages Redis connections and provides high-level operations for agentbx.
    
    Features:
    - Connection pooling and health checks
    - Automatic serialization/deserialization of complex objects
    - Bundle storage and retrieval with metadata
    - Caching with TTL support
    - Error handling and retry logic
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 10,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
        retry_on_timeout: bool = True,
        health_check_interval: int = 30,
        default_ttl: int = 3600,  # 1 hour default
    ):
        """
        Initialize Redis manager with connection parameters.
        
        Args:
            host: Redis server hostname
            port: Redis server port
            db: Redis database number
            password: Redis password (if required)
            max_connections: Maximum connections in pool
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Connection timeout in seconds
            retry_on_timeout: Whether to retry on timeout
            health_check_interval: Health check interval in seconds
            default_ttl: Default TTL for cached items in seconds
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.default_ttl = default_ttl
        self.health_check_interval = health_check_interval
        self.last_health_check = None
        
        # Connection pool configuration
        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            retry_on_timeout=retry_on_timeout,
            decode_responses=False,  # We handle encoding ourselves
        )
        
        self._redis_client = None
        self._test_connection()
    
    def _get_client(self) -> redis.Redis:
        """Get Redis client, creating if necessary."""
        if self._redis_client is None:
            self._redis_client = redis.Redis(connection_pool=self.pool)
        return self._redis_client
    
    def _test_connection(self) -> bool:
        """Test Redis connection and log status."""
        try:
            client = self._get_client()
            client.ping()
            logger.info(f"Redis connection established to {self.host}:{self.port}")
            return True
        except (ConnectionError, RedisError) as e:
            logger.error(f"Failed to connect to Redis at {self.host}:{self.port}: {e}")
            return False
    
    def is_healthy(self) -> bool:
        """Check if Redis connection is healthy."""
        now = datetime.now()
        if (self.last_health_check is None or 
            (now - self.last_health_check).seconds > self.health_check_interval):
            try:
                client = self._get_client()
                client.ping()
                self.last_health_check = now
                return True
            except (ConnectionError, RedisError):
                return False
        return True
    
    def _serialize(self, obj: Any) -> bytes:
        """
        Serialize object to bytes for Redis storage.
        
        Handles:
        - Basic Python types (str, int, float, bool, None)
        - Lists, tuples, dicts
        - Complex objects (pickle)
        """
        if obj is None:
            return b"null"
        elif isinstance(obj, (str, int, float, bool)):
            return json.dumps(obj).encode('utf-8')
        elif isinstance(obj, (list, tuple, dict)):
            return json.dumps(obj, default=str).encode('utf-8')
        else:
            # Use pickle for complex objects (CCTBX objects, etc.)
            return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _deserialize(self, data: bytes) -> Any:
        """
        Deserialize bytes from Redis back to Python object.
        
        Handles the reverse of _serialize.
        """
        if data == b"null":
            return None
        elif data.startswith(b'{') or data.startswith(b'[') or data.startswith(b'"'):
            # JSON-encoded data
            try:
                return json.loads(data.decode('utf-8'))
            except json.JSONDecodeError:
                # Fall back to pickle if JSON fails
                return pickle.loads(data)
        else:
            # Pickle-encoded data
            return pickle.loads(data)
    
    def _generate_key(self, prefix: str, identifier: str) -> str:
        """Generate Redis key with prefix."""
        return f"agentbx:{prefix}:{identifier}"
    
    def store_bundle(self, bundle: Any, bundle_id: Optional[str] = None) -> str:
        """
        Store a bundle in Redis.
        
        Args:
            bundle: Bundle object to store
            bundle_id: Optional custom ID, auto-generated if None
            
        Returns:
            Bundle ID
        """
        if not self.is_healthy():
            raise ConnectionError("Redis connection is not healthy")
        
        # Generate bundle ID if not provided
        if bundle_id is None:
            bundle_id = self._generate_bundle_id(bundle)
        
        # Store bundle data
        key = self._generate_key("bundle", bundle_id)
        serialized_data = self._serialize(bundle)
        
        client = self._get_client()
        client.setex(key, self.default_ttl, serialized_data)
        
        # Store metadata
        metadata = {
            "bundle_id": bundle_id,
            "bundle_type": getattr(bundle, 'bundle_type', 'unknown'),
            "created_at": datetime.now().isoformat(),
            "size_bytes": len(serialized_data),
            "checksum": self._calculate_checksum(serialized_data)
        }
        
        meta_key = self._generate_key("bundle_meta", bundle_id)
        client.setex(meta_key, self.default_ttl, self._serialize(metadata))
        
        logger.debug(f"Stored bundle {bundle_id} ({len(serialized_data)} bytes)")
        return bundle_id
    
    def get_bundle(self, bundle_id: str) -> Any:
        """
        Retrieve a bundle from Redis.
        
        Args:
            bundle_id: Bundle ID to retrieve
            
        Returns:
            Deserialized bundle object
        """
        if not self.is_healthy():
            raise ConnectionError("Redis connection is not healthy")
        
        key = self._generate_key("bundle", bundle_id)
        client = self._get_client()
        
        data = client.get(key)
        if data is None:
            raise KeyError(f"Bundle {bundle_id} not found in Redis")
        
        bundle = self._deserialize(data)
        logger.debug(f"Retrieved bundle {bundle_id}")
        return bundle
    
    def delete_bundle(self, bundle_id: str) -> bool:
        """
        Delete a bundle from Redis.
        
        Args:
            bundle_id: Bundle ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        if not self.is_healthy():
            raise ConnectionError("Redis connection is not healthy")
        
        client = self._get_client()
        bundle_key = self._generate_key("bundle", bundle_id)
        meta_key = self._generate_key("bundle_meta", bundle_id)
        
        # Delete both bundle and metadata
        result = client.delete(bundle_key, meta_key)
        deleted = result > 0
        
        if deleted:
            logger.debug(f"Deleted bundle {bundle_id}")
        else:
            logger.debug(f"Bundle {bundle_id} not found for deletion")
        
        return deleted
    
    def list_bundles(self, bundle_type: Optional[str] = None) -> list[str]:
        """
        List all bundle IDs, optionally filtered by type.
        
        Args:
            bundle_type: Optional bundle type filter
            
        Returns:
            List of bundle IDs
        """
        if not self.is_healthy():
            raise ConnectionError("Redis connection is not healthy")
        
        client = self._get_client()
        pattern = self._generate_key("bundle", "*")
        keys = client.keys(pattern)
        
        bundle_ids = []
        for key in keys:
            bundle_id = key.decode('utf-8').split(':')[-1]
            
            if bundle_type:
                # Check bundle type by retrieving metadata
                try:
                    meta_key = self._generate_key("bundle_meta", bundle_id)
                    meta_data = client.get(meta_key)
                    if meta_data:
                        metadata = self._deserialize(meta_data)
                        if metadata.get('bundle_type') == bundle_type:
                            bundle_ids.append(bundle_id)
                except Exception:
                    continue
            else:
                bundle_ids.append(bundle_id)
        
        return bundle_ids
    
    def cache_get(self, key: str) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if not self.is_healthy():
            return None
        
        try:
            client = self._get_client()
            cache_key = self._generate_key("cache", key)
            data = client.get(cache_key)
            return self._deserialize(data) if data else None
        except Exception as e:
            logger.warning(f"Cache get failed for key {key}: {e}")
            return None
    
    def cache_set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
            
        Returns:
            True if successful
        """
        if not self.is_healthy():
            return False
        
        try:
            client = self._get_client()
            cache_key = self._generate_key("cache", key)
            serialized_value = self._serialize(value)
            ttl = ttl or self.default_ttl
            client.setex(cache_key, ttl, serialized_value)
            return True
        except Exception as e:
            logger.warning(f"Cache set failed for key {key}: {e}")
            return False
    
    def _generate_bundle_id(self, bundle: Any) -> str:
        """Generate unique bundle ID based on content and timestamp."""
        content = str(bundle.__dict__) + str(datetime.now().timestamp())
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA256 checksum of data."""
        return hashlib.sha256(data).hexdigest()[:16]
    
    def close(self):
        """Close Redis connection."""
        if self._redis_client:
            self._redis_client.close()
            self._redis_client = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 