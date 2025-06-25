"""Tests for the BaseClient class."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict

from agentbx.core.base_client import BaseClient
from agentbx.core.redis_manager import RedisManager


class ConcreteClient(BaseClient):
    """Concrete implementation of BaseClient for testing."""
    
    def __init__(self, redis_manager: RedisManager, client_id: str):
        super().__init__(redis_manager, client_id)
    
    @property
    def client_type(self) -> str:
        return "concrete_test_client"


class TestBaseClient:
    """Test cases for BaseClient class."""

    @pytest.fixture
    def mock_redis_manager(self):
        """Create a mock RedisManager for testing."""
        mock_manager = Mock(spec=RedisManager)
        mock_manager.is_healthy.return_value = True
        return mock_manager

    @pytest.fixture
    def base_client(self, mock_redis_manager):
        """Create a BaseClient instance for testing."""
        return ConcreteClient(mock_redis_manager, "test_client_001")

    @pytest.fixture
    def sample_bundle(self):
        """Create a sample bundle for testing."""
        bundle = Mock()
        bundle.bundle_type = "test_bundle"
        bundle.data = "test_data"
        return bundle

    def test_init(self, mock_redis_manager):
        """Test BaseClient initialization."""
        client = ConcreteClient(mock_redis_manager, "test_client_001")
        
        assert client.redis_manager == mock_redis_manager
        assert client.client_id == "test_client_001"

    def test_init_with_different_client_id(self, mock_redis_manager):
        """Test BaseClient initialization with different client ID."""
        client = ConcreteClient(mock_redis_manager, "another_client")
        
        assert client.client_id == "another_client"

    def test_store_bundle_default_id(self, base_client, sample_bundle, mock_redis_manager):
        """Test storing a bundle with auto-generated ID."""
        mock_redis_manager.store_bundle.return_value = "auto_generated_id"
        
        result = base_client.store_bundle(sample_bundle)
        
        assert result == "auto_generated_id"
        mock_redis_manager.store_bundle.assert_called_once_with(sample_bundle, None)

    def test_store_bundle_custom_id(self, base_client, sample_bundle, mock_redis_manager):
        """Test storing a bundle with custom ID."""
        custom_id = "custom_bundle_123"
        mock_redis_manager.store_bundle.return_value = custom_id
        
        result = base_client.store_bundle(sample_bundle, custom_id)
        
        assert result == custom_id
        mock_redis_manager.store_bundle.assert_called_once_with(sample_bundle, custom_id)

    def test_store_bundle_redis_error(self, base_client, sample_bundle, mock_redis_manager):
        """Test store_bundle when Redis manager raises an error."""
        from redis.exceptions import RedisError
        mock_redis_manager.store_bundle.side_effect = RedisError("Redis error")
        
        with pytest.raises(RedisError, match="Redis error"):
            base_client.store_bundle(sample_bundle)

    def test_get_bundle_success(self, base_client, sample_bundle, mock_redis_manager):
        """Test retrieving a bundle successfully."""
        bundle_id = "test_bundle_123"
        mock_redis_manager.get_bundle.return_value = sample_bundle
        
        result = base_client.get_bundle(bundle_id)
        
        assert result == sample_bundle
        mock_redis_manager.get_bundle.assert_called_once_with(bundle_id)

    def test_get_bundle_not_found(self, base_client, mock_redis_manager):
        """Test retrieving a non-existent bundle."""
        bundle_id = "non_existent_bundle"
        mock_redis_manager.get_bundle.return_value = None
        
        result = base_client.get_bundle(bundle_id)
        
        assert result is None
        mock_redis_manager.get_bundle.assert_called_once_with(bundle_id)

    def test_get_bundle_redis_error(self, base_client, mock_redis_manager):
        """Test get_bundle when Redis manager raises an error."""
        from redis.exceptions import RedisError
        bundle_id = "test_bundle_123"
        mock_redis_manager.get_bundle.side_effect = RedisError("Redis error")
        
        with pytest.raises(RedisError, match="Redis error"):
            base_client.get_bundle(bundle_id)

    def test_delete_bundle_success(self, base_client, mock_redis_manager):
        """Test deleting a bundle successfully."""
        bundle_id = "test_bundle_123"
        mock_redis_manager.delete_bundle.return_value = True
        
        result = base_client.delete_bundle(bundle_id)
        
        assert result is True
        mock_redis_manager.delete_bundle.assert_called_once_with(bundle_id)

    def test_delete_bundle_not_found(self, base_client, mock_redis_manager):
        """Test deleting a non-existent bundle."""
        bundle_id = "non_existent_bundle"
        mock_redis_manager.delete_bundle.return_value = False
        
        result = base_client.delete_bundle(bundle_id)
        
        assert result is False
        mock_redis_manager.delete_bundle.assert_called_once_with(bundle_id)

    def test_delete_bundle_redis_error(self, base_client, mock_redis_manager):
        """Test delete_bundle when Redis manager raises an error."""
        from redis.exceptions import RedisError
        bundle_id = "test_bundle_123"
        mock_redis_manager.delete_bundle.side_effect = RedisError("Redis error")
        
        with pytest.raises(RedisError, match="Redis error"):
            base_client.delete_bundle(bundle_id)

    def test_list_bundles_all(self, base_client, mock_redis_manager):
        """Test listing all bundles."""
        expected_bundles = ["bundle_1", "bundle_2", "bundle_3"]
        mock_redis_manager.list_bundles.return_value = expected_bundles
        
        result = base_client.list_bundles()
        
        assert result == expected_bundles
        mock_redis_manager.list_bundles.assert_called_once_with(None)

    def test_list_bundles_filtered(self, base_client, mock_redis_manager):
        """Test listing bundles filtered by type."""
        bundle_type = "experimental_data"
        expected_bundles = ["exp_bundle_1", "exp_bundle_2"]
        mock_redis_manager.list_bundles.return_value = expected_bundles
        
        result = base_client.list_bundles(bundle_type)
        
        assert result == expected_bundles
        mock_redis_manager.list_bundles.assert_called_once_with(bundle_type)

    def test_list_bundles_empty(self, base_client, mock_redis_manager):
        """Test listing bundles when none exist."""
        mock_redis_manager.list_bundles.return_value = []
        
        result = base_client.list_bundles()
        
        assert result == []
        mock_redis_manager.list_bundles.assert_called_once_with(None)

    def test_list_bundles_redis_error(self, base_client, mock_redis_manager):
        """Test list_bundles when Redis manager raises an error."""
        from redis.exceptions import RedisError
        mock_redis_manager.list_bundles.side_effect = RedisError("Redis error")
        
        with pytest.raises(RedisError, match="Redis error"):
            base_client.list_bundles()

    def test_cache_get_success(self, base_client, mock_redis_manager):
        """Test getting a value from cache successfully."""
        cache_key = "test_cache_key"
        cached_value = {"data": "cached_data"}
        mock_redis_manager.cache_get.return_value = cached_value
        
        result = base_client.cache_get(cache_key)
        
        assert result == cached_value
        mock_redis_manager.cache_get.assert_called_once_with(cache_key)

    def test_cache_get_not_found(self, base_client, mock_redis_manager):
        """Test getting a non-existent value from cache."""
        cache_key = "non_existent_key"
        mock_redis_manager.cache_get.return_value = None
        
        result = base_client.cache_get(cache_key)
        
        assert result is None
        mock_redis_manager.cache_get.assert_called_once_with(cache_key)

    def test_cache_get_redis_error(self, base_client, mock_redis_manager):
        """Test cache_get when Redis manager raises an error."""
        from redis.exceptions import RedisError
        cache_key = "test_cache_key"
        mock_redis_manager.cache_get.side_effect = RedisError("Redis error")
        
        with pytest.raises(RedisError, match="Redis error"):
            base_client.cache_get(cache_key)

    def test_cache_set_success(self, base_client, mock_redis_manager):
        """Test setting a value in cache successfully."""
        cache_key = "test_cache_key"
        cache_value = {"data": "value_to_cache"}
        mock_redis_manager.cache_set.return_value = True
        
        result = base_client.cache_set(cache_key, cache_value)
        
        assert result is True
        mock_redis_manager.cache_set.assert_called_once_with(cache_key, cache_value, None)

    def test_cache_set_with_ttl(self, base_client, mock_redis_manager):
        """Test setting a value in cache with custom TTL."""
        cache_key = "test_cache_key"
        cache_value = {"data": "value_to_cache"}
        ttl = 1800  # 30 minutes
        mock_redis_manager.cache_set.return_value = True
        
        result = base_client.cache_set(cache_key, cache_value, ttl)
        
        assert result is True
        mock_redis_manager.cache_set.assert_called_once_with(cache_key, cache_value, ttl)

    def test_cache_set_failure(self, base_client, mock_redis_manager):
        """Test setting a value in cache when it fails."""
        cache_key = "test_cache_key"
        cache_value = {"data": "value_to_cache"}
        mock_redis_manager.cache_set.return_value = False
        
        result = base_client.cache_set(cache_key, cache_value)
        
        assert result is False
        mock_redis_manager.cache_set.assert_called_once_with(cache_key, cache_value, None)

    def test_cache_set_redis_error(self, base_client, mock_redis_manager):
        """Test cache_set when Redis manager raises an error."""
        from redis.exceptions import RedisError
        cache_key = "test_cache_key"
        cache_value = {"data": "value_to_cache"}
        mock_redis_manager.cache_set.side_effect = RedisError("Redis error")
        
        with pytest.raises(RedisError, match="Redis error"):
            base_client.cache_set(cache_key, cache_value)

    def test_get_client_info_healthy(self, base_client, mock_redis_manager):
        """Test getting client info when Redis is healthy."""
        mock_redis_manager.is_healthy.return_value = True
        
        result = base_client.get_client_info()
        
        expected_info = {
            "client_id": "test_client_001",
            "client_type": "ConcreteClient",
            "redis_healthy": True
        }
        assert result == expected_info
        mock_redis_manager.is_healthy.assert_called_once()

    def test_get_client_info_unhealthy(self, base_client, mock_redis_manager):
        """Test getting client info when Redis is unhealthy."""
        mock_redis_manager.is_healthy.return_value = False
        
        result = base_client.get_client_info()
        
        expected_info = {
            "client_id": "test_client_001",
            "client_type": "ConcreteClient",
            "redis_healthy": False
        }
        assert result == expected_info
        mock_redis_manager.is_healthy.assert_called_once()

    def test_get_client_info_different_client_type(self, mock_redis_manager):
        """Test getting client info for different client types."""
        class AnotherClient(BaseClient):
            pass
        
        client = AnotherClient(mock_redis_manager, "another_client")
        mock_redis_manager.is_healthy.return_value = True
        
        result = client.get_client_info()
        
        expected_info = {
            "client_id": "another_client",
            "client_type": "AnotherClient",
            "redis_healthy": True
        }
        assert result == expected_info

    def test_method_chaining(self, base_client, mock_redis_manager, sample_bundle):
        """Test that methods can be chained properly."""
        # Setup mocks
        mock_redis_manager.store_bundle.return_value = "bundle_123"
        mock_redis_manager.get_bundle.return_value = sample_bundle
        mock_redis_manager.delete_bundle.return_value = True
        
        # Test chaining store -> get -> delete
        bundle_id = base_client.store_bundle(sample_bundle)
        retrieved_bundle = base_client.get_bundle(bundle_id)
        deleted = base_client.delete_bundle(bundle_id)
        
        assert bundle_id == "bundle_123"
        assert retrieved_bundle == sample_bundle
        assert deleted is True

    def test_edge_case_empty_string_client_id(self, mock_redis_manager):
        """Test BaseClient with empty string client ID."""
        client = ConcreteClient(mock_redis_manager, "")
        
        assert client.client_id == ""
        
        mock_redis_manager.is_healthy.return_value = True
        info = client.get_client_info()
        
        assert info["client_id"] == ""
        assert info["client_type"] == "ConcreteClient"

    def test_edge_case_none_values(self, base_client, mock_redis_manager):
        """Test BaseClient methods with None values."""
        # Test storing None bundle
        mock_redis_manager.store_bundle.return_value = "bundle_none"
        result = base_client.store_bundle(None)
        assert result == "bundle_none"
        
        # Test getting None from cache
        mock_redis_manager.cache_get.return_value = None
        result = base_client.cache_get("none_key")
        assert result is None

    def test_edge_case_complex_objects(self, base_client, mock_redis_manager):
        """Test BaseClient with complex objects."""
        complex_obj = {
            "nested": {
                "list": [1, 2, 3],
                "tuple": (4, 5, 6),
                "set": {7, 8, 9}
            },
            "function": lambda x: x * 2
        }
        
        mock_redis_manager.store_bundle.return_value = "complex_bundle"
        mock_redis_manager.get_bundle.return_value = complex_obj
        
        # Test storing complex object
        bundle_id = base_client.store_bundle(complex_obj)
        assert bundle_id == "complex_bundle"
        
        # Test retrieving complex object
        retrieved = base_client.get_bundle(bundle_id)
        assert retrieved == complex_obj

    def test_integration_like_workflow(self, base_client, mock_redis_manager, sample_bundle):
        """Test a complete workflow similar to real usage."""
        # Setup
        mock_redis_manager.store_bundle.return_value = "workflow_bundle_123"
        mock_redis_manager.get_bundle.return_value = sample_bundle
        mock_redis_manager.list_bundles.return_value = ["workflow_bundle_123"]
        mock_redis_manager.delete_bundle.return_value = True
        mock_redis_manager.cache_set.return_value = True
        mock_redis_manager.cache_get.return_value = "cached_result"
        mock_redis_manager.is_healthy.return_value = True
        
        # Simulate a complete workflow
        # 1. Store initial bundle
        bundle_id = base_client.store_bundle(sample_bundle)
        
        # 2. Cache some computation result
        cache_key = f"computation_{bundle_id}"
        base_client.cache_set(cache_key, "cached_result")
        
        # 3. List bundles
        bundles = base_client.list_bundles()
        
        # 4. Retrieve bundle
        retrieved_bundle = base_client.get_bundle(bundle_id)
        
        # 5. Get cached result
        cached_result = base_client.cache_get(cache_key)
        
        # 6. Delete bundle
        deleted = base_client.delete_bundle(bundle_id)
        
        # 7. Get client info
        info = base_client.get_client_info()
        
        # Assertions
        assert bundle_id == "workflow_bundle_123"
        assert bundles == ["workflow_bundle_123"]
        assert retrieved_bundle == sample_bundle
        assert cached_result == "cached_result"
        assert deleted is True
        assert info["client_id"] == "test_client_001"
        assert info["redis_healthy"] is True

    def test_error_propagation(self, base_client, mock_redis_manager, sample_bundle):
        """Test that errors from Redis manager are properly propagated."""
        from redis.exceptions import ConnectionError, RedisError
        
        # Test different types of Redis errors
        errors_to_test = [
            ConnectionError("Connection failed"),
            RedisError("General Redis error"),
            ValueError("Invalid value"),
            TypeError("Type error")
        ]
        
        for error in errors_to_test:
            mock_redis_manager.store_bundle.side_effect = error
            
            with pytest.raises(type(error)):
                base_client.store_bundle(sample_bundle)
            
            # Reset for next iteration
            mock_redis_manager.store_bundle.side_effect = None 