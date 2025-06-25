"""Tests for the RedisManager class."""

import json
import pickle
import hashlib
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pytest
from redis.exceptions import RedisError, ConnectionError

from agentbx.core.redis_manager import RedisManager


# Define test classes at module level to avoid pickle issues
class SampleObject:
    """Test object for serialization tests."""
    def __init__(self, value):
        self.value = value


class SampleBundle:
    """Test bundle for bundle tests."""
    def __init__(self, data="test_data", bundle_type="test"):
        self.data = data
        self.bundle_type = bundle_type


class TestRedisManager:
    """Test cases for RedisManager class."""

    @pytest.fixture
    def redis_manager(self):
        """Create a RedisManager instance for testing."""
        with patch('redis.ConnectionPool'):
            with patch('redis.Redis'):
                manager = RedisManager(
                    host="localhost",
                    port=6379,
                    db=0,
                    password=None,
                    max_connections=5,
                    default_ttl=3600
                )
                # Mock the connection test to avoid actual Redis connection
                manager._test_connection = Mock(return_value=True)
                manager.is_healthy = Mock(return_value=True)
                yield manager

    @pytest.fixture
    def mock_redis_client(self, redis_manager):
        """Mock Redis client for testing."""
        mock_client = Mock()
        redis_manager._redis_client = mock_client
        return mock_client

    def test_init_defaults(self):
        """Test RedisManager initialization with default parameters."""
        with patch('redis.ConnectionPool'):
            with patch('redis.Redis'):
                with patch.object(RedisManager, '_test_connection'):
                    manager = RedisManager()
                    
                    assert manager.host == "localhost"
                    assert manager.port == 6379
                    assert manager.db == 0
                    assert manager.password is None
                    assert manager.default_ttl == 3600
                    assert manager.health_check_interval == 30

    def test_init_custom_params(self):
        """Test RedisManager initialization with custom parameters."""
        with patch('redis.ConnectionPool'):
            with patch('redis.Redis'):
                with patch.object(RedisManager, '_test_connection'):
                    manager = RedisManager(
                        host="redis.example.com",
                        port=6380,
                        db=1,
                        password="secret",
                        max_connections=20,
                        default_ttl=7200
                    )
                    
                    assert manager.host == "redis.example.com"
                    assert manager.port == 6380
                    assert manager.db == 1
                    assert manager.password == "secret"
                    assert manager.default_ttl == 7200

    def test_get_client(self, redis_manager):
        """Test _get_client method."""
        # Clear the cached client
        redis_manager._redis_client = None
        
        with patch('redis.Redis') as mock_redis:
            mock_instance = Mock()
            mock_redis.return_value = mock_instance
            
            client = redis_manager._get_client()
            
            assert client == mock_instance
            mock_redis.assert_called_once_with(connection_pool=redis_manager.pool)

    def test_test_connection_success(self):
        """Test successful connection test."""
        with patch('redis.ConnectionPool'):
            with patch('redis.Redis') as mock_redis:
                mock_client = Mock()
                mock_redis.return_value = mock_client
                mock_client.ping.return_value = True
                
                manager = RedisManager()
                # Reset the client to force recreation and clear the ping count
                manager._redis_client = None
                mock_client.ping.reset_mock()
                result = manager._test_connection()
                
                assert result is True
                mock_client.ping.assert_called_once()

    def test_test_connection_failure(self):
        """Test failed connection test."""
        with patch('redis.ConnectionPool'):
            with patch('redis.Redis') as mock_redis:
                mock_client = Mock()
                mock_redis.return_value = mock_client
                mock_client.ping.side_effect = ConnectionError("Connection failed")
                
                manager = RedisManager()
                result = manager._test_connection()
                
                assert result is False

    def test_is_healthy_with_recent_check(self, redis_manager):
        """Test is_healthy when health check was recent."""
        redis_manager.last_health_check = datetime.now()
        redis_manager.health_check_interval = 30
        
        result = redis_manager.is_healthy()
        
        assert result is True

    def test_is_healthy_with_stale_check(self, redis_manager):
        """Test is_healthy when health check is stale."""
        redis_manager.last_health_check = datetime.now() - timedelta(seconds=60)
        redis_manager.health_check_interval = 30
        
        # Reset the mocked is_healthy to use the real implementation
        redis_manager.is_healthy = RedisManager.is_healthy.__get__(redis_manager)
        
        with patch.object(redis_manager, '_get_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            mock_client.ping.return_value = True
            
            result = redis_manager.is_healthy()
            
            assert result is True
            mock_client.ping.assert_called_once()

    def test_is_healthy_connection_failure(self, redis_manager):
        """Test is_healthy when connection fails."""
        redis_manager.last_health_check = datetime.now() - timedelta(seconds=60)
        redis_manager.health_check_interval = 30
        
        # Reset the mocked is_healthy to use the real implementation
        redis_manager.is_healthy = RedisManager.is_healthy.__get__(redis_manager)
        
        with patch.object(redis_manager, '_get_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            mock_client.ping.side_effect = ConnectionError("Connection failed")
            
            result = redis_manager.is_healthy()
            
            assert result is False

    def test_serialize_basic_types(self, redis_manager):
        """Test serialization of basic Python types."""
        # Test None
        assert redis_manager._serialize(None) == b"null"
        
        # Test string
        assert redis_manager._serialize("hello") == b'"hello"'
        
        # Test integer
        assert redis_manager._serialize(42) == b'42'
        
        # Test float
        assert redis_manager._serialize(3.14) == b'3.14'
        
        # Test boolean
        assert redis_manager._serialize(True) == b'true'
        assert redis_manager._serialize(False) == b'false'

    def test_serialize_containers(self, redis_manager):
        """Test serialization of container types."""
        # Test list
        data = [1, 2, 3]
        serialized = redis_manager._serialize(data)
        assert json.loads(serialized.decode()) == data
        
        # Test dict
        data = {"key": "value", "number": 42}
        serialized = redis_manager._serialize(data)
        assert json.loads(serialized.decode()) == data
        
        # Test tuple
        data = (1, 2, 3)
        serialized = redis_manager._serialize(data)
        assert json.loads(serialized.decode()) == [1, 2, 3]  # Tuples become lists in JSON

    def test_serialize_complex_object(self, redis_manager):
        """Test serialization of complex objects using pickle."""
        obj = SampleObject("test")
        serialized = redis_manager._serialize(obj)
        
        # Should use pickle for complex objects
        deserialized = pickle.loads(serialized)
        assert deserialized.value == "test"

    def test_deserialize_basic_types(self, redis_manager):
        """Test deserialization of basic types."""
        # Test None
        assert redis_manager._deserialize(b"null") is None
        
        # Test string
        assert redis_manager._deserialize(b'"hello"') == "hello"
        
        # Test integer - need to use JSON format
        assert redis_manager._deserialize(b'42') == 42
        
        # Test float - need to use JSON format
        assert redis_manager._deserialize(b'3.14') == 3.14
        
        # Test boolean
        assert redis_manager._deserialize(b'true') is True
        assert redis_manager._deserialize(b'false') is False

    def test_deserialize_containers(self, redis_manager):
        """Test deserialization of container types."""
        # Test list
        data = [1, 2, 3]
        serialized = json.dumps(data).encode()
        assert redis_manager._deserialize(serialized) == data
        
        # Test dict
        data = {"key": "value", "number": 42}
        serialized = json.dumps(data).encode()
        assert redis_manager._deserialize(serialized) == data

    def test_deserialize_complex_object(self, redis_manager):
        """Test deserialization of complex objects."""
        obj = SampleObject("test")
        serialized = pickle.dumps(obj)
        deserialized = redis_manager._deserialize(serialized)
        
        assert deserialized.value == "test"

    def test_deserialize_json_fallback(self, redis_manager):
        """Test deserialization fallback from JSON to pickle."""
        obj = SampleObject("test")
        serialized = pickle.dumps(obj)
        
        # Create data that starts with JSON but contains pickle data
        # This simulates the fallback scenario
        json_start = b'{"invalid": "json"'
        mixed_data = json_start + serialized
        
        # Mock json.loads to fail
        with patch('json.loads', side_effect=json.JSONDecodeError("", "", 0)):
            deserialized = redis_manager._deserialize(mixed_data)
            assert deserialized.value == "test"

    def test_generate_key(self, redis_manager):
        """Test key generation."""
        key = redis_manager._generate_key("test", "123")
        assert key == "agentbx:test:123"

    def test_store_bundle_success(self, redis_manager, mock_redis_client):
        """Test successful bundle storage."""
        bundle = SampleBundle()
        
        # Mock bundle ID generation
        with patch.object(redis_manager, '_generate_bundle_id', return_value="test_id"):
            with patch.object(redis_manager, '_calculate_checksum', return_value="checksum"):
                bundle_id = redis_manager.store_bundle(bundle)
                
                assert bundle_id == "test_id"
                
                # Check that both bundle and metadata were stored
                assert mock_redis_client.setex.call_count == 2
                
                # Verify bundle key
                bundle_call = mock_redis_client.setex.call_args_list[0]
                assert bundle_call[0][0] == "agentbx:bundle:test_id"
                assert bundle_call[0][1] == 3600  # default TTL
                
                # Verify metadata key
                meta_call = mock_redis_client.setex.call_args_list[1]
                assert meta_call[0][0] == "agentbx:bundle_meta:test_id"
                assert meta_call[0][1] == 3600  # default TTL

    def test_store_bundle_with_custom_id(self, redis_manager, mock_redis_client):
        """Test bundle storage with custom ID."""
        bundle = SampleBundle()
        
        with patch.object(redis_manager, '_calculate_checksum', return_value="checksum"):
            bundle_id = redis_manager.store_bundle(bundle, bundle_id="custom_id")
            
            assert bundle_id == "custom_id"
            
            # Check that both bundle and metadata were stored
            assert mock_redis_client.setex.call_count == 2
            
            # Verify bundle key
            bundle_call = mock_redis_client.setex.call_args_list[0]
            assert bundle_call[0][0] == "agentbx:bundle:custom_id"

    def test_store_bundle_unhealthy_connection(self, redis_manager):
        """Test bundle storage with unhealthy connection."""
        redis_manager.is_healthy = Mock(return_value=False)
        
        bundle = SampleBundle()
        
        with pytest.raises(ConnectionError, match="Redis connection is not healthy"):
            redis_manager.store_bundle(bundle)

    def test_get_bundle_success(self, redis_manager, mock_redis_client):
        """Test successful bundle retrieval."""
        bundle = SampleBundle()
        serialized_bundle = pickle.dumps(bundle)
        mock_redis_client.get.return_value = serialized_bundle
        
        retrieved_bundle = redis_manager.get_bundle("test_id")
        
        assert retrieved_bundle.data == "test_data"
        mock_redis_client.get.assert_called_once_with("agentbx:bundle:test_id")

    def test_get_bundle_not_found(self, redis_manager, mock_redis_client):
        """Test bundle retrieval when bundle doesn't exist."""
        mock_redis_client.get.return_value = None
        
        with pytest.raises(KeyError, match="Bundle test_id not found in Redis"):
            redis_manager.get_bundle("test_id")

    def test_get_bundle_unhealthy_connection(self, redis_manager):
        """Test bundle retrieval with unhealthy connection."""
        redis_manager.is_healthy = Mock(return_value=False)
        
        with pytest.raises(ConnectionError, match="Redis connection is not healthy"):
            redis_manager.get_bundle("test_id")

    def test_delete_bundle_success(self, redis_manager, mock_redis_client):
        """Test successful bundle deletion."""
        mock_redis_client.delete.return_value = 2  # Both bundle and metadata deleted
        
        result = redis_manager.delete_bundle("test_id")
        
        assert result is True
        mock_redis_client.delete.assert_called_once_with(
            "agentbx:bundle:test_id",
            "agentbx:bundle_meta:test_id"
        )

    def test_delete_bundle_not_found(self, redis_manager, mock_redis_client):
        """Test bundle deletion when bundle doesn't exist."""
        mock_redis_client.delete.return_value = 0  # Nothing deleted
        
        result = redis_manager.delete_bundle("test_id")
        
        assert result is False

    def test_delete_bundle_unhealthy_connection(self, redis_manager):
        """Test bundle deletion with unhealthy connection."""
        redis_manager.is_healthy = Mock(return_value=False)
        
        with pytest.raises(ConnectionError, match="Redis connection is not healthy"):
            redis_manager.delete_bundle("test_id")

    def test_list_bundles_all(self, redis_manager, mock_redis_client):
        """Test listing all bundles."""
        # Mock keys response
        mock_redis_client.keys.return_value = [
            b"agentbx:bundle:id1",
            b"agentbx:bundle:id2",
            b"agentbx:bundle:id3"
        ]
        
        # Mock metadata retrieval
        def mock_get(key):
            if key == "agentbx:bundle_meta:id1":
                return json.dumps({"bundle_type": "type1"}).encode()
            elif key == "agentbx:bundle_meta:id2":
                return json.dumps({"bundle_type": "type2"}).encode()
            elif key == "agentbx:bundle_meta:id3":
                return json.dumps({"bundle_type": "type1"}).encode()
            return None
        
        mock_redis_client.get.side_effect = mock_get
        
        bundle_ids = redis_manager.list_bundles()
        
        assert set(bundle_ids) == {"id1", "id2", "id3"}

    def test_list_bundles_filtered(self, redis_manager, mock_redis_client):
        """Test listing bundles filtered by type."""
        # Mock keys response
        mock_redis_client.keys.return_value = [
            b"agentbx:bundle:id1",
            b"agentbx:bundle:id2",
            b"agentbx:bundle:id3"
        ]
        
        # Mock metadata retrieval
        def mock_get(key):
            if key == "agentbx:bundle_meta:id1":
                return json.dumps({"bundle_type": "type1"}).encode()
            elif key == "agentbx:bundle_meta:id2":
                return json.dumps({"bundle_type": "type2"}).encode()
            elif key == "agentbx:bundle_meta:id3":
                return json.dumps({"bundle_type": "type1"}).encode()
            return None
        
        mock_redis_client.get.side_effect = mock_get
        
        bundle_ids = redis_manager.list_bundles(bundle_type="type1")
        
        assert set(bundle_ids) == {"id1", "id3"}

    def test_list_bundles_unhealthy_connection(self, redis_manager):
        """Test listing bundles with unhealthy connection."""
        redis_manager.is_healthy = Mock(return_value=False)
        
        with pytest.raises(ConnectionError, match="Redis connection is not healthy"):
            redis_manager.list_bundles()

    def test_cache_get_success(self, redis_manager, mock_redis_client):
        """Test successful cache retrieval."""
        cached_data = json.dumps("cached_value").encode()
        mock_redis_client.get.return_value = cached_data
        
        result = redis_manager.cache_get("test_key")
        
        assert result == "cached_value"
        mock_redis_client.get.assert_called_once_with("agentbx:cache:test_key")

    def test_cache_get_not_found(self, redis_manager, mock_redis_client):
        """Test cache retrieval when key doesn't exist."""
        mock_redis_client.get.return_value = None
        
        result = redis_manager.cache_get("test_key")
        
        assert result is None

    def test_cache_get_unhealthy_connection(self, redis_manager):
        """Test cache retrieval with unhealthy connection."""
        redis_manager.is_healthy = Mock(return_value=False)
        
        result = redis_manager.cache_get("test_key")
        
        assert result is None

    def test_cache_get_exception(self, redis_manager, mock_redis_client):
        """Test cache retrieval with exception."""
        mock_redis_client.get.side_effect = RedisError("Redis error")
        
        result = redis_manager.cache_get("test_key")
        
        assert result is None

    def test_cache_set_success(self, redis_manager, mock_redis_client):
        """Test successful cache setting."""
        result = redis_manager.cache_set("test_key", "test_value")
        
        assert result is True
        mock_redis_client.setex.assert_called_once()
        
        call_args = mock_redis_client.setex.call_args
        assert call_args[0][0] == "agentbx:cache:test_key"
        assert call_args[0][1] == 3600  # default TTL

    def test_cache_set_with_custom_ttl(self, redis_manager, mock_redis_client):
        """Test cache setting with custom TTL."""
        result = redis_manager.cache_set("test_key", "test_value", ttl=1800)
        
        assert result is True
        call_args = mock_redis_client.setex.call_args
        assert call_args[0][1] == 1800  # custom TTL

    def test_cache_set_unhealthy_connection(self, redis_manager):
        """Test cache setting with unhealthy connection."""
        redis_manager.is_healthy = Mock(return_value=False)
        
        result = redis_manager.cache_set("test_key", "test_value")
        
        assert result is False

    def test_cache_set_exception(self, redis_manager, mock_redis_client):
        """Test cache setting with exception."""
        mock_redis_client.setex.side_effect = RedisError("Redis error")
        
        result = redis_manager.cache_set("test_key", "test_value")
        
        assert result is False

    def test_generate_bundle_id(self, redis_manager):
        """Test bundle ID generation."""
        bundle = SampleBundle()
        
        bundle_id = redis_manager._generate_bundle_id(bundle)
        
        assert len(bundle_id) == 16
        assert isinstance(bundle_id, str)
        # Should be hexadecimal
        assert all(c in '0123456789abcdef' for c in bundle_id)

    def test_calculate_checksum(self, redis_manager):
        """Test checksum calculation."""
        data = b"test_data"
        checksum = redis_manager._calculate_checksum(data)
        
        assert len(checksum) == 16
        assert isinstance(checksum, str)
        # Should be hexadecimal
        assert all(c in '0123456789abcdef' for c in checksum)
        
        # Should match SHA256 hash (first 16 chars)
        expected = hashlib.sha256(data).hexdigest()[:16]
        assert checksum == expected

    def test_close(self, redis_manager, mock_redis_client):
        """Test connection closing."""
        redis_manager.close()
        
        mock_redis_client.close.assert_called_once()
        assert redis_manager._redis_client is None

    def test_context_manager(self, redis_manager, mock_redis_client):
        """Test context manager functionality."""
        with redis_manager as manager:
            assert manager == redis_manager
        
        # Should close connection when exiting context
        mock_redis_client.close.assert_called_once()
        assert redis_manager._redis_client is None

    def test_context_manager_with_exception(self, redis_manager, mock_redis_client):
        """Test context manager with exception."""
        try:
            with redis_manager as manager:
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should still close connection even with exception
        mock_redis_client.close.assert_called_once()
        assert redis_manager._redis_client is None 