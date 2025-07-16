"""
Tests for the async geometry agent system.
"""

import asyncio
import json
import pytest
import time
from datetime import datetime
from unittest.mock import Mock, patch

from agentbx.core.async_geometry_agent import AsyncGeometryAgent, GeometryRequest, GeometryResponse
from agentbx.core.agent_security_manager import AgentSecurityManager, AgentRegistration
from agentbx.core.coordinate_translator import CoordinateTranslator
from agentbx.core.redis_manager import RedisManager
from agentbx.core.redis_stream_manager import RedisStreamManager, MessageHandler


class TestAsyncGeometryAgent:
    """Test the AsyncGeometryAgent class."""
    
    @pytest.fixture
    def redis_manager(self):
        """Create a mock Redis manager."""
        manager = Mock(spec=RedisManager)
        manager.host = "localhost"
        manager.port = 6379
        manager.db = 0
        manager.password = None
        return manager
    
    @pytest.fixture
    def agent(self, redis_manager):
        """Create an AsyncGeometryAgent instance."""
        return AsyncGeometryAgent(
            agent_id="test_agent",
            redis_manager=redis_manager,
            stream_name="test_stream",
            consumer_group="test_group"
        )
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """Test agent initialization."""
        with patch('redis.asyncio.Redis') as mock_redis:
            mock_client = Mock()
            mock_redis.return_value = mock_client
            mock_client.ping.return_value = None
            
            await agent.initialize()
            
            assert agent.redis_client is not None
            mock_client.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_agent_start_stop(self, agent):
        """Test agent start and stop."""
        with patch('redis.asyncio.Redis') as mock_redis:
            mock_client = Mock()
            mock_redis.return_value = mock_client
            mock_client.ping.return_value = None
            
            await agent.initialize()
            
            # Start agent
            start_task = asyncio.create_task(agent.start())
            
            # Wait a bit for startup
            await asyncio.sleep(0.1)
            
            # Stop agent
            await agent.stop()
            
            # Cancel start task
            start_task.cancel()
            
            assert not agent.is_running
    
    def test_geometry_request_creation(self):
        """Test GeometryRequest creation."""
        request = GeometryRequest(
            request_id="test_req",
            macromolecule_bundle_id="test_bundle",
            priority=1,
            timeout_seconds=300
        )
        
        assert request.request_id == "test_req"
        assert request.macromolecule_bundle_id == "test_bundle"
        assert request.priority == 1
        assert request.timeout_seconds == 300
        assert request.created_at is not None
    
    def test_geometry_response_creation(self):
        """Test GeometryResponse creation."""
        response = GeometryResponse(
            request_id="test_req",
            success=True,
            geometry_bundle_id="test_geo_bundle",
            processing_time=1.5
        )
        
        assert response.request_id == "test_req"
        assert response.success is True
        assert response.geometry_bundle_id == "test_geo_bundle"
        assert response.processing_time == 1.5
        assert response.timestamp is not None


class TestAgentSecurityManager:
    """Test the AgentSecurityManager class."""
    
    @pytest.fixture
    def redis_manager(self):
        """Create a mock Redis manager."""
        manager = Mock(spec=RedisManager)
        manager.store_bundle.return_value = "test_bundle_id"
        return manager
    
    @pytest.fixture
    def security_manager(self, redis_manager):
        """Create an AgentSecurityManager instance."""
        return AgentSecurityManager(redis_manager)
    
    def test_agent_registration(self, security_manager):
        """Test agent registration."""
        registration = AgentRegistration(
            agent_id="test_agent",
            agent_name="Test Agent",
            agent_type="geometry_agent",
            version="1.0.0",
            permissions=["geometry_calculation", "bundle_read"]
        )
        
        success = security_manager.register_agent(registration)
        
        assert success is True
        assert "test_agent" in security_manager.registered_agents
    
    def test_agent_unregistration(self, security_manager):
        """Test agent unregistration."""
        # First register an agent
        registration = AgentRegistration(
            agent_id="test_agent",
            agent_name="Test Agent",
            agent_type="geometry_agent",
            version="1.0.0",
            permissions=["geometry_calculation"]
        )
        
        security_manager.register_agent(registration)
        
        # Then unregister
        success = security_manager.unregister_agent("test_agent")
        
        assert success is True
        assert "test_agent" not in security_manager.registered_agents
    
    def test_permission_checking(self, security_manager):
        """Test permission checking."""
        # Register an agent with permissions
        registration = AgentRegistration(
            agent_id="test_agent",
            agent_name="Test Agent",
            agent_type="geometry_agent",
            version="1.0.0",
            permissions=["geometry_calculation", "bundle_read"]
        )
        
        security_manager.register_agent(registration)
        
        # Check permissions
        assert security_manager.check_permission("test_agent", "geometry_calculation") is True
        assert security_manager.check_permission("test_agent", "bundle_read") is True
        assert security_manager.check_permission("test_agent", "invalid_permission") is False
    
    def test_module_import_validation(self, security_manager):
        """Test module import validation."""
        # Test whitelisted module
        assert security_manager.validate_module_import("test_agent", "cctbx") is True
        
        # Test blacklisted module
        assert security_manager.validate_module_import("test_agent", "os") is False
        
        # Test unwhitelisted module
        assert security_manager.validate_module_import("test_agent", "unknown_module") is False
    
    def test_security_report(self, security_manager):
        """Test security report generation."""
        # Register an agent
        registration = AgentRegistration(
            agent_id="test_agent",
            agent_name="Test Agent",
            agent_type="geometry_agent",
            version="1.0.0",
            permissions=["geometry_calculation"]
        )
        
        security_manager.register_agent(registration)
        
        # Get security report
        report = security_manager.get_agent_security_report("test_agent")
        
        assert "error" not in report
        assert report["agent_id"] == "test_agent"
        assert report["violation_count"] == 0


class TestCoordinateTranslator:
    """Test the CoordinateTranslator class."""
    
    @pytest.fixture
    def redis_manager(self):
        """Create a mock Redis manager."""
        manager = Mock(spec=RedisManager)
        manager.store_bundle.return_value = "test_bundle_id"
        return manager
    
    @pytest.fixture
    def translator(self, redis_manager):
        """Create a CoordinateTranslator instance."""
        return CoordinateTranslator(
            redis_manager=redis_manager,
            coordinate_system="cartesian",
            requires_grad=True
        )
    
    def test_cctbx_to_torch_conversion(self, translator):
        """Test CCTBX to PyTorch conversion."""
        import torch
        
        # Create mock CCTBX array
        mock_cctbx_array = Mock()
        mock_cctbx_array.size.return_value = 100
        mock_cctbx_array.__iter__.return_value = [(1.0, 2.0, 3.0)] * 100
        
        # Convert to tensor
        tensor = translator.cctbx_to_torch(mock_cctbx_array)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (100, 3)
        assert tensor.requires_grad is True
    
    def test_torch_to_cctbx_conversion(self, translator):
        """Test PyTorch to CCTBX conversion."""
        import torch
        
        # Create tensor
        tensor = torch.randn(100, 3, requires_grad=True)
        
        # Convert to CCTBX
        with patch('cctbx.array_family.flex') as mock_flex:
            mock_flex.vec3_double.return_value = Mock()
            cctbx_array = translator.torch_to_cctbx(tensor)
            
            assert cctbx_array is not None
    
    def test_bundle_registration(self, translator):
        """Test tensor bundle registration."""
        import torch
        
        tensor = torch.randn(100, 3)
        bundle_id = translator.register_bundle("test_coords", tensor)
        
        assert bundle_id == "test_bundle_id"
    
    def test_conversion_history(self, translator):
        """Test conversion history tracking."""
        import torch
        
        # Perform some conversions
        mock_cctbx = Mock()
        mock_cctbx.size.return_value = 10
        mock_cctbx.__iter__.return_value = [(1.0, 2.0, 3.0)] * 10
        
        translator.cctbx_to_torch(mock_cctbx)
        
        tensor = torch.randn(10, 3)
        translator.torch_to_cctbx(tensor)
        
        history = translator.get_conversion_history()
        
        assert len(history) == 2
        assert history[0].conversion_type == "cctbx_to_torch"
        assert history[1].conversion_type == "torch_to_cctbx"


class TestRedisStreamManager:
    """Test the RedisStreamManager class."""
    
    @pytest.fixture
    def redis_client(self):
        """Create a mock Redis client."""
        client = Mock()
        client.ping.return_value = None
        return client
    
    @pytest.fixture
    def stream_manager(self, redis_client):
        """Create a RedisStreamManager instance."""
        return RedisStreamManager(
            redis_client=redis_client,
            stream_name="test_stream",
            consumer_group="test_group",
            consumer_name="test_consumer"
        )
    
    @pytest.mark.asyncio
    async def test_stream_manager_initialization(self, stream_manager):
        """Test stream manager initialization."""
        with patch.object(stream_manager.redis_client, 'xgroup_create') as mock_create:
            mock_create.side_effect = Exception("BUSYGROUP")
            
            await stream_manager.initialize()
            
            mock_create.assert_called_once()
    
    def test_handler_registration(self, stream_manager):
        """Test message handler registration."""
        async def test_handler(message):
            return {"status": "success"}
        
        handler = MessageHandler(
            handler_name="test_handler",
            handler_func=test_handler,
            timeout_seconds=60
        )
        
        stream_manager.register_handler(handler)
        
        assert "test_handler" in stream_manager.handlers
    
    def test_metrics_tracking(self, stream_manager):
        """Test metrics tracking."""
        # Update metrics
        stream_manager._update_metrics(True, 1.5)
        stream_manager._update_metrics(False, 0.5)
        
        metrics = stream_manager.get_metrics()
        
        assert metrics.messages_processed == 1
        assert metrics.messages_failed == 1
        assert metrics.average_processing_time == 1.5


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Test a complete geometry calculation workflow."""
        # This would be a full integration test
        # For now, we'll just test that the components work together
        
        # Mock Redis manager
        redis_manager = Mock(spec=RedisManager)
        redis_manager.host = "localhost"
        redis_manager.port = 6379
        redis_manager.db = 0
        redis_manager.password = None
        redis_manager.store_bundle.return_value = "test_bundle_id"
        redis_manager.get_bundle.return_value = Mock()
        
        # Create security manager
        security_manager = AgentSecurityManager(redis_manager)
        
        # Register agent
        registration = AgentRegistration(
            agent_id="test_agent",
            agent_name="Test Agent",
            agent_type="geometry_agent",
            version="1.0.0",
            permissions=["geometry_calculation", "bundle_read", "bundle_write"]
        )
        
        success = security_manager.register_agent(registration)
        assert success is True
        
        # Check permissions
        assert security_manager.check_permission("test_agent", "geometry_calculation") is True
        
        # Create coordinate translator
        translator = CoordinateTranslator(redis_manager)
        
        # Test coordinate conversion
        import torch
        tensor = torch.randn(100, 3)
        bundle_id = translator.register_bundle("test_coords", tensor)
        assert bundle_id == "test_bundle_id"
        
        # Test that all components work together
        assert True  # If we get here, the integration test passed


if __name__ == "__main__":
    pytest.main([__file__]) 