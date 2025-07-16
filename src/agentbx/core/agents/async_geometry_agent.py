"""
Asynchronous geometry agent for background geometry calculations.

This agent runs as a background service and processes geometry calculation
requests from Redis streams using the existing CctbxGeometryProcessor.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, List
from dataclasses import dataclass

import redis.asyncio as redis
from pydantic import BaseModel, Field

from ..redis_manager import RedisManager
from ..bundle_base import Bundle
from ..processors.geometry_processor import CctbxGeometryProcessor
from ..schemas.generated import AgentSecurityBundle, AgentConfigurationBundle, RedisStreamsBundle


@dataclass
class GeometryRequest:
    """Represents a geometry calculation request."""
    request_id: str
    macromolecule_bundle_id: str
    priority: int = 1
    timeout_seconds: int = 300
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class GeometryResponse(BaseModel):
    """Response from geometry calculation."""
    request_id: str
    success: bool
    geometry_bundle_id: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.now)


class AsyncGeometryAgent:
    """
    Asynchronous geometry agent for background geometry calculations.
    
    Features:
    - Listens to Redis streams for geometry calculation requests
    - Processes macromolecule bundles using CctbxGeometryProcessor
    - Returns results via Redis streams
    - Implements proper error handling and retry logic
    - Supports agent security and permissions
    """
    
    def __init__(
        self,
        agent_id: str,
        redis_manager: RedisManager,
        stream_name: str = "geometry_requests",
        consumer_group: str = "geometry_agents",
        consumer_name: str = None,
        max_processing_time: int = 300,
        health_check_interval: int = 30,
    ):
        """
        Initialize the async geometry agent.
        
        Args:
            agent_id: Unique identifier for this agent
            redis_manager: Redis manager for bundle operations
            stream_name: Redis stream name for requests
            consumer_group: Consumer group name
            consumer_name: Consumer name (auto-generated if None)
            max_processing_time: Maximum processing time in seconds
            health_check_interval: Health check interval in seconds
        """
        self.agent_id = agent_id
        self.redis_manager = redis_manager
        self.stream_name = stream_name
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name or f"{agent_id}_{uuid.uuid4().hex[:8]}"
        self.max_processing_time = max_processing_time
        self.health_check_interval = health_check_interval
        
        # Initialize components
        self.geometry_processor = CctbxGeometryProcessor(redis_manager, f"{agent_id}_processor")
        self.redis_client = None
        self.is_running = False
        self.stats = {
            "requests_processed": 0,
            "requests_failed": 0,
            "total_processing_time": 0.0,
            "last_request_time": None,
        }
        
        # Security and configuration
        self.security_bundle: Optional[AgentSecurityBundle] = None
        self.config_bundle: Optional[AgentConfigurationBundle] = None
        self.streams_bundle: Optional[RedisStreamsBundle] = None
        
        self.logger = logging.getLogger(f"AsyncGeometryAgent.{agent_id}")
    
    async def initialize(self) -> None:
        """Initialize the agent and establish Redis connection."""
        try:
            # Create async Redis client
            self.redis_client = redis.Redis(
                host=self.redis_manager.host,
                port=self.redis_manager.port,
                db=self.redis_manager.db,
                password=self.redis_manager.password,
                decode_responses=True,
            )
            
            # Test connection
            await self.redis_client.ping()
            self.logger.info(f"Agent {self.agent_id} initialized successfully")
            
            # Load security and configuration bundles
            await self._load_agent_configuration()
            
            # Create consumer group if it doesn't exist
            await self._setup_consumer_group()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agent: {e}")
            raise
    
    async def _load_agent_configuration(self) -> None:
        """Load agent security and configuration bundles."""
        try:
            # Load security bundle
            security_bundle_id = f"{self.agent_id}_security"
            try:
                security_bundle = self.redis_manager.get_bundle(security_bundle_id)
                self.security_bundle = AgentSecurityBundle(**security_bundle.__dict__)
                self.logger.info(f"Loaded security bundle for agent {self.agent_id}")
            except KeyError:
                self.logger.warning(f"No security bundle found for agent {self.agent_id}")
            
            # Load configuration bundle
            config_bundle_id = f"{self.agent_id}_config"
            try:
                config_bundle = self.redis_manager.get_bundle(config_bundle_id)
                self.config_bundle = AgentConfigurationBundle(**config_bundle.__dict__)
                self.logger.info(f"Loaded configuration bundle for agent {self.agent_id}")
            except KeyError:
                self.logger.warning(f"No configuration bundle found for agent {self.agent_id}")
            
            # Load streams configuration
            streams_bundle_id = f"{self.agent_id}_streams"
            try:
                streams_bundle = self.redis_manager.get_bundle(streams_bundle_id)
                self.streams_bundle = RedisStreamsBundle(**streams_bundle.__dict__)
                self.logger.info(f"Loaded streams bundle for agent {self.agent_id}")
            except KeyError:
                self.logger.warning(f"No streams bundle found for agent {self.agent_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to load agent configuration: {e}")
    
    async def _setup_consumer_group(self) -> None:
        """Setup Redis consumer group for the stream."""
        try:
            # Create consumer group if it doesn't exist
            try:
                await self.redis_client.xgroup_create(
                    self.stream_name, 
                    self.consumer_group, 
                    id="0", 
                    mkstream=True
                )
                self.logger.info(f"Created consumer group {self.consumer_group}")
            except redis.ResponseError as e:
                if "BUSYGROUP" in str(e):
                    self.logger.info(f"Consumer group {self.consumer_group} already exists")
                else:
                    raise
            
        except Exception as e:
            self.logger.error(f"Failed to setup consumer group: {e}")
            raise
    
    async def start(self) -> None:
        """Start the agent and begin processing requests."""
        if self.is_running:
            self.logger.warning("Agent is already running")
            return
        
        self.is_running = True
        self.logger.info(f"Starting geometry agent {self.agent_id}")
        
        try:
            # Start health check task
            health_task = asyncio.create_task(self._health_check_loop())
            
            # Start main processing loop
            processing_task = asyncio.create_task(self._processing_loop())
            
            # Wait for both tasks
            await asyncio.gather(health_task, processing_task)
            
        except Exception as e:
            self.logger.error(f"Agent processing failed: {e}")
            raise
        finally:
            self.is_running = False
    
    async def stop(self) -> None:
        """Stop the agent gracefully."""
        self.logger.info(f"Stopping geometry agent {self.agent_id}")
        self.is_running = False
        
        if self.redis_client:
            await self.redis_client.close()
    
    async def _health_check_loop(self) -> None:
        """Periodic health check loop."""
        while self.is_running:
            try:
                # Check Redis connection
                await self.redis_client.ping()
                
                # Update agent status
                await self._update_agent_status()
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
                await asyncio.sleep(5)  # Shorter interval on error
    
    async def _update_agent_status(self) -> None:
        """Update agent status in Redis."""
        try:
            status = {
                "agent_id": self.agent_id,
                "status": "running" if self.is_running else "stopped",
                "last_heartbeat": datetime.now().isoformat(),
                "stats": json.dumps(self.stats),  # Serialize stats as JSON string
                "consumer_name": self.consumer_name,
            }
            
            await self.redis_client.hset(
                f"agentbx:agents:{self.agent_id}",
                mapping=status
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update agent status: {e}")
    
    async def _processing_loop(self) -> None:
        """Main processing loop for handling geometry requests."""
        while self.is_running:
            try:
                # Read messages from stream
                messages = await self.redis_client.xreadgroup(
                    self.consumer_group,
                    self.consumer_name,
                    {self.stream_name: ">"},
                    count=1,
                    block=1000  # 1 second timeout
                )
                
                if messages:
                    for stream, stream_messages in messages:
                        for message_id, fields in stream_messages:
                            # Process message asynchronously
                            asyncio.create_task(
                                self._process_message(message_id, fields)
                            )
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_message(self, message_id: str, fields: Dict[str, str]) -> None:
        """Process a single geometry request message."""
        start_time = time.time()
        
        try:
            # Parse request
            request_data = json.loads(fields.get("request", "{}"))
            request = GeometryRequest(**request_data)
            
            self.logger.info(f"Processing geometry request {request.request_id}")
            
            # Validate permissions
            if not await self._validate_permissions(request):
                response = GeometryResponse(
                    request_id=request.request_id,
                    success=False,
                    error_message="Insufficient permissions for geometry calculation",
                    processing_time=time.time() - start_time
                )
                await self._send_response(response)
                return
            
            # Process geometry calculation
            geometry_bundle_id = await self._calculate_geometry(request)
            
            # Create response
            response = GeometryResponse(
                request_id=request.request_id,
                success=True,
                geometry_bundle_id=geometry_bundle_id,
                processing_time=time.time() - start_time
            )
            
            # Update stats
            self.stats["requests_processed"] += 1
            self.stats["total_processing_time"] += response.processing_time
            self.stats["last_request_time"] = datetime.now().isoformat()
            
            await self._send_response(response)
            
        except Exception as e:
            self.logger.error(f"Failed to process message {message_id}: {e}")
            
            # Create error response
            response = GeometryResponse(
                request_id=fields.get("request_id", "unknown"),
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time
            )
            
            self.stats["requests_failed"] += 1
            await self._send_response(response)
            
        finally:
            # Acknowledge message
            try:
                await self.redis_client.xack(self.stream_name, self.consumer_group, message_id)
            except Exception as e:
                self.logger.error(f"Failed to acknowledge message {message_id}: {e}")
    
    async def _validate_permissions(self, request: GeometryRequest) -> bool:
        """Validate agent permissions for the request."""
        if not self.security_bundle:
            self.logger.warning("No security bundle loaded, allowing request")
            return True
        
        permissions = self.security_bundle.permissions
        if not permissions:
            self.logger.warning("No permissions defined, allowing request")
            return True
        
        required_permissions = ["geometry_calculation", "bundle_read", "bundle_write"]
        
        for permission in required_permissions:
            if permission not in permissions:
                self.logger.warning(f"Missing required permission: {permission}")
                return False
        
        return True
    
    async def _calculate_geometry(self, request: GeometryRequest) -> str:
        """Calculate geometry gradients for the request."""
        try:
            # Get macromolecule bundle
            macromolecule_bundle = self.redis_manager.get_bundle(
                request.macromolecule_bundle_id
            )
            
            # Process geometry calculation
            output_bundles = self.geometry_processor.process_bundles({
                "macromolecule_data": macromolecule_bundle
            })
            
            # Store result bundle
            geometry_bundle = output_bundles["geometry_gradient_data"]
            geometry_bundle_id = self.redis_manager.store_bundle(geometry_bundle)
            
            self.logger.info(f"Geometry calculation completed: {geometry_bundle_id}")
            return geometry_bundle_id
            
        except Exception as e:
            self.logger.error(f"Geometry calculation failed: {e}")
            raise
    
    async def _send_response(self, response: GeometryResponse) -> None:
        """Send response to the response stream."""
        try:
            response_stream = f"{self.stream_name}_responses"
            
            await self.redis_client.xadd(
                response_stream,
                {
                    "response": response.model_dump_json(),
                    "timestamp": datetime.now().isoformat(),
                    "agent_id": self.agent_id
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to send response: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            **self.stats,
            "agent_id": self.agent_id,
            "is_running": self.is_running,
            "consumer_name": self.consumer_name,
        } 