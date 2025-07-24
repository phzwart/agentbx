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
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Union
from typing import cast

import numpy as np
import redis.asyncio as redis
from pydantic import BaseModel
from pydantic import Field

from agentbx.core.bundle_base import Bundle
from agentbx.core.clients.array_translator import ArrayTranslator
from agentbx.core.processors.geometry_processor import CctbxGeometryProcessor
from agentbx.core.redis_manager import RedisManager
from agentbx.schemas.generated import AgentConfigurationBundle
from agentbx.schemas.generated import AgentSecurityBundle
from agentbx.schemas.generated import RedisStreamsBundle


# fmt: off
from cctbx.array_family import flex  # noqa: F401  # Needed for unpickling cctbx objects from Redis  # isort: skip
# fmt: on


@dataclass
class GeometryRequest:
    """Represents a geometry calculation request."""

    request_id: str
    macromolecule_bundle_id: str
    calculation_type: str = (
        "gradients_and_energy"  # "gradients_and_energy", "energy_only", "gradients_only"
    )
    priority: int = 1
    timeout_seconds: int = 300
    retry_count: int = 0
    max_retries: int = 3
    refresh_restraints: bool = False  # New flag to control restraint rebuilding
    created_at: Optional[datetime] = None

    def __post_init__(self):
        """Post-initialization hook."""
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


class GradientResultBundle(BaseModel):
    """Bundle for gradient result."""

    bundle_type: str = "gradient"
    gradient_key: str
    shape: list
    dtype: str
    parent_bundle_id: str
    step: int = 0
    timestamp: float = 0.0


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
        consumer_name: Optional[str] = None,
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
        self.geometry_processor = CctbxGeometryProcessor(
            redis_manager, f"{agent_id}_processor"
        )
        self.redis_client: Optional[redis.Redis] = None
        self.is_running = False
        self.stats: Dict[str, Union[int, float, str]] = {
            "requests_processed": 0,
            "requests_failed": 0,
            "total_processing_time": 0.0,
            "last_request_time": "",
        }

        # Security and configuration
        self.security_bundle: Optional[AgentSecurityBundle] = None
        self.config_bundle: Optional[AgentConfigurationBundle] = None
        self.streams_bundle: Optional[RedisStreamsBundle] = None

        # Add a cache for model_manager and restraint_manager per bundle_id
        self._model_manager_cache: Dict[str, Any] = {}
        self._restraint_manager_cache: Dict[str, Any] = {}
        self._asset_cache: Dict[str, dict] = {}  # Cache all assets per bundle_id

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
                self.logger.warning(
                    f"No security bundle found for agent {self.agent_id}"
                )

            # Load configuration bundle
            config_bundle_id = f"{self.agent_id}_config"
            try:
                config_bundle = self.redis_manager.get_bundle(config_bundle_id)
                self.config_bundle = AgentConfigurationBundle(**config_bundle.__dict__)
                self.logger.info(
                    f"Loaded configuration bundle for agent {self.agent_id}"
                )
            except KeyError:
                self.logger.warning(
                    f"No configuration bundle found for agent {self.agent_id}"
                )

            # Load streams configuration
            streams_bundle_id = f"{self.agent_id}_streams"
            try:
                streams_bundle = self.redis_manager.get_bundle(streams_bundle_id)
                self.streams_bundle = RedisStreamsBundle(**streams_bundle.__dict__)
                self.logger.info(f"Loaded streams bundle for agent {self.agent_id}")
            except KeyError:
                self.logger.warning(
                    f"No streams bundle found for agent {self.agent_id}"
                )

        except Exception as e:
            self.logger.error(f"Failed to load agent configuration: {e}")

    async def _setup_consumer_group(self) -> None:
        """Setup Redis consumer group for the stream."""
        try:
            # Create consumer group if it doesn't exist
            if self.redis_client is None:
                raise RuntimeError("redis_client is not initialized")
            try:
                await self.redis_client.xgroup_create(
                    self.stream_name, self.consumer_group, id="0", mkstream=True
                )
                self.logger.info(f"Created consumer group {self.consumer_group}")
            except redis.ResponseError as e:
                if "BUSYGROUP" in str(e):
                    self.logger.info(
                        f"Consumer group {self.consumer_group} already exists"
                    )
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
                # Update agent status
                await self._update_agent_status()

                # Log heartbeat
                self.logger.debug(
                    f"Heartbeat: agent_id={self.agent_id} is alive at {datetime.now().isoformat()}"
                )

                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(self.health_check_interval)

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

            if self.redis_client is None:
                raise RuntimeError("redis_client is not initialized")

            # Before calling hset, cast all values in status to str, int, or float as appropriate:
            def _cast_status_value(val):
                if isinstance(val, (str, int, float, bytes)):
                    return val
                try:
                    return float(val)
                except Exception:
                    return str(val)

            status_cast = {k: _cast_status_value(v) for k, v in status.items()}
            await self.redis_client.hset(
                f"agentbx:agents:{self.agent_id}",
                mapping=cast(
                    Mapping[Union[str, bytes], Union[bytes, float, int, str]],
                    status_cast,
                ),
            )

        except Exception as e:
            self.logger.error(f"Failed to update agent status: {e}")

    async def _processing_loop(self) -> None:
        while self.is_running:
            try:
                # Poll for messages from the geometry_requests stream
                if self.redis_client is None:
                    raise RuntimeError("redis_client is not initialized")
                # Use xreadgroup for immediate message pickup
                messages = await self.redis_client.xreadgroup(
                    groupname=self.consumer_group,
                    consumername=self.consumer_name,
                    streams={self.stream_name: ">"},
                    count=10,  # Read more messages to find high priority ones
                    block=1,  # 1ms for extremely fast response
                )
                if messages:
                    # Collect all messages and sort by priority
                    all_messages = []
                    for stream, message_list in messages:
                        for message_id, fields in message_list:
                            # Check if this is a HIGH priority message
                            priority = fields.get(b"priority", b"").decode("utf-8")
                            is_high_priority = priority == "HIGH"
                            all_messages.append((message_id, fields, is_high_priority))

                    # Sort messages: HIGH priority first, then normal priority
                    all_messages.sort(
                        key=lambda x: (not x[2], x[0])
                    )  # Sort by priority (HIGH first), then by message_id

                    # Process messages in priority order
                    for message_id, fields, is_high_priority in all_messages:
                        try:
                            await self._process_message(message_id, fields)
                            # Acknowledge the message
                            if self.redis_client is None:
                                raise RuntimeError(
                                    "redis_client is not initialized"
                                ) from None
                            await self.redis_client.xack(
                                self.stream_name, self.consumer_group, message_id
                            )
                        except Exception as e:
                            self.logger.error(
                                f"Error processing message {message_id}: {e}"
                            )
                            import traceback

                            self.logger.error(
                                f"Message processing error traceback: {traceback.format_exc()}"
                            )
                            # Still acknowledge to avoid infinite retry
                            if self.redis_client is None:
                                raise RuntimeError(
                                    "redis_client is not initialized"
                                ) from None
                            await self.redis_client.xack(
                                self.stream_name, self.consumer_group, message_id
                            )
                else:
                    # No messages found, continue immediately without waiting
                    await asyncio.sleep(0.001)  # 1ms sleep to prevent CPU spinning
            except redis.ConnectionError as e:
                self.logger.info(f"Redis connection closed during shutdown: {e}")
                break
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(1)

    async def _process_message(self, message_id: str, fields: Dict[str, str]) -> None:
        """Process a single message from the stream."""
        try:
            # Debug: Log what message we received
            self.logger.info(
                f"Processing message {message_id} with fields: {list(fields.keys())}"
            )

            # Check if this is a coordinate update or geometry request
            if "coordinate_update" in fields:
                await self._process_coordinate_update(message_id, fields)
            elif "request" in fields:
                await self._process_geometry_request(message_id, fields)
            else:
                self.logger.warning(f"Unknown message type: {list(fields.keys())}")

        except Exception as e:
            self.logger.error(f"Error processing message {message_id}: {e}")
            import traceback

            self.logger.error(
                f"Message processing error traceback: {traceback.format_exc()}"
            )

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
        bundle_id = request.macromolecule_bundle_id
        if bundle_id in self._model_manager_cache:
            model_manager = self._model_manager_cache[bundle_id]
            restraint_manager = self._restraint_manager_cache.get(bundle_id)
            asset_dict = self._asset_cache[bundle_id]
        else:
            macromolecule_bundle = self.redis_manager.get_bundle(bundle_id)
            model_manager = macromolecule_bundle.get_asset("model_manager")
            model_manager.process(make_restraints=True)
            restraint_manager = model_manager.get_restraints_manager()
            self._model_manager_cache[bundle_id] = model_manager
            self._restraint_manager_cache[bundle_id] = restraint_manager
            asset_dict = dict(macromolecule_bundle.assets)
            self._asset_cache[bundle_id] = asset_dict
        # Always use cached restraint manager - never refresh after initial creation
        macromolecule_data = Bundle(bundle_type="macromolecule_data")
        for key, value in asset_dict.items():
            macromolecule_data.add_asset(key, value)
        macromolecule_data.add_asset("model_manager", model_manager)

        # Handle different calculation types
        if request.calculation_type == "energy_only":
            # Only calculate energy (for LBFGS closure)
            energy_bundle = await self._calculate_energy_only(
                model_manager, restraint_manager, bundle_id
            )
            return energy_bundle
        elif request.calculation_type == "gradients_only":
            # Only calculate gradients (for standard optimizers)
            gradients_bundle = await self._calculate_gradients_only(
                model_manager, restraint_manager, bundle_id
            )
            return gradients_bundle
        else:
            # Calculate both gradients and energy (default behavior)
            output_bundles = self.geometry_processor.process_bundles_with_refresh(
                {"macromolecule_data": macromolecule_data}, refresh_restraints=False
            )
            # Get the geometry bundle from the output
            geometry_bundle = output_bundles["geometry_gradients"]

            # Extract gradients as CCTBX flex array
            gradients = geometry_bundle.get_asset("geometry_gradients")
            gradient_norm = geometry_bundle.get_asset("gradient_norm")
            total_geometry_energy = geometry_bundle.get_asset("total_geometry_energy")
            print(
                f"[AsyncGeometryAgent] total_energy={total_geometry_energy:.6f} gradient_norm={gradient_norm:.6f}"
            )

        # Handle tuple return (gradients, energy) from geometry processor
        if isinstance(gradients, tuple):
            gradients, _ = gradients

        # Use ArrayTranslator for dialect-aware conversion
        translator = ArrayTranslator()

        # Convert to numpy for storage
        numpy_gradients = translator.convert(gradients, "numpy")

        # Pack for bundle storage
        grad_bytes, grad_metadata = translator.pack_for_bundle(
            numpy_gradients, "numpy", "bytes"
        )

        # Store as a true Bundle
        grad_bundle = Bundle(bundle_type="geometry_gradient_data")
        grad_bundle.add_asset("geometry_gradients", grad_bytes)
        grad_bundle.add_asset("shape", grad_metadata["shape"])
        grad_bundle.add_asset("dtype", grad_metadata["dtype"])
        grad_bundle.add_asset("parent_bundle_id", bundle_id)
        grad_bundle.add_asset(
            "total_geometry_energy", total_geometry_energy
        )  # Copy the energy!
        grad_bundle.add_metadata("dialect", "numpy")
        grad_bundle.add_metadata("step", getattr(request, "step", 0))
        grad_bundle.add_metadata("timestamp", time.time())
        grad_bundle_id = self.redis_manager.store_bundle(grad_bundle)
        self.logger.info(f"Gradient result stored as bundle {grad_bundle_id}")

        return grad_bundle_id

    async def _calculate_energy_only(
        self, model_manager, restraint_manager, bundle_id: str
    ) -> str:
        """Calculate only the geometry energy (for LBFGS closure)."""
        try:
            # Get current coordinates
            sites_cart = model_manager.get_sites_cart()

            # Calculate only energy (no gradients)
            if restraint_manager is not None:
                geometry_restraints = restraint_manager.geometry
                call_args = {"sites_cart": sites_cart, "compute_gradients": False}
                energies = geometry_restraints.energies_sites(**call_args)
                total_geometry_energy = energies.target
            else:
                # Fallback: use model_manager's energy calculation
                total_geometry_energy = model_manager.get_geometry_energy()

            # Create energy-only bundle
            energy_bundle = Bundle(bundle_type="geometry_energy_data")
            energy_bundle.add_asset("total_geometry_energy", total_geometry_energy)
            energy_bundle.add_asset("parent_bundle_id", bundle_id)
            energy_bundle.add_metadata("calculation_type", "energy_only")
            energy_bundle.add_metadata("timestamp", time.time())

            energy_bundle_id = self.redis_manager.store_bundle(energy_bundle)
            self.logger.info(
                f"Energy-only calculation stored as bundle {energy_bundle_id}"
            )

            return energy_bundle_id
        except Exception as err:
            self.logger.error(f"Failed to calculate energy only: {err}")
            raise err from err

    async def _calculate_gradients_only(
        self, model_manager, restraint_manager, bundle_id: str
    ) -> str:
        """Calculate only the geometry gradients (for standard optimizers)."""
        try:
            # Get current coordinates
            sites_cart = model_manager.get_sites_cart()

            # Calculate gradients (energy will be computed as side effect)
            if restraint_manager is not None:
                geometry_restraints = restraint_manager.geometry
                call_args = {"sites_cart": sites_cart, "compute_gradients": True}
                energies_and_gradients = geometry_restraints.energies_sites(**call_args)
                coordinate_gradients = energies_and_gradients.gradients
                total_geometry_energy = energies_and_gradients.target
            else:
                # Fallback: use model_manager's gradient calculation
                coordinate_gradients = model_manager.get_geometry_gradients()
                total_geometry_energy = model_manager.get_geometry_energy()

            # Use ArrayTranslator for dialect-aware conversion
            translator = ArrayTranslator()
            numpy_gradients = translator.convert(coordinate_gradients, "numpy")

            # Pack for bundle storage
            grad_bytes, grad_metadata = translator.pack_for_bundle(
                numpy_gradients, "numpy", "bytes"
            )

            # Create gradients-only bundle
            grad_bundle = Bundle(bundle_type="geometry_gradient_data")
            grad_bundle.add_asset("geometry_gradients", grad_bytes)
            grad_bundle.add_asset("shape", grad_metadata["shape"])
            grad_bundle.add_asset("dtype", grad_metadata["dtype"])
            grad_bundle.add_asset("parent_bundle_id", bundle_id)
            grad_bundle.add_asset("total_geometry_energy", total_geometry_energy)
            grad_bundle.add_metadata("dialect", "numpy")
            grad_bundle.add_metadata("calculation_type", "gradients_only")
            grad_bundle.add_metadata("timestamp", time.time())

            grad_bundle_id = self.redis_manager.store_bundle(grad_bundle)
            self.logger.info(
                f"Gradients-only calculation stored as bundle {grad_bundle_id}"
            )

            return grad_bundle_id
        except Exception as err:
            self.logger.error(f"Failed to calculate gradients only: {err}")
            raise err from err

    async def _send_response(self, response: GeometryResponse) -> None:
        """Send response to the response stream."""
        try:
            # Add debugging for bundle ID
            if response.geometry_bundle_id:
                self.logger.info(
                    f"Sending response with bundle ID: {response.geometry_bundle_id}"
                )
                self.logger.info(f"Response JSON: {response.model_dump_json()}")

            # Send response to response stream
            response_stream = f"{self.stream_name}_responses"
            if self.redis_client is None:
                raise RuntimeError("redis_client is not initialized")
            await self.redis_client.xadd(
                response_stream,
                {
                    "response": response.model_dump_json(),
                    "timestamp": datetime.now().isoformat(),
                    "agent_id": self.agent_id,
                },
            )
        except Exception as e:
            self.logger.error(f"Failed to send response: {e}")

    async def _process_coordinate_update(
        self, message_id: str, fields: Dict[str, str]
    ) -> None:
        """Process a coordinate update message."""
        try:
            # Parse coordinate update message
            coord_update_data = json.loads(fields.get("coordinate_update", "{}"))
            bundle_id = coord_update_data.get("bundle_id")
            parent_bundle_id = coord_update_data.get("parent_bundle_id")

            if bundle_id and parent_bundle_id in self._model_manager_cache:
                # Get the coordinate update bundle
                coordinate_update_bundle = self.redis_manager.get_bundle(bundle_id)

                # Get the model manager
                model_manager = self._model_manager_cache[parent_bundle_id]

                # Convert coordinates from list to CCTBX flex format
                # import numpy as np
                # from cctbx.array_family import flex

                # Convert list to numpy array
                coords_numpy = np.array(coordinate_update_bundle.coordinates)

                # Use ArrayTranslator to convert to CCTBX flex format
                translator = ArrayTranslator()
                coords_flex = translator.convert(coords_numpy, "cctbx")

                # Set coordinates in model_manager
                model_manager.set_sites_cart(coords_flex)
                self.logger.info(f"Updated coordinates for bundle {parent_bundle_id}")

            # Optionally, delete the bundle after processing
            self.redis_manager.delete_bundle(bundle_id)

        except Exception as e:
            self.logger.warning(f"Error processing coordinate update {message_id}: {e}")
            import traceback

            self.logger.warning(
                f"Coordinate update error traceback: {traceback.format_exc()}"
            )

    async def _process_geometry_request(
        self, message_id: str, fields: Dict[str, str]
    ) -> None:
        """Process a geometry calculation request."""
        start_time = time.time()

        try:
            # Parse request
            request_data = json.loads(fields.get("request", "{}"))
            request = GeometryRequest(**request_data)

            self.logger.info(
                f"Processing geometry request {request.request_id} (type: {request.calculation_type})"
            )

            # Validate permissions
            if not await self._validate_permissions(request):
                response = GeometryResponse(
                    request_id=request.request_id,
                    success=False,
                    error_message="Insufficient permissions for geometry calculation",
                    processing_time=time.time() - start_time,
                )
                await self._send_response(response)
                return

            # Send acknowledgment for all geometry requests
            result_key = fields.get("result_key")
            if result_key:
                # Send acknowledgment that we're processing this request
                ack_key = f"ack:{result_key}"
                ack_data = {
                    "status": "processing",
                    "request_id": request.request_id,
                    "calculation_type": request.calculation_type,
                    "timestamp": time.time(),
                }
                if self.redis_client is None:
                    raise RuntimeError("redis_client is not initialized")
                await self.redis_client.setex(ack_key, 10, json.dumps(ack_data))

            # Process geometry calculation
            geometry_bundle_id = await self._calculate_geometry(request)

            # Check if this is a UUID key request (for LBFGS closure)
            if result_key and request.calculation_type == "energy_only":
                # Send acknowledgment that we're processing this request
                ack_key = f"ack:{result_key}"
                ack_data = {
                    "status": "processing",
                    "request_id": request.request_id,
                    "timestamp": time.time(),
                }
                if self.redis_client is None:
                    raise RuntimeError("redis_client is not initialized")
                await self.redis_client.setex(ack_key, 10, json.dumps(ack_data))

                # Write energy directly to UUID key for fast LBFGS access
                try:
                    energy_bundle = self.redis_manager.get_bundle(geometry_bundle_id)
                    energy_value = energy_bundle.get_asset("total_geometry_energy")

                    # Write result directly to the UUID key
                    result_data = {
                        "success": True,
                        "energy": energy_value,
                        "request_id": request.request_id,
                        "processing_time": time.time() - start_time,
                    }

                    # Write to Redis key with 10 second expiration
                    await self.redis_client.setex(
                        result_key, 10, json.dumps(result_data)  # 10 second expiration
                    )

                    self.logger.info(
                        f"Energy result written to key {result_key}: {energy_value}"
                    )

                except Exception as e:
                    # Write error to UUID key
                    error_data = {
                        "success": False,
                        "error_message": str(e),
                        "request_id": request.request_id,
                        "processing_time": time.time() - start_time,
                    }
                    if self.redis_client is None:
                        raise RuntimeError("redis_client is not initialized") from e
                    await self.redis_client.setex(
                        result_key, 10, json.dumps(error_data)
                    )

                # Don't send stream response for UUID key requests
                self.stats["requests_processed"] = (
                    int(self.stats.get("requests_processed", 0)) + 1
                )
                return

            # Standard stream response (for non-UUID requests)
            response = GeometryResponse(
                request_id=request.request_id,
                success=True,
                geometry_bundle_id=geometry_bundle_id,
                processing_time=time.time() - start_time,
            )

            # Update stats
            self.stats["requests_processed"] = (
                int(self.stats.get("requests_processed", 0)) + 1
            )
            self.stats["total_processing_time"] = (
                float(self.stats.get("total_processing_time", 0.0))
                + response.processing_time
            )
            self.stats["last_request_time"] = datetime.now().isoformat()

            await self._send_response(response)

        except Exception as e:
            self.logger.error(f"Error processing geometry request {message_id}: {e}")
            import traceback

            self.logger.error(
                f"Geometry request error traceback: {traceback.format_exc()}"
            )

            # Create error response
            response = GeometryResponse(
                request_id=fields.get("request_id", "unknown"),
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time,
            )

            self.stats["requests_failed"] = (
                int(self.stats.get("requests_failed", 0)) + 1
            )
            await self._send_response(response)
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            **self.stats,
            "agent_id": self.agent_id,
            "is_running": self.is_running,
            "consumer_name": self.consumer_name,
        }
