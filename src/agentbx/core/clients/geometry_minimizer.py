"""
Geometry Minimizer Module

A PyTorch-style geometry minimizer that integrates with Redis bundles and
the async geometry agent system for coordinate optimization.
"""

import asyncio
import dataclasses
import logging
import time
import uuid

# Import for type hints only
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn

from agentbx.core.clients.coordinate_translator import CoordinateTranslator
from agentbx.core.processors.macromolecule_processor import MacromoleculeProcessor
from agentbx.core.redis_manager import RedisManager


# fmt: off
from cctbx.array_family import flex  # noqa: F401  # Needed for unpickling cctbx objects from Redis  # isort: skip
# fmt: on


if TYPE_CHECKING:
    pass

import json

import numpy as np
import redis.asyncio as redis

from agentbx.core.clients.array_translator import ArrayTranslator


class GeometryMinimizer(nn.Module):
    """
    PyTorch-style geometry minimizer with Redis bundle integration.

    Features:
    - Async geometry gradient calculations via Redis streams
    - Coordinate updates in macromolecule bundles
    - Multiple optimization algorithms (GD, Adam)
    - Convergence monitoring
    - Bundle registration and tracking
    """

    def __init__(
        self,
        redis_manager: RedisManager,
        macromolecule_bundle_id: str,
        optimizer_factory: Any,  # Callable, required
        optimizer_kwargs: dict,
        scheduler_factory: Optional[Any] = None,  # Optional callable
        scheduler_kwargs: Optional[Dict[Any, Any]] = None,  # Optional dict
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6,
        timeout_seconds: float = 30.0,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        # Stream configuration parameters
        request_stream_name: str = "geometry_requests",
        response_stream_name: str = "geometry_requests_responses",
        consumer_group: str = "minimizer_consumer",
        consumer_name: Optional[str] = None,
    ):
        """
        Initialize the async geometry minimizer.

        NOTE: This is an ASYNC-ONLY implementation. LBFGS is NOT supported due to
        incompatibility with the async architecture. Use a synchronous geometry
        minimizer for LBFGS optimization.

        Args:
            redis_manager: Redis manager for bundle operations
            macromolecule_bundle_id: ID of the macromolecule bundle to minimize
            optimizer_factory: Callable that returns a PyTorch optimizer (EXCEPT LBFGS)
            optimizer_kwargs: Dict of kwargs for the optimizer
            scheduler_factory: Optional callable that returns a PyTorch scheduler
            scheduler_kwargs: Optional dict of kwargs for the scheduler
            max_iterations: Maximum number of iterations
            convergence_threshold: Gradient norm threshold for convergence
            timeout_seconds: Timeout for geometry calculations
            device: Device for tensors
            dtype: Data type for tensors
            request_stream_name: Redis stream name for sending requests
            response_stream_name: Redis stream name for receiving responses
            consumer_group: Consumer group name for response handling
            consumer_name: Consumer name (auto-generated if None)

        Raises:
            ValueError: If optimizer_factory is torch.optim.LBFGS
        """
        super().__init__()

        # Reject LBFGS explicitly
        if optimizer_factory == torch.optim.LBFGS:
            raise ValueError(
                "LBFGS optimizer is not supported in the async geometry minimizer. "
                "LBFGS requires synchronous closures which are incompatible with "
                "the async Redis-based architecture. Please use a different optimizer "
                "(e.g., Adam, SGD) or use a synchronous geometry minimizer for LBFGS."
            )

        self.redis_manager = redis_manager
        self.macromolecule_bundle_id = macromolecule_bundle_id
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.timeout_seconds = timeout_seconds
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.request_stream_name = request_stream_name
        self.response_stream_name = response_stream_name
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name or f"minimizer_{uuid.uuid4().hex[:8]}"
        self.logger = logging.getLogger("GeometryMinimizer")
        self.coordinate_translator = CoordinateTranslator(
            redis_manager=redis_manager,
            coordinate_system="cartesian",
            requires_grad=True,
            dtype=self.dtype,
            device=self.device,
        )
        self.macromolecule_processor = MacromoleculeProcessor(
            redis_manager, "minimizer_processor"
        )
        self.initial_coordinates = self._load_coordinates_from_bundle()
        self.current_coordinates = (
            self.initial_coordinates.detach().clone().requires_grad_(True)
        )
        self.optimizer = optimizer_factory(
            [self.current_coordinates], **optimizer_kwargs
        )
        self.scheduler = None
        if scheduler_factory is not None:
            if scheduler_kwargs is None:
                scheduler_kwargs = {}
            self.scheduler = scheduler_factory(self.optimizer, **scheduler_kwargs)
        self.iteration_history: List[Dict[str, float]] = []
        self.best_coordinates: Optional[torch.Tensor] = None
        self.best_gradient_norm = float("inf")
        self.latest_total_energy = None
        self.async_redis_client = redis.Redis(
            host=self.redis_manager.host,
            port=self.redis_manager.port,
            db=self.redis_manager.db,
            password=self.redis_manager.password,
            decode_responses=False,
        )

    def add_scheduler(self, scheduler: Any) -> None:
        """
        Attach a PyTorch learning rate scheduler to this minimizer.

        Args:
            scheduler: A PyTorch scheduler instance (e.g., CosineAnnealingLR)
        """
        self.scheduler = scheduler

    def _load_coordinates_from_bundle(self) -> torch.Tensor:
        """Load coordinates from the macromolecule bundle."""
        try:
            # Get macromolecule bundle
            macromolecule_bundle = self.redis_manager.get_bundle(
                self.macromolecule_bundle_id
            )

            # Initialize dialect variable
            dialect = "cctbx"  # Default dialect for fallback case

            # Check if bundle has coordinates as a separate asset (numpy dialect)
            if macromolecule_bundle.has_asset("coordinates"):
                # Load numpy coordinates
                coordinates_asset = macromolecule_bundle.get_asset("coordinates")
                dialect = macromolecule_bundle.get_metadata("dialect", "numpy")

                # Use ArrayTranslator for conversion
                translator = ArrayTranslator(
                    default_dtype=np.dtype("float64"), default_device=self.device
                )

                if dialect == "numpy":
                    # Coordinates are stored as numpy array
                    coordinates_tensor = translator.convert(
                        coordinates_asset, "torch", requires_grad=True
                    )
                elif dialect == "cctbx":
                    # Coordinates are stored as CCTBX flex array
                    coordinates_tensor = translator.convert(
                        coordinates_asset, "torch", requires_grad=True
                    )
                else:
                    raise ValueError(f"Unsupported coordinate dialect: {dialect}")

            else:
                # Fallback: get coordinates from xray_structure (CCTBX dialect)
                dialect = "cctbx"
                xray_structure = macromolecule_bundle.get_asset("xray_structure")
                sites_cart = xray_structure.sites_cart()

                # Use ArrayTranslator for CCTBX to torch conversion
                translator = ArrayTranslator(
                    default_dtype=np.dtype("float64"), default_device=self.device
                )
                coordinates_tensor = translator.convert(
                    sites_cart, "torch", requires_grad=True
                )

            self.logger.info(
                f"Loaded {coordinates_tensor.shape[0]} coordinates from bundle (dialect: {dialect})"
            )
            return coordinates_tensor

        except Exception as e:
            self.logger.error(f"Failed to load coordinates: {e}")
            raise

    async def _request_geometry_calculation(
        self,
        refresh_restraints: bool = False,
        calculation_type: str = "gradients_and_energy",
    ) -> str:
        """Request geometry calculation from the async agent."""
        try:
            # Import GeometryRequest here to avoid circular import
            from agentbx.core.agents.async_geometry_agent import GeometryRequest

            request = GeometryRequest(
                request_id=str(uuid.uuid4()),
                macromolecule_bundle_id=self.macromolecule_bundle_id,
                calculation_type=calculation_type,
                priority=1,
                refresh_restraints=refresh_restraints,
            )

            # Create a unique result key for acknowledgment tracking
            result_key = f"geometry_result:{request.request_id}"

            # Send request to Redis stream with result key
            message = {
                "request": json.dumps(dataclasses.asdict(request), default=str),
                "result_key": result_key,  # Enable acknowledgment tracking
                "timestamp": time.time(),
                "source": "geometry_minimizer",
            }
            await self.async_redis_client.xadd(self.request_stream_name, message)
            refresh_status = (
                "with full restraint refresh"
                if refresh_restraints
                else "using existing restraints"
            )
            self.logger.info(
                f"Geometry calculation request sent ({refresh_status}, type: {calculation_type})"
            )

            # Wait for acknowledgment (optional - for debugging)
            ack_key = f"ack:{result_key}"
            ack_start_time = time.time()
            ack_timeout = 1.0  # 1 second timeout for acknowledgment

            while time.time() - ack_start_time < ack_timeout:
                ack_data = await self.async_redis_client.get(ack_key)
                if ack_data:
                    break
                await asyncio.sleep(0.01)

            # Wait for response
            response_bundle_id = await self._wait_for_geometry_response()
            return response_bundle_id
        except Exception as e:
            self.logger.error(f"Geometry calculation request failed: {e}")
            raise

    async def _request_energy_calculation(
        self, refresh_restraints: bool = False
    ) -> str:
        """Request energy-only calculation (for LBFGS closure)."""
        return await self._request_geometry_calculation(
            refresh_restraints=refresh_restraints, calculation_type="energy_only"
        )

    async def _request_gradients_calculation(
        self, refresh_restraints: bool = False
    ) -> str:
        """Request gradients-only calculation (for standard optimizers)."""
        return await self._request_geometry_calculation(
            refresh_restraints=refresh_restraints, calculation_type="gradients_only"
        )

    async def _wait_for_geometry_response_thread(self, thread_redis_client) -> str:
        """Wait for geometry calculation response using a thread-specific Redis client."""
        try:
            # Create consumer group if it doesn't exist
            try:
                await thread_redis_client.xgroup_create(
                    self.response_stream_name, self.consumer_group, mkstream=True
                )
            except Exception:
                # Group already exists
                pass

            # Get the current timestamp to filter out old messages
            current_time = time.time()
            max_age_seconds = 30  # Ignore messages older than 30 seconds

            # Read from response stream
            start_time = time.time()
            while time.time() - start_time < self.timeout_seconds:
                try:
                    # Read messages from the stream - use ">" to get only new messages
                    messages = await thread_redis_client.xreadgroup(
                        self.consumer_group,
                        self.consumer_name,
                        {self.response_stream_name: ">"},
                        count=10,  # Read more messages to find the latest
                        block=1000,
                    )
                    if messages:
                        for stream, message_list in messages:
                            # Sort messages by timestamp to get the latest
                            valid_messages = []
                            for message_id, fields in message_list:
                                # Check message timestamp
                                timestamp_str = fields.get(b"timestamp", b"").decode(
                                    "utf-8"
                                )
                                try:
                                    # Parse timestamp and check if it's recent
                                    from datetime import datetime

                                    msg_time = datetime.fromisoformat(
                                        timestamp_str.replace("Z", "+00:00")
                                    )
                                    msg_timestamp = msg_time.timestamp()
                                    if current_time - msg_timestamp < max_age_seconds:
                                        valid_messages.append(
                                            (message_id, fields, msg_timestamp)
                                        )
                                except Exception:
                                    # If timestamp parsing fails, assume it's recent
                                    valid_messages.append(
                                        (message_id, fields, current_time)
                                    )

                            # Sort by timestamp (newest first) and process the latest
                            if valid_messages:
                                valid_messages.sort(key=lambda x: x[2], reverse=True)
                                message_id, fields, _ = valid_messages[0]

                                # Parse response
                                response_data = fields.get(b"response", b"{}")
                                if isinstance(response_data, bytes):
                                    response_data = response_data.decode("utf-8")
                                response_dict = json.loads(response_data)
                                bundle_id = response_dict.get("geometry_bundle_id")
                                if bundle_id:
                                    # Acknowledge message
                                    await thread_redis_client.xack(
                                        self.response_stream_name,
                                        self.consumer_group,
                                        message_id,
                                    )

                                    return bundle_id

                                # Acknowledge other messages to clear them from the stream
                                for msg_id, _, _ in valid_messages[1:]:
                                    await thread_redis_client.xack(
                                        self.response_stream_name,
                                        self.consumer_group,
                                        msg_id,
                                    )
                    await asyncio.sleep(0.1)
                except Exception as e:
                    self.logger.warning(f"Error reading response stream (thread): {e}")
                    await asyncio.sleep(0.1)
            raise TimeoutError(
                f"Geometry calculation timed out after {self.timeout_seconds} seconds"
            )
        except Exception as e:
            self.logger.error(f"Failed to wait for geometry response (thread): {e}")
            raise

    async def _wait_for_geometry_response(self) -> str:
        """Wait for geometry calculation response."""
        try:
            # Create consumer group if it doesn't exist
            try:
                await self.async_redis_client.xgroup_create(
                    self.response_stream_name, self.consumer_group, mkstream=True
                )
            except Exception:
                # Group already exists
                pass

            # Get the current timestamp to filter out old messages
            current_time = time.time()
            max_age_seconds = 30  # Ignore messages older than 30 seconds

            # Read from response stream
            start_time = time.time()
            while time.time() - start_time < self.timeout_seconds:
                try:
                    # Read messages from the stream - use ">" to get only new messages
                    messages = await self.async_redis_client.xreadgroup(
                        self.consumer_group,
                        self.consumer_name,
                        {self.response_stream_name: ">"},
                        count=10,  # Read more messages to find the latest
                        block=1000,
                    )
                    if messages:
                        for stream, message_list in messages:
                            # Sort messages by timestamp to get the latest
                            valid_messages = []
                            for message_id, fields in message_list:
                                # Check message timestamp
                                timestamp_str = fields.get(b"timestamp", b"").decode(
                                    "utf-8"
                                )
                                try:
                                    # Parse timestamp and check if it's recent
                                    from datetime import datetime

                                    msg_time = datetime.fromisoformat(
                                        timestamp_str.replace("Z", "+00:00")
                                    )
                                    msg_timestamp = msg_time.timestamp()
                                    if current_time - msg_timestamp < max_age_seconds:
                                        valid_messages.append(
                                            (message_id, fields, msg_timestamp)
                                        )
                                except Exception:
                                    # If timestamp parsing fails, assume it's recent
                                    valid_messages.append(
                                        (message_id, fields, current_time)
                                    )

                            # Sort by timestamp (newest first) and process the latest
                            if valid_messages:
                                valid_messages.sort(key=lambda x: x[2], reverse=True)
                                message_id, fields, _ = valid_messages[0]

                                # Parse response
                                response_data = fields.get(b"response", b"{}")
                                if isinstance(response_data, bytes):
                                    response_data = response_data.decode("utf-8")
                                response_dict = json.loads(response_data)
                                bundle_id = response_dict.get("geometry_bundle_id")
                                if bundle_id:
                                    # Acknowledge message
                                    await self.async_redis_client.xack(
                                        self.response_stream_name,
                                        self.consumer_group,
                                        message_id,
                                    )

                                    return bundle_id

                                # Acknowledge other messages to clear them from the stream
                                for msg_id, _, _ in valid_messages[1:]:
                                    await self.async_redis_client.xack(
                                        self.response_stream_name,
                                        self.consumer_group,
                                        msg_id,
                                    )
                    await asyncio.sleep(0.1)
                except Exception as e:
                    self.logger.warning(f"Error reading response stream: {e}")
                    await asyncio.sleep(0.1)
            raise TimeoutError(
                f"Geometry calculation timed out after {self.timeout_seconds} seconds"
            )
        except Exception as e:
            self.logger.error(f"Failed to wait for geometry response: {e}")
            raise

    def _update_coordinates_in_bundle(self, new_coordinates: torch.Tensor) -> str:
        """Update coordinates in the macromolecule bundle."""
        try:
            # Use ArrayTranslator for torch to CCTBX conversion
            translator = ArrayTranslator(
                default_dtype=np.dtype("float64"), default_device=self.device
            )
            cctbx_coordinates = translator.convert(new_coordinates, "cctbx")

            # Update coordinates in macromolecule bundle
            updated_bundle_id = self.macromolecule_processor.update_coordinates(
                self.macromolecule_bundle_id, cctbx_coordinates
            )

            # Update our reference to the new bundle ID
            self.macromolecule_bundle_id = updated_bundle_id

            self.logger.info(f"Updated coordinates in bundle: {updated_bundle_id}")
            return updated_bundle_id

        except Exception as e:
            self.logger.error(f"Failed to update coordinates in bundle: {e}")
            raise

    async def forward(
        self, refresh_restraints: bool = False
    ) -> Tuple[torch.Tensor, str]:
        """
        Forward pass: request geometry calculation and return gradients.

        Args:
            refresh_restraints: If True, rebuild restraints on each iteration

        Returns:
            Tuple of (gradient tensor, geometry gradient bundle ID)

        Raises:
            ValueError: If the gradient bundle dialect is not supported.
            Exception: If bundle retrieval fails after retries.
        """
        # Always get both gradients and energy for logging and convergence purposes
        calculation_type = "gradients_and_energy"

        # Request geometry calculation, get result bundle key
        result_bundle_key = await self._request_geometry_calculation(
            refresh_restraints=refresh_restraints, calculation_type=calculation_type
        )

        # Add retry logic for bundle retrieval
        max_retries = 5
        retry_delay = 0.1

        for attempt in range(max_retries):
            try:
                # Load result bundle from Redis as a true Bundle
                grad_bundle = self.redis_manager.get_bundle(result_bundle_key)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(
                        f"Failed to retrieve bundle {result_bundle_key} after {max_retries} attempts: {e}"
                    )
                    raise
                else:
                    self.logger.warning(
                        f"Bundle {result_bundle_key} not found, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff

        dialect = grad_bundle.get_metadata("dialect")

        # Use ArrayTranslator for dialect-aware unpacking
        translator = ArrayTranslator(
            default_dtype=np.dtype("float64"), default_device=self.device
        )

        if dialect == "numpy":
            # Unpack numpy array from bundle
            grad_bytes = grad_bundle.get_asset("geometry_gradients")
            grad_shape = grad_bundle.get_asset("shape")
            grad_dtype = grad_bundle.get_asset("dtype")
            metadata = {
                "dialect": "numpy",
                "shape": grad_shape,
                "dtype": grad_dtype,
                "storage_format": "bytes",
            }
            numpy_gradients = translator.unpack_from_bundle(grad_bytes, metadata)
            gradients_tensor = translator.convert(
                numpy_gradients, "torch", requires_grad=False
            )

        elif dialect == "cctbx":
            # Unpack CCTBX flex array from bundle
            cctbx_gradients = grad_bundle.get_asset("geometry_gradients")
            gradients_tensor = translator.convert(
                cctbx_gradients, "torch", requires_grad=False
            )

        else:
            raise ValueError(f"Unsupported gradient bundle dialect: {dialect}")

        # Extract and store the energy value for LBFGS closure
        if grad_bundle.has_asset("total_geometry_energy"):
            self._current_energy = grad_bundle.get_asset("total_geometry_energy")
        else:
            self._current_energy = None
            self.logger.warning("No total_geometry_energy found in gradient bundle")

        return gradients_tensor, result_bundle_key

    async def backward(self, gradients: torch.Tensor, step: int = 0) -> None:
        """
        Backward pass: update coordinates using gradients and send coordinate update bundle to Redis.

        NOTE: This method only supports standard optimizers (Adam, SGD, etc.) that don't
        require closure functions. LBFGS is not supported in this async implementation.

        Args:
            gradients: Gradient tensor
            step: Current optimization step

        """
        # Zero gradients
        self.optimizer.zero_grad()
        # Set gradients (positive for minimization - move opposite to gradient)
        self.current_coordinates.grad = gradients

        # Standard optimizers (Adam, SGD, etc.) - async compatible
        self.optimizer.step()

        # Convert coordinates to numpy list format for coordinate update bundle
        translator = ArrayTranslator(
            default_dtype=np.dtype("float64"), default_device=self.device
        )
        coordinates_numpy = translator.convert(
            self.current_coordinates.detach(), "numpy"
        )
        coordinates_list = coordinates_numpy.tolist()

        # Create coordinate update bundle
        from agentbx.schemas.generated import CoordinateUpdateBundle

        coordinate_update_bundle = CoordinateUpdateBundle(
            coordinates=coordinates_list,
            parent_bundle_id=self.macromolecule_bundle_id,
            step=step,
            timestamp=time.time(),
            dialect="numpy",
        )

        # Store the bundle and send bundle ID as message
        bundle_id = self.redis_manager.store_bundle(coordinate_update_bundle)

        # Send bundle ID as a message on the geometry_requests stream
        coordinate_update_message = {
            "type": "coordinate_update",
            "bundle_id": bundle_id,
            "parent_bundle_id": self.macromolecule_bundle_id,
            "step": step,
            "timestamp": time.time(),
            "dialect": "numpy",
        }

        stream_name = self.request_stream_name
        message = {
            "coordinate_update": json.dumps(coordinate_update_message, default=str),
            "timestamp": time.time(),
            "source": "geometry_minimizer",
        }
        await self.async_redis_client.xadd(stream_name, message)

    async def minimize(self, refresh_restraints: bool = False) -> Dict[str, Any]:
        """
        Minimize the geometry of the macromolecule bundle.

        Args:
            refresh_restraints: If True, rebuild restraints on each iteration

        Returns:
            Dictionary containing minimization results
        """
        self.logger.info(
            f"Starting geometry minimization with {self.max_iterations} max iterations"
        )
        if refresh_restraints:
            self.logger.info(
                "🔄 Full restraint refresh enabled - restraints will be rebuilt on each iteration"
            )
        else:
            self.logger.info("📋 Using existing restraints - no refresh requested")
        start_time = time.time()
        gradient_norm = None
        for iteration in range(self.max_iterations):
            try:
                gradients, geometry_bundle_id = await self.forward(
                    refresh_restraints=refresh_restraints
                )
                geometry_gradient_bundle = self.redis_manager.get_bundle(
                    geometry_bundle_id
                )
                try:
                    self.latest_total_energy = geometry_gradient_bundle.get_asset(
                        "total_geometry_energy"
                    )
                except Exception:
                    self.latest_total_energy = None
                gradient_norm = torch.norm(gradients).item()
                iteration_info = {
                    "iteration": iteration,
                    "gradient_norm": gradient_norm,
                    "total_geometry_energy": self.latest_total_energy,
                    "timestamp": time.time(),
                }
                self.iteration_history.append(iteration_info)
                if gradient_norm < self.best_gradient_norm:
                    self.best_gradient_norm = gradient_norm
                    self.best_coordinates = self.current_coordinates.clone()
                self.logger.info(
                    f"Iteration {iteration}: gradient_norm = {gradient_norm:.6f}"
                )
                if gradient_norm < self.convergence_threshold:
                    self.logger.info(f"Converged at iteration {iteration}")
                    break
                await self.backward(gradients, step=iteration)
                if self.scheduler is not None:
                    self.scheduler.step()  # <-- Step the scheduler
                await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error in iteration {iteration}: {e}")
                break
        if self.best_coordinates is not None:
            self.current_coordinates = self.best_coordinates

        # Update the macromolecule bundle with final coordinates
        translator = ArrayTranslator(
            default_dtype=np.dtype("float64"), default_device=self.device
        )
        coordinates_cctbx = translator.convert(
            self.current_coordinates.detach(), "cctbx"
        )

        # Update the macromolecule bundle with final coordinates
        updated_bundle_id = self._update_coordinates_in_bundle(
            self.current_coordinates.detach()
        )
        self.macromolecule_bundle_id = updated_bundle_id

        # Get the updated macromolecule bundle and update the model_manager's sites_cart
        macromolecule_bundle = self.redis_manager.get_bundle(
            self.macromolecule_bundle_id
        )
        model_manager = macromolecule_bundle.get_asset("model_manager")
        model_manager.set_sites_cart(coordinates_cctbx)

        results = {
            "converged": gradient_norm is not None
            and gradient_norm < self.convergence_threshold,
            "final_gradient_norm": (
                gradient_norm if gradient_norm is not None else float("nan")
            ),
            "best_gradient_norm": self.best_gradient_norm,
            "iterations": len(self.iteration_history),
            "total_time": time.time() - start_time,
            "final_bundle_id": self.macromolecule_bundle_id,
            "iteration_history": self.iteration_history,
            "final_total_geometry_energy": (
                self.latest_total_energy
                if hasattr(self, "latest_total_energy")
                else None
            ),
            "optimizer_type": type(self.optimizer).__name__,
        }
        return results

    def get_best_coordinates(self) -> torch.Tensor:
        """Get the best coordinates found during minimization."""
        if self.best_coordinates is not None:
            return self.best_coordinates
        else:
            return self.current_coordinates

    def save_coordinates(self, filepath: str) -> None:
        """Save current coordinates to file."""
        try:
            coordinates = self.get_best_coordinates()
            torch.save(coordinates, filepath)
            self.logger.info(f"Coordinates saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save coordinates: {e}")
            raise

    def get_minimization_stats(self) -> Dict[str, Any]:
        """Get minimization statistics."""
        return {
            "total_iterations": len(self.iteration_history),
            "best_gradient_norm": self.best_gradient_norm,
            "final_gradient_norm": (
                self.iteration_history[-1]["gradient_norm"]
                if self.iteration_history
                else None
            ),
            "converged": self.best_gradient_norm < self.convergence_threshold,
            "optimizer": self.optimizer_name,
            "learning_rate": self.learning_rate,
            "current_bundle_id": self.macromolecule_bundle_id,
        }
