"""
Abstract Optimization Client

A base class for parameter optimization in macromolecule bundles with
Redis integration and async geometry calculations.
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC
from abc import abstractmethod
from dataclasses import asdict

# Import for type hints only
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple  # noqa: F401

import torch
import torch.nn as nn
import torch.optim as optim

from agentbx.core.processors.macromolecule_processor import MacromoleculeProcessor
from agentbx.core.redis_manager import RedisManager

from .coordinate_translator import CoordinateTranslator


# fmt: off
from cctbx.array_family import flex  # noqa: F401  # Needed for unpickling cctbx objects from Redis  # isort: skip
# fmt: on


if TYPE_CHECKING:
    pass


class OptimizationClient(ABC, nn.Module):
    """
    Abstract base class for parameter optimization in macromolecule bundles.

    Features:
    - Single source for parameter updates
    - Async geometry calculations via Redis streams
    - Multiple optimization algorithms
    - Convergence monitoring
    - Bundle registration and tracking
    """

    def __init__(
        self,
        redis_manager: RedisManager,
        macromolecule_bundle_id: str,
        learning_rate: float = 0.01,
        optimizer: str = "adam",
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6,
        timeout_seconds: float = 30.0,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        parameter_name: str = "parameters",
    ):
        """
        Initialize the optimization client.

        Args:
            redis_manager: Redis manager for bundle operations
            macromolecule_bundle_id: ID of the macromolecule bundle to optimize
            learning_rate: Learning rate for optimization
            optimizer: Optimization algorithm ("gd" or "adam")
            max_iterations: Maximum number of iterations
            convergence_threshold: Gradient norm threshold for convergence
            timeout_seconds: Timeout for geometry calculations
            device: Device for tensors
            dtype: Data type for tensors
            parameter_name: Name of the parameter being optimized
        """
        super().__init__()

        self.redis_manager = redis_manager
        self.macromolecule_bundle_id = macromolecule_bundle_id
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.timeout_seconds = timeout_seconds
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype
        self.parameter_name = parameter_name

        self.logger = logging.getLogger(f"{self.__class__.__name__}")

        # Initialize coordinate translator for tensor conversions
        self.coordinate_translator = CoordinateTranslator(
            redis_manager=redis_manager,
            coordinate_system="cartesian",
            requires_grad=True,
            dtype=self.dtype,
            device=self.device,
        )

        # Initialize macromolecule processor for bundle updates
        self.macromolecule_processor = MacromoleculeProcessor(
            redis_manager, f"{self.__class__.__name__}_processor"
        )

        # Load initial parameters from macromolecule bundle
        self.initial_parameters = self._load_parameters_from_bundle()
        self.current_parameters = self.initial_parameters.clone()

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Tracking
        self.iteration_history: List[Dict[str, float]] = []
        self.best_parameters: Optional[torch.Tensor] = None
        self.best_gradient_norm = float("inf")

    @abstractmethod
    def _load_parameters_from_bundle(self) -> torch.Tensor:
        """Load parameters from the macromolecule bundle. Must be implemented by subclasses."""

    @abstractmethod
    def _update_parameters_in_bundle(self, new_parameters: torch.Tensor) -> str:
        """Update parameters in the macromolecule bundle. Must be implemented by subclasses."""

    @abstractmethod
    def _get_parameter_description(self) -> str:
        """Get description of the parameter being optimized. Must be implemented by subclasses."""

    def _create_optimizer(self) -> optim.Optimizer:
        """Create the optimizer."""
        if self.optimizer_name.lower() == "adam":
            return optim.Adam([self.current_parameters], lr=self.learning_rate)
        elif self.optimizer_name.lower() == "gd":
            return optim.SGD([self.current_parameters], lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    async def _request_geometry_calculation(self) -> str:
        """Request geometry calculation from the async agent."""
        try:
            # Import GeometryRequest here to avoid circular import
            from agentbx.core.agents.async_geometry_agent import GeometryRequest

            request = GeometryRequest(
                macromolecule_bundle_id=self.macromolecule_bundle_id,
                priority=1,
                request_id=str(uuid.uuid4()),
            )

            # Send request to Redis stream
            stream_name = "geometry_requests"
            redis_client = self.redis_manager._get_client()
            await redis_client.xadd(
                stream_name,
                {
                    "request": json.dumps(asdict(request)),
                    "timestamp": time.time(),
                    "source": f"{self.__class__.__name__}",
                },
            )

            self.logger.info("Geometry calculation request sent")

            # Wait for response
            response_bundle_id = await self._wait_for_geometry_response()

            return response_bundle_id

        except Exception as e:
            self.logger.error(f"Geometry calculation request failed: {e}")
            raise

    async def _wait_for_geometry_response(self) -> str:
        """Wait for geometry calculation response."""
        try:
            response_stream = "geometry_responses"
            consumer_group = f"{self.__class__.__name__}_consumer"
            consumer_name = f"{self.__class__.__name__}_1"

            # Create consumer group if it doesn't exist
            try:
                await self.redis_manager._get_client().xgroup_create(
                    response_stream, consumer_group, mkstream=True
                )
            except Exception:
                # Group already exists
                pass

            # Read from response stream
            start_time = time.time()
            while time.time() - start_time < self.timeout_seconds:
                try:
                    # Read messages from the stream
                    messages = await self.redis_manager._get_client().xreadgroup(
                        consumer_group,
                        consumer_name,
                        {response_stream: ">"},
                        count=1,
                        block=1000,
                    )

                    if messages:
                        for stream, message_list in messages:
                            for message_id, fields in message_list:
                                # Parse response
                                response_data = fields.get(b"response", b"{}").decode()

                                # Extract bundle ID from response
                                response_dict = json.loads(response_data)
                                bundle_id = response_dict.get("geometry_bundle_id")

                                if bundle_id:
                                    # Acknowledge message
                                    await self.redis_manager._get_client().xack(
                                        response_stream, consumer_group, message_id
                                    )

                                    self.logger.info(
                                        f"Received geometry response: {bundle_id}"
                                    )
                                    return bundle_id

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

    async def forward(self) -> torch.Tensor:
        """
        Forward pass: calculate geometry gradients.

        Returns:
            Gradient tensor
        """
        # Request geometry calculation
        geometry_bundle_id = await self._request_geometry_calculation()

        # Load geometry gradients from bundle
        geometry_bundle = self.redis_manager.get_bundle(geometry_bundle_id)
        gradients = geometry_bundle.get_asset("geometry_gradients")

        # Convert to PyTorch tensor
        gradients_tensor = self.coordinate_translator.cctbx_to_torch(gradients)

        return gradients_tensor

    async def backward(self, gradients: torch.Tensor) -> None:
        """
        Backward pass: set gradients for parameters.

        Args:
            gradients: Gradient tensor
        """
        # Zero gradients
        self.optimizer.zero_grad()

        # Set gradients
        self.current_parameters.grad = gradients

    async def optimize(self) -> Dict[str, Any]:
        """
        Run the optimization loop.

        Returns:
            Dictionary with optimization results
        """
        parameter_desc = self._get_parameter_description()
        self.logger.info(
            f"Starting {parameter_desc} optimization with {self.max_iterations} max iterations"
        )

        start_time = time.time()

        for iteration in range(self.max_iterations):
            try:
                # Forward pass: calculate gradients
                gradients = await self.forward()

                # Calculate gradient norm
                gradient_norm = torch.norm(gradients).item()

                # Record iteration
                iteration_info = {
                    "iteration": iteration,
                    "gradient_norm": gradient_norm,
                    "timestamp": time.time(),
                }
                self.iteration_history.append(iteration_info)

                # Check for best parameters
                if gradient_norm < self.best_gradient_norm:
                    self.best_gradient_norm = gradient_norm
                    self.best_parameters = self.current_parameters.clone()

                self.logger.info(
                    f"Iteration {iteration}: gradient_norm = {gradient_norm:.6f}"
                )

                # Check convergence
                if gradient_norm < self.convergence_threshold:
                    self.logger.info(f"Converged at iteration {iteration}")
                    break

                # Backward pass: set gradients
                await self.backward(gradients)

                # Optimization step: update parameters
                self.optimizer.step()

                # Update parameters in macromolecule bundle
                self._update_parameters_in_bundle(self.current_parameters)

                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error in iteration {iteration}: {e}")
                break

        # Restore best parameters if available
        if self.best_parameters is not None:
            self.current_parameters = self.best_parameters
            self._update_parameters_in_bundle(self.current_parameters)

        # Prepare results
        total_time = time.time() - start_time
        results = {
            "converged": gradient_norm < self.convergence_threshold,
            "final_gradient_norm": gradient_norm,
            "best_gradient_norm": self.best_gradient_norm,
            "iterations": len(self.iteration_history),
            "total_time": total_time,
            "final_bundle_id": self.macromolecule_bundle_id,
            "iteration_history": self.iteration_history,
            "parameter_type": self.parameter_name,
        }

        self.logger.info(f"Optimization completed in {total_time:.2f}s")
        return results

    def get_best_parameters(self) -> torch.Tensor:
        """Get the best parameters found during optimization."""
        if self.best_parameters is not None:
            return self.best_parameters
        else:
            return self.current_parameters

    def save_parameters(self, filepath: str) -> None:
        """Save current parameters to file."""
        try:
            parameters = self.get_best_parameters()
            torch.save(parameters, filepath)
            self.logger.info(f"Parameters saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save parameters: {e}")
            raise

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
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
            "parameter_type": self.parameter_name,
        }
