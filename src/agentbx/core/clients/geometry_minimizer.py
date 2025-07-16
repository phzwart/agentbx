"""
Geometry Minimizer Module

A PyTorch-style geometry minimizer that integrates with Redis bundles and
the async geometry agent system for coordinate optimization.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from cctbx.array_family import flex

from ..redis_manager import RedisManager
from .coordinate_translator import CoordinateTranslator
from ..processors.macromolecule_processor import MacromoleculeProcessor

# Import for type hints only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..agents.async_geometry_agent import AsyncGeometryAgent


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
        learning_rate: float = 0.01,
        optimizer: str = "adam",
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6,
        timeout_seconds: float = 30.0,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize the geometry minimizer.
        
        Args:
            redis_manager: Redis manager for bundle operations
            macromolecule_bundle_id: ID of the macromolecule bundle to minimize
            learning_rate: Learning rate for optimization
            optimizer: Optimization algorithm ("gd" or "adam")
            max_iterations: Maximum number of iterations
            convergence_threshold: Gradient norm threshold for convergence
            timeout_seconds: Timeout for geometry calculations
            device: Device for tensors
            dtype: Data type for tensors
        """
        super().__init__()
        
        self.redis_manager = redis_manager
        self.macromolecule_bundle_id = macromolecule_bundle_id
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.timeout_seconds = timeout_seconds
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        
        self.logger = logging.getLogger("GeometryMinimizer")
        
        # Initialize coordinate translator
        self.coordinate_translator = CoordinateTranslator(
            redis_manager=redis_manager,
            coordinate_system="cartesian",
            requires_grad=True,
            dtype=self.dtype,
            device=self.device
        )
        
        # Initialize macromolecule processor for coordinate updates
        self.macromolecule_processor = MacromoleculeProcessor(
            redis_manager, "minimizer_processor"
        )
        
        # Load initial coordinates from macromolecule bundle
        self.initial_coordinates = self._load_coordinates_from_bundle()
        self.current_coordinates = self.initial_coordinates.clone()
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Tracking
        self.iteration_history: List[Dict[str, float]] = []
        self.best_coordinates: Optional[torch.Tensor] = None
        self.best_gradient_norm = float('inf')
        
    def _load_coordinates_from_bundle(self) -> torch.Tensor:
        """Load coordinates from the macromolecule bundle."""
        try:
            # Get macromolecule bundle
            macromolecule_bundle = self.redis_manager.get_bundle(self.macromolecule_bundle_id)
            
            # Get xray_structure from bundle
            xray_structure = macromolecule_bundle.get_asset("xray_structure")
            
            # Get coordinates as CCTBX flex array
            sites_cart = xray_structure.sites_cart()
            
            # Convert to PyTorch tensor
            coordinates_tensor = self.coordinate_translator.cctbx_to_torch(sites_cart)
            
            self.logger.info(f"Loaded {coordinates_tensor.shape[0]} coordinates from bundle")
            return coordinates_tensor
            
        except Exception as e:
            self.logger.error(f"Failed to load coordinates: {e}")
            raise
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create the optimizer."""
        if self.optimizer_name.lower() == "adam":
            return optim.Adam([self.current_coordinates], lr=self.learning_rate)
        elif self.optimizer_name.lower() == "gd":
            return optim.SGD([self.current_coordinates], lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
    
    async def _request_geometry_calculation(self) -> str:
        """Request geometry calculation from the async agent."""
        try:
            # Import GeometryRequest here to avoid circular import
            from ..agents.async_geometry_agent import GeometryRequest
            
            request = GeometryRequest(
                macromolecule_bundle_id=self.macromolecule_bundle_id,
                calculation_type="geometry_gradients",
                priority="normal"
            )
            
            # Send request to Redis stream
            stream_name = "geometry_requests"
            await self.redis_manager.redis_client.xadd(
                stream_name,
                {
                    "request": request.model_dump_json(),
                    "timestamp": time.time(),
                    "source": "geometry_minimizer"
                }
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
            consumer_group = "minimizer_consumer"
            consumer_name = "minimizer_1"
            
            # Create consumer group if it doesn't exist
            try:
                await self.redis_manager.redis_client.xgroup_create(
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
                    messages = await self.redis_manager.redis_client.xreadgroup(
                        consumer_group, consumer_name,
                        {response_stream: ">"}, count=1, block=1000
                    )
                    
                    if messages:
                        for stream, message_list in messages:
                            for message_id, fields in message_list:
                                # Parse response
                                response_data = fields.get(b"response", b"{}").decode()
                                
                                # Extract bundle ID from response
                                import json
                                response_dict = json.loads(response_data)
                                bundle_id = response_dict.get("geometry_bundle_id")
                                
                                if bundle_id:
                                    # Acknowledge message
                                    await self.redis_manager.redis_client.xack(
                                        response_stream, consumer_group, message_id
                                    )
                                    
                                    self.logger.info(f"Received geometry response: {bundle_id}")
                                    return bundle_id
                    
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    self.logger.warning(f"Error reading response stream: {e}")
                    await asyncio.sleep(0.1)
            
            raise TimeoutError(f"Geometry calculation timed out after {self.timeout_seconds} seconds")
            
        except Exception as e:
            self.logger.error(f"Failed to wait for geometry response: {e}")
            raise
    
    def _update_coordinates_in_bundle(self, new_coordinates: torch.Tensor) -> str:
        """Update coordinates in the macromolecule bundle."""
        try:
            # Convert PyTorch tensor to CCTBX flex array
            cctbx_coordinates = self.coordinate_translator.torch_to_cctbx(new_coordinates)
            
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
        Backward pass: update coordinates using gradients.
        
        Args:
            gradients: Gradient tensor
        """
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Set gradients
        self.current_coordinates.grad = gradients
        
        # Update coordinates
        self.optimizer.step()
        
        # Update coordinates in macromolecule bundle
        self._update_coordinates_in_bundle(self.current_coordinates)
    
    async def minimize(self) -> Dict[str, Any]:
        """
        Run the minimization loop.
        
        Returns:
            Dictionary with minimization results
        """
        self.logger.info(f"Starting geometry minimization with {self.max_iterations} max iterations")
        
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
                    "timestamp": time.time()
                }
                self.iteration_history.append(iteration_info)
                
                # Check for best coordinates
                if gradient_norm < self.best_gradient_norm:
                    self.best_gradient_norm = gradient_norm
                    self.best_coordinates = self.current_coordinates.clone()
                
                self.logger.info(f"Iteration {iteration}: gradient_norm = {gradient_norm:.6f}")
                
                # Check convergence
                if gradient_norm < self.convergence_threshold:
                    self.logger.info(f"Converged at iteration {iteration}")
                    break
                
                # Backward pass: update coordinates
                await self.backward(gradients)
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in iteration {iteration}: {e}")
                break
        
        # Restore best coordinates if available
        if self.best_coordinates is not None:
            self.current_coordinates = self.best_coordinates
            self._update_coordinates_in_bundle(self.current_coordinates)
        
        # Prepare results
        total_time = time.time() - start_time
        results = {
            "converged": gradient_norm < self.convergence_threshold,
            "final_gradient_norm": gradient_norm,
            "best_gradient_norm": self.best_gradient_norm,
            "iterations": len(self.iteration_history),
            "total_time": total_time,
            "final_bundle_id": self.macromolecule_bundle_id,
            "iteration_history": self.iteration_history
        }
        
        self.logger.info(f"Minimization completed in {total_time:.2f}s")
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
            "final_gradient_norm": self.iteration_history[-1]["gradient_norm"] if self.iteration_history else None,
            "converged": self.best_gradient_norm < self.convergence_threshold,
            "optimizer": self.optimizer_name,
            "learning_rate": self.learning_rate,
            "current_bundle_id": self.macromolecule_bundle_id
        } 