"""
CoordinateTranslator - PyTorch module for CCTBX coordinate conversion.

This module provides seamless conversion between CCTBX coordinates and PyTorch tensors,
with automatic Redis bundle registration and gradient tracking.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from cctbx.array_family import (  # noqa: F401  # Needed for unpickling cctbx objects from Redis
    flex,
)

from agentbx.core.bundle_base import Bundle
from agentbx.core.clients.array_translator import ArrayTranslator
from agentbx.core.redis_manager import RedisManager


@dataclass
class CoordinateInfo:
    """Information about coordinate conversion."""

    original_shape: Tuple[int, ...]
    converted_shape: Tuple[int, ...]
    conversion_type: str  # "cctbx_to_torch" or "torch_to_cctbx"
    metadata: Dict[str, Any]


class CoordinateTranslator(nn.Module):
    """
    PyTorch module for converting between CCTBX coordinates and PyTorch tensors.

    Features:
    - Automatic conversion between CCTBX flex arrays and PyTorch tensors
    - Redis bundle integration for persistent storage
    - Automatic gradient registration and tracking
    - Support for different coordinate systems (Cartesian, fractional)
    - Memory-efficient operations with gradient checkpointing
    """

    def __init__(
        self,
        redis_manager: RedisManager,
        coordinate_system: str = "cartesian",
        requires_grad: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
        bundle_prefix: str = "coordinates",
    ):
        """
        Initialize the coordinate translator.

        Args:
            redis_manager: Redis manager for bundle operations
            coordinate_system: Coordinate system ("cartesian" or "fractional")
            requires_grad: Whether tensors should require gradients
            dtype: Data type for tensors
            device: Device for tensors (auto-detect if None)
            bundle_prefix: Prefix for Redis bundle keys
        """
        super().__init__()

        self.redis_manager = redis_manager
        self.coordinate_system = coordinate_system
        self.requires_grad = requires_grad
        self.dtype = dtype
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.bundle_prefix = bundle_prefix

        self.logger = logging.getLogger("CoordinateTranslator")

        # Conversion history for debugging
        self.conversion_history: List[CoordinateInfo] = []

        # Register hooks for gradient tracking
        self._register_hooks()

    def forward(self, input_data: Union[Any, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass - convert input to PyTorch tensor.

        Args:
            input_data: CCTBX flex array or PyTorch tensor

        Returns:
            PyTorch tensor with gradients if requires_grad=True

        Raises:
            ValueError: If the input type is not supported.
        """
        if self._is_cctbx_array(input_data):
            return self.cctbx_to_torch(input_data)
        elif isinstance(input_data, torch.Tensor):
            return self._ensure_tensor_properties(input_data)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

    def backward(self, gradients: torch.Tensor) -> Any:
        """
        Backward pass - convert gradients back to CCTBX format.

        Args:
            gradients: PyTorch tensor gradients

        Returns:
            CCTBX flex array with gradients
        """
        return self.torch_to_cctbx(gradients)

    def cctbx_to_torch(self, cctbx_array: Any) -> torch.Tensor:
        """
        Convert CCTBX flex array to PyTorch tensor.

        Args:
            cctbx_array: CCTBX flex array (vec3_double, double, etc.)

        Returns:
            PyTorch tensor
        """
        # Convert torch.dtype to numpy.dtype for ArrayTranslator
        numpy_dtype: Any
        if self.dtype == torch.float32:
            numpy_dtype = np.dtype(np.float32)
        elif self.dtype == torch.float64:
            numpy_dtype = np.dtype(np.float64)
        elif self.dtype == torch.float16:
            numpy_dtype = np.dtype(np.float16)
        else:
            # Default fallback
            numpy_dtype = np.dtype(np.float32)

        # Use ArrayTranslator for conversion
        translator = ArrayTranslator(
            default_dtype=numpy_dtype, default_device=self.device
        )
        tensor = translator.convert(
            cctbx_array, "torch", requires_grad=self.requires_grad
        )

        # Record conversion
        self._record_conversion(cctbx_array, tensor, "cctbx_to_torch")

        return tensor

    def torch_to_cctbx(self, tensor: torch.Tensor) -> Any:
        """
        Convert PyTorch tensor to CCTBX flex array.

        Args:
            tensor: PyTorch tensor

        Returns:
            CCTBX flex array

        Raises:
            ValueError: If the input is not a PyTorch tensor.
        """
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Expected PyTorch tensor, got {type(tensor)}")

        # Convert torch.dtype to numpy.dtype for ArrayTranslator
        numpy_dtype: Any
        if self.dtype == torch.float32:
            numpy_dtype = np.dtype(np.float32)
        elif self.dtype == torch.float64:
            numpy_dtype = np.dtype(np.float64)
        elif self.dtype == torch.float16:
            numpy_dtype = np.dtype(np.float16)
        else:
            # Default fallback
            numpy_dtype = np.dtype(np.float32)

        # Use ArrayTranslator for conversion
        translator = ArrayTranslator(
            default_dtype=numpy_dtype, default_device=self.device
        )
        cctbx_array = translator.convert(tensor, "cctbx")

        # Record conversion
        self._record_conversion(cctbx_array, tensor, "torch_to_cctbx")

        return cctbx_array

    def register_bundle(self, bundle_id: str, tensor: torch.Tensor) -> str:
        """
        Register a tensor as a Redis bundle.

        Args:
            bundle_id: Bundle ID (auto-generated if None)
            tensor: PyTorch tensor to register

        Returns:
            Bundle ID
        """
        # Create coordinate bundle
        bundle = Bundle(bundle_type="coordinate_data")
        bundle.add_asset("coordinates", tensor)
        bundle.add_asset("coordinate_system", self.coordinate_system)
        bundle.add_asset("requires_grad", self.requires_grad)
        bundle.add_asset("dtype", str(self.dtype))
        bundle.add_asset("device", str(self.device))

        # Store in Redis
        stored_bundle_id = self.redis_manager.store_bundle(bundle, bundle_id)

        self.logger.info(f"Registered coordinate bundle: {stored_bundle_id}")
        return stored_bundle_id

    def load_bundle(self, bundle_id: str) -> torch.Tensor:
        """
        Load a tensor from a Redis bundle.

        Args:
            bundle_id: Bundle ID to load

        Returns:
            PyTorch tensor

        Raises:
            ValueError: If the bundle type is not "coordinate_data".
        """
        bundle = self.redis_manager.get_bundle(bundle_id)

        if bundle.bundle_type != "coordinate_data":
            raise ValueError(
                f"Expected coordinate_data bundle, got {bundle.bundle_type}"
            )

        tensor = bundle.get_asset("coordinates")

        # Ensure tensor has correct properties
        tensor = self._ensure_tensor_properties(tensor)

        self.logger.info(f"Loaded coordinate bundle: {bundle_id}")
        return tensor

    def update_coordinates(self, bundle_id: str, new_coordinates: torch.Tensor) -> str:
        """
        Update coordinates in an existing bundle.

        Args:
            bundle_id: Bundle ID to update
            new_coordinates: New coordinate tensor

        Returns:
            Updated bundle ID
        """
        # Load existing bundle
        bundle = self.redis_manager.get_bundle(bundle_id)

        # Update coordinates
        bundle.add_asset("coordinates", new_coordinates)
        bundle.add_metadata("last_updated", datetime.now().isoformat())

        # Store updated bundle
        updated_bundle_id = self.redis_manager.store_bundle(bundle, bundle_id)

        self.logger.info(f"Updated coordinate bundle: {updated_bundle_id}")
        return updated_bundle_id

    def _is_cctbx_array(self, obj: Any) -> bool:
        """Check if object is a CCTBX flex array."""
        # Accept flex.double, flex.vec3_double, etc.
        return isinstance(obj, (flex.double, flex.vec3_double))

    def _cctbx_to_numpy(self, cctbx_array: Any) -> np.ndarray:
        """Convert CCTBX flex array to numpy array."""
        try:
            # Handle different CCTBX array types
            if hasattr(cctbx_array, "as_numpy_array"):
                return cctbx_array.as_numpy_array()
            elif hasattr(cctbx_array, "__iter__"):
                # Convert to list first, then to numpy
                return np.array(list(cctbx_array), dtype=np.float64)
            else:
                # Fallback: try to access data directly
                return np.array(cctbx_array, dtype=np.float64)
        except Exception as e:
            self.logger.error(f"Failed to convert CCTBX array to numpy: {e}")
            raise

    def _numpy_to_cctbx(self, numpy_array):
        import numpy as np
        from cctbx.array_family import flex

        arr = np.ascontiguousarray(numpy_array, dtype=np.float64)
        # Convert numpy array to flex.vec3_double
        if arr.ndim == 2 and arr.shape[1] == 3:
            return flex.vec3_double(arr)
        else:
            raise ValueError(
                f"Cannot convert numpy array to flex.vec3_double: shape={arr.shape}"
            )

    def _ensure_tensor_properties(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor has correct dtype, device, and requires_grad."""
        if tensor.dtype != self.dtype:
            tensor = tensor.to(dtype=self.dtype)

        if tensor.device != self.device:
            tensor = tensor.to(device=self.device)

        if tensor.requires_grad != self.requires_grad:
            tensor.requires_grad_(self.requires_grad)

        return tensor

    def _record_conversion(
        self, cctbx_array: Any, tensor: torch.Tensor, conversion_type: str
    ) -> None:
        """Record conversion information for debugging."""
        info = CoordinateInfo(
            original_shape=self._get_cctbx_shape(cctbx_array),
            converted_shape=tuple(tensor.shape),
            conversion_type=conversion_type,
            metadata={
                "coordinate_system": self.coordinate_system,
                "requires_grad": self.requires_grad,
                "dtype": str(self.dtype),
                "device": str(self.device),
            },
        )

        self.conversion_history.append(info)

        # Keep only last 100 conversions
        if len(self.conversion_history) > 100:
            self.conversion_history = self.conversion_history[-100:]

    def _get_cctbx_shape(self, cctbx_array: Any) -> Tuple[int, ...]:
        """Get shape of CCTBX array."""
        try:
            if hasattr(cctbx_array, "size"):
                if hasattr(cctbx_array, "all"):
                    # Multi-dimensional array
                    return tuple(cctbx_array.all())
                else:
                    # 1D array
                    return (cctbx_array.size(),)
            else:
                return (len(cctbx_array),)
        except Exception:
            return (0,)

    def _register_hooks(self) -> None:
        """Register hooks for gradient tracking."""

        def gradient_hook(grad):
            self.logger.debug(f"Gradient computed: {grad.shape}")
            return grad

        # Register hook for all parameters
        for name, param in self.named_parameters():
            param.register_hook(gradient_hook)

    def get_conversion_history(self) -> List[CoordinateInfo]:
        """Get conversion history for debugging."""
        return self.conversion_history.copy()

    def clear_history(self) -> None:
        """Clear conversion history."""
        self.conversion_history.clear()

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information."""
        if torch.cuda.is_available():
            return {
                "cuda_memory_allocated": torch.cuda.memory_allocated(),
                "cuda_memory_reserved": torch.cuda.memory_reserved(),
                "device": str(self.device),
            }
        else:
            return {"device": str(self.device), "memory_info": "CUDA not available"}
