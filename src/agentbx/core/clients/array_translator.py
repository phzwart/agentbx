"""
ArrayTranslator - Centralized dialect-aware array conversion system.

This module provides comprehensive conversion between different array types:
- CCTBX (flex.vec3_double, flex.double, etc.)
- NumPy arrays
- PyTorch tensors
- Bundle storage formats (bytes, lists, etc.)

Features:
- Dialect-aware conversions (cctbx, numpy, torch)
- Einops integration for reshaping
- Bundle packing/unpacking utilities
- Automatic shape validation and conversion
"""

import logging
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from cctbx.array_family import flex


# Optional einops import
try:
    import einops

    EINOPS_AVAILABLE = True
except ImportError:
    EINOPS_AVAILABLE = False
    einops = None


@dataclass
class ConversionInfo:
    """Information about array conversion."""

    source_dialect: str
    target_dialect: str
    original_shape: Tuple[int, ...]
    target_shape: Tuple[int, ...]
    conversion_type: str
    metadata: Dict[str, Any]


class ArrayTranslator:
    """
    Centralized dialect-aware array conversion system.

    Supports conversions between:
    - CCTBX flex arrays (cctbx dialect)
    - NumPy arrays (numpy dialect)
    - PyTorch tensors (torch dialect)
    - Bundle storage formats (bytes, lists)
    """

    def __init__(
        self,
        default_dtype: np.dtype = np.dtype("float64"),
        default_device: torch.device = None,
        use_einops: bool = True,
    ):
        """
        Initialize the array translator.

        Args:
            default_dtype: Default numpy dtype for conversions
            default_device: Default torch device for conversions
            use_einops: Whether to use einops for reshaping (if available)
        """
        self.default_dtype = default_dtype
        self.default_device = default_device or torch.device("cpu")
        self.use_einops = use_einops and EINOPS_AVAILABLE

        self.logger = logging.getLogger("ArrayTranslator")
        self.conversion_history: List[ConversionInfo] = []

    def convert(
        self,
        array: Any,
        target_dialect: Literal["cctbx", "numpy", "torch"],
        target_shape: Optional[Tuple[int, ...]] = None,
        **kwargs,
    ) -> Any:
        """
        Convert array to target dialect with optional reshaping.

        Args:
            array: Input array (any supported type)
            target_dialect: Target dialect ("cctbx", "numpy", "torch")
            target_shape: Optional target shape for reshaping
            **kwargs: Additional conversion parameters # noqa: RST210

        Returns:
            Converted array in target dialect

        """

        # Detect source dialect
        source_dialect = self._detect_dialect(array)

        # Convert to numpy first (intermediate format)
        numpy_array = self._to_numpy(array, source_dialect)

        # Reshape if requested
        if target_shape is not None:
            numpy_array = self._reshape_array(numpy_array, target_shape)

        # Convert to target dialect
        result = self._from_numpy(numpy_array, target_dialect, **kwargs)

        # Record conversion
        self._record_conversion(
            source_dialect,
            target_dialect,
            numpy_array.shape,
            result.shape if hasattr(result, "shape") else None,
            f"{source_dialect}_to_{target_dialect}",
        )

        return result

    def pack_for_bundle(
        self,
        array: Any,
        dialect: Literal["cctbx", "numpy", "torch"],
        storage_format: Literal["bytes", "list"] = "bytes",
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Pack array for bundle storage.

        Args:
            array: Input array (any supported type)
            dialect: Target dialect ("cctbx", "numpy", "torch")
            storage_format: Storage format ("bytes", "list")

        Returns:
            Tuple of (packed_data, metadata)

        Raises:
            ValueError: If the dialect is not supported.
        """
        if dialect == "cctbx":
            # For CCTBX, we typically pickle or store as-is
            packed_data = array
            metadata = {
                "dialect": "cctbx",
                "type": str(type(array)),
                "size": array.size() if hasattr(array, "size") else len(array),
            }
        elif dialect == "numpy":
            if storage_format == "bytes":
                packed_data = array.tobytes()
                metadata = {
                    "dialect": "numpy",
                    "shape": array.shape,
                    "dtype": str(array.dtype),
                    "storage_format": "bytes",
                }
            else:  # list
                packed_data = array.tolist()
                metadata = {
                    "dialect": "numpy",
                    "shape": array.shape,
                    "dtype": str(array.dtype),
                    "storage_format": "list",
                }
        elif dialect == "torch":
            # Convert torch to numpy first, then pack
            numpy_array = self._to_numpy(array, "torch")
            return self.pack_for_bundle(numpy_array, "numpy", storage_format)
        else:
            raise ValueError(f"Unsupported dialect: {dialect}")

        return packed_data, metadata

    def unpack_from_bundle(self, packed_data: Any, metadata: Dict[str, Any]) -> Any:
        """
        Unpack array from bundle storage.

        Args:
            packed_data: Packed data from bundle
            metadata: Bundle metadata

        Returns:
            Unpacked array

        Raises:
            ValueError: If the dialect is not supported.
        """
        dialect = metadata.get("dialect")

        if dialect == "cctbx":
            # CCTBX objects are typically stored as-is
            return packed_data
        elif dialect == "numpy":
            storage_format = metadata.get("storage_format", "bytes")
            shape = metadata.get("shape")
            dtype = metadata.get("dtype", str(self.default_dtype))

            if storage_format == "bytes":
                array = np.frombuffer(packed_data, dtype=np.dtype(dtype))
            else:  # list
                array = np.array(packed_data, dtype=np.dtype(dtype))

            # Reshape if shape is provided
            if shape is not None:
                array = array.reshape(shape)

            return array
        else:
            raise ValueError(f"Unsupported dialect: {dialect}")

    def _detect_dialect(self, array: Any) -> str:
        """Detect the dialect of an array."""
        try:
            if isinstance(array, (flex.double, flex.vec3_double)):
                return "cctbx"
        except (TypeError, AttributeError):
            # Handle cases where flex types are not properly loaded
            pass

        if isinstance(array, np.ndarray):
            return "numpy"
        elif isinstance(array, torch.Tensor):
            return "torch"
        elif hasattr(array, "as_numpy_array"):  # Duck typing for CCTBX arrays
            return "cctbx"
        else:
            raise ValueError(f"Cannot detect dialect for type: {type(array)}")

    def _to_numpy(self, array: Any, source_dialect: str) -> np.ndarray:
        """Convert array to numpy."""
        if source_dialect == "cctbx":
            return self._cctbx_to_numpy(array)
        elif source_dialect == "numpy":
            return array
        elif source_dialect == "torch":
            return array.detach().cpu().numpy()
        else:
            raise ValueError(f"Unsupported source dialect: {source_dialect}")

    def _from_numpy(
        self, numpy_array: np.ndarray, target_dialect: str, **kwargs
    ) -> Any:
        """Convert numpy array to target dialect."""
        if target_dialect == "cctbx":
            return self._numpy_to_cctbx(numpy_array)
        elif target_dialect == "numpy":
            return numpy_array
        elif target_dialect == "torch":
            device = kwargs.get("device", self.default_device)
            dtype = kwargs.get("dtype", torch.float32)
            requires_grad = kwargs.get("requires_grad", False)

            if not numpy_array.flags.writeable:
                numpy_array = numpy_array.copy()
            tensor = torch.from_numpy(numpy_array).to(device=device, dtype=dtype)
            if requires_grad:
                tensor.requires_grad_(True)
            return tensor
        else:
            raise ValueError(f"Unsupported target dialect: {target_dialect}")

    def _cctbx_to_numpy(self, cctbx_array: Any) -> np.ndarray:
        """Convert CCTBX array to numpy."""
        try:
            if hasattr(cctbx_array, "as_numpy_array"):
                return cctbx_array.as_numpy_array()
            elif hasattr(cctbx_array, "__iter__"):
                return np.array(list(cctbx_array), dtype=self.default_dtype)
            else:
                return np.array(cctbx_array, dtype=self.default_dtype)
        except Exception as e:
            self.logger.error(f"Failed to convert CCTBX array to numpy: {e}")
            raise

    def _numpy_to_cctbx(self, numpy_array: np.ndarray) -> Any:
        """Convert numpy array to CCTBX."""
        try:
            arr = np.ascontiguousarray(numpy_array, dtype=self.default_dtype)

            if arr.ndim == 1:
                return flex.double(arr)
            elif arr.ndim == 2 and arr.shape[1] == 3:
                return flex.vec3_double(arr)
            else:
                raise ValueError(
                    f"Unsupported numpy array shape for CCTBX conversion: {arr.shape}"
                )
        except Exception as e:
            self.logger.error(f"Failed to convert numpy array to CCTBX: {e}")
            raise

    def _reshape_array(
        self, array: np.ndarray, target_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """Reshape array using einops or numpy."""
        if self.use_einops and einops is not None:
            # Use einops for more flexible reshaping
            try:
                # Convert shape tuple to einops pattern
                pattern = " ".join([f"d{i}" for i in range(len(target_shape))])
                return einops.rearrange(array, f"({pattern}) -> {pattern}")
            except Exception as e:
                self.logger.warning(
                    f"Einops reshaping failed, falling back to numpy: {e}"
                )
                return array.reshape(target_shape)
        else:
            # Use numpy reshape
            return array.reshape(target_shape)

    def _record_conversion(
        self,
        source_dialect: str,
        target_dialect: str,
        original_shape: Tuple[int, ...],
        target_shape: Optional[Tuple[int, ...]],
        conversion_type: str,
    ) -> None:
        """Record conversion information."""
        info = ConversionInfo(
            source_dialect=source_dialect,
            target_dialect=target_dialect,
            original_shape=original_shape,
            target_shape=target_shape or original_shape,
            conversion_type=conversion_type,
            metadata={
                "use_einops": self.use_einops,
                "default_dtype": str(self.default_dtype),
                "default_device": str(self.default_device),
            },
        )

        self.conversion_history.append(info)

        # Keep only last 100 conversions
        if len(self.conversion_history) > 100:
            self.conversion_history = self.conversion_history[-100:]

    def get_conversion_history(self) -> List[ConversionInfo]:
        """Get conversion history."""
        return self.conversion_history.copy()

    def clear_history(self) -> None:
        """Clear conversion history."""
        self.conversion_history.clear()


# Convenience functions for common conversions
def cctbx_to_torch(cctbx_array: Any, **kwargs) -> torch.Tensor:
    """Convert CCTBX array to PyTorch tensor."""
    translator = ArrayTranslator()
    return translator.convert(cctbx_array, "torch", **kwargs)


def torch_to_cctbx(tensor: torch.Tensor, **kwargs) -> Any:
    """Convert PyTorch tensor to CCTBX array."""
    translator = ArrayTranslator()
    return translator.convert(tensor, "cctbx", **kwargs)


def numpy_to_torch(numpy_array: np.ndarray, **kwargs) -> torch.Tensor:
    """Convert NumPy array to PyTorch tensor."""
    translator = ArrayTranslator()
    return translator.convert(numpy_array, "torch", **kwargs)


def torch_to_numpy(tensor: torch.Tensor, **kwargs) -> np.ndarray:
    """Convert PyTorch tensor to NumPy array."""
    translator = ArrayTranslator()
    return translator.convert(tensor, "numpy", **kwargs)
