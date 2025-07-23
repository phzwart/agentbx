"""
Coordinate Optimizer

A specialized optimization client for coordinate optimization in macromolecule bundles.
"""

from typing import Any
from typing import Dict

import torch

from .optimization_client import OptimizationClient


class CoordinateOptimizer(OptimizationClient):
    """
    Optimizer for atomic coordinates in macromolecule bundles.

    Features:
    - Loads coordinates from xray_structure in macromolecule bundle
    - Updates coordinates in all bundle representations (pdb_hierarchy, model_manager, xray_structure)
    - Maintains coordinate consistency across all representations
    """

    def __init__(
        self,
        redis_manager,
        macromolecule_bundle_id: str,
        learning_rate: float = 0.01,
        optimizer: str = "adam",
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6,
        timeout_seconds: float = 30.0,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize the coordinate optimizer.

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
        """
        super().__init__(
            redis_manager=redis_manager,
            macromolecule_bundle_id=macromolecule_bundle_id,
            learning_rate=learning_rate,
            optimizer=optimizer,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            timeout_seconds=timeout_seconds,
            device=device,
            dtype=dtype,
            parameter_name="coordinates",
        )

    def _load_parameters_from_bundle(self) -> torch.Tensor:
        """Load coordinates from the macromolecule bundle."""
        try:
            # Get macromolecule bundle
            macromolecule_bundle = self.redis_manager.get_bundle(
                self.macromolecule_bundle_id
            )

            # Get xray_structure from bundle
            xray_structure = macromolecule_bundle.get_asset("xray_structure")

            # Get coordinates as CCTBX flex array
            sites_cart = xray_structure.sites_cart()

            # Convert to PyTorch tensor
            coordinates_tensor = self.coordinate_translator.cctbx_to_torch(sites_cart)

            self.logger.info(
                f"Loaded {coordinates_tensor.shape[0]} coordinates from bundle"
            )
            return coordinates_tensor

        except Exception as e:
            self.logger.error(f"Failed to load coordinates: {e}")
            raise

    def _update_parameters_in_bundle(self, new_coordinates: torch.Tensor) -> str:
        """Update coordinates in the macromolecule bundle."""
        try:
            # Convert PyTorch tensor to CCTBX flex array
            cctbx_coordinates = self.coordinate_translator.torch_to_cctbx(
                new_coordinates
            )

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

    def _get_parameter_description(self) -> str:
        """Get description of the parameter being optimized."""
        return "coordinate"

    def get_coordinate_stats(self) -> Dict[str, Any]:
        """Get coordinate-specific statistics."""
        stats = self.get_optimization_stats()

        # Add coordinate-specific info
        if self.current_parameters is not None:
            stats.update(
                {
                    "n_atoms": self.current_parameters.shape[0],
                    "coordinate_range": {
                        "min": float(torch.min(self.current_parameters)),
                        "max": float(torch.max(self.current_parameters)),
                        "mean": float(torch.mean(self.current_parameters)),
                        "std": float(torch.std(self.current_parameters)),
                    },
                }
            )

        return stats
