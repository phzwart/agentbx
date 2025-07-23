"""
B-Factor Optimizer

A specialized optimization client for B-factor optimization in macromolecule bundles.
"""

from typing import Any
from typing import Dict

import torch

from .optimization_client import OptimizationClient


class BFactorOptimizer(OptimizationClient):
    """
    Optimizer for B-factors (temperature factors) in macromolecule bundles.

    Features:
    - Loads B-factors from xray_structure in macromolecule bundle
    - Updates B-factors in xray_structure
    - Maintains B-factor consistency
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
        Initialize the B-factor optimizer.

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
            parameter_name="b_factors",
        )

    def _load_parameters_from_bundle(self) -> torch.Tensor:
        """Load B-factors from the macromolecule bundle."""
        try:
            # Get macromolecule bundle
            macromolecule_bundle = self.redis_manager.get_bundle(
                self.macromolecule_bundle_id
            )

            # Get xray_structure from bundle
            xray_structure = macromolecule_bundle.get_asset("xray_structure")

            # Get B-factors as CCTBX flex array
            b_factors = xray_structure.extract_u_iso_or_u_equiv() * 8 * 3.14159**2

            # Convert to PyTorch tensor
            b_factors_tensor = self.coordinate_translator.cctbx_to_torch(b_factors)

            self.logger.info(
                f"Loaded {b_factors_tensor.shape[0]} B-factors from bundle"
            )
            return b_factors_tensor

        except Exception as e:
            self.logger.error(f"Failed to load B-factors: {e}")
            raise

    def _update_parameters_in_bundle(self, new_b_factors: torch.Tensor) -> str:
        """Update B-factors in the macromolecule bundle."""
        try:
            # Convert PyTorch tensor to CCTBX flex array
            cctbx_b_factors = self.coordinate_translator.torch_to_cctbx(new_b_factors)

            # Get macromolecule bundle
            macromolecule_bundle = self.redis_manager.get_bundle(
                self.macromolecule_bundle_id
            )
            xray_structure = macromolecule_bundle.get_asset("xray_structure")

            # Update B-factors in xray_structure
            # Convert B-factors back to U_iso (B = 8π²U)
            u_iso = cctbx_b_factors / (8 * 3.14159**2)

            # Update U_iso values
            scatterers = xray_structure.scatterers()
            for i, scatterer in enumerate(scatterers):
                scatterer.u_iso = u_iso[i]

            # Store updated bundle
            updated_bundle_id = self.redis_manager.store_bundle(macromolecule_bundle)

            # Update our reference to the new bundle ID
            self.macromolecule_bundle_id = updated_bundle_id

            self.logger.info(f"Updated B-factors in bundle: {updated_bundle_id}")
            return updated_bundle_id

        except Exception as e:
            self.logger.error(f"Failed to update B-factors in bundle: {e}")
            raise

    def _get_parameter_description(self) -> str:
        """Get description of the parameter being optimized."""
        return "B-factor"

    def get_bfactor_stats(self) -> Dict[str, Any]:
        """Get B-factor-specific statistics."""
        stats = self.get_optimization_stats()

        # Add B-factor-specific info
        if self.current_parameters is not None:
            stats.update(
                {
                    "n_atoms": self.current_parameters.shape[0],
                    "b_factor_range": {
                        "min": float(torch.min(self.current_parameters)),
                        "max": float(torch.max(self.current_parameters)),
                        "mean": float(torch.mean(self.current_parameters)),
                        "std": float(torch.std(self.current_parameters)),
                    },
                }
            )

        return stats
