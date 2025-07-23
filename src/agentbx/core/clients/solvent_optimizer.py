"""
Solvent Parameter Optimizer

A specialized optimization client for bulk solvent parameter optimization in macromolecule bundles.
"""

from typing import Any
from typing import Dict

import torch

from .optimization_client import OptimizationClient


class SolventOptimizer(OptimizationClient):
    """
    Optimizer for bulk solvent parameters in macromolecule bundles.

    Features:
    - Loads solvent parameters from macromolecule bundle
    - Updates solvent parameters for structure factor calculations
    - Maintains solvent parameter consistency
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
        Initialize the solvent parameter optimizer.

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
            parameter_name="solvent_parameters",
        )

    def _load_parameters_from_bundle(self) -> torch.Tensor:
        """Load solvent parameters from the macromolecule bundle."""
        try:
            # Get macromolecule bundle
            macromolecule_bundle = self.redis_manager.get_bundle(
                self.macromolecule_bundle_id
            )

            # Get solvent parameters from bundle (or create defaults)
            solvent_params = macromolecule_bundle.get_asset("solvent_parameters")

            if solvent_params is None:
                # Create default solvent parameters
                solvent_params = {
                    "k_sol": 0.35,
                    "b_sol": 50.0,
                    "grid_resolution_factor": 1 / 3,
                    "solvent_radius": 1.11,
                }

            # Convert to tensor
            param_values = torch.tensor(
                [
                    solvent_params["k_sol"],
                    solvent_params["b_sol"],
                    solvent_params["grid_resolution_factor"],
                    solvent_params["solvent_radius"],
                ],
                dtype=self.dtype,
                device=self.device,
            )

            self.logger.info(
                f"Loaded solvent parameters: k_sol={param_values[0]:.3f}, b_sol={param_values[1]:.3f}"
            )
            return param_values

        except Exception as e:
            self.logger.error(f"Failed to load solvent parameters: {e}")
            raise

    def _update_parameters_in_bundle(self, new_parameters: torch.Tensor) -> str:
        """Update solvent parameters in the macromolecule bundle."""
        try:
            # Convert tensor back to parameter dict
            solvent_params = {
                "k_sol": float(new_parameters[0]),
                "b_sol": float(new_parameters[1]),
                "grid_resolution_factor": float(new_parameters[2]),
                "solvent_radius": float(new_parameters[3]),
            }

            # Get macromolecule bundle
            macromolecule_bundle = self.redis_manager.get_bundle(
                self.macromolecule_bundle_id
            )

            # Update solvent parameters
            macromolecule_bundle.add_asset("solvent_parameters", solvent_params)

            # Store updated bundle
            updated_bundle_id = self.redis_manager.store_bundle(macromolecule_bundle)

            # Update our reference to the new bundle ID
            self.macromolecule_bundle_id = updated_bundle_id

            self.logger.info(
                f"Updated solvent parameters in bundle: {updated_bundle_id}"
            )
            return updated_bundle_id

        except Exception as e:
            self.logger.error(f"Failed to update solvent parameters in bundle: {e}")
            raise

    def _get_parameter_description(self) -> str:
        """Get description of the parameter being optimized."""
        return "solvent parameter"

    def get_solvent_stats(self) -> Dict[str, Any]:
        """Get solvent parameter-specific statistics."""
        stats = self.get_optimization_stats()

        # Add solvent parameter-specific info
        if self.current_parameters is not None:
            stats.update(
                {
                    "k_sol": float(self.current_parameters[0]),
                    "b_sol": float(self.current_parameters[1]),
                    "grid_resolution_factor": float(self.current_parameters[2]),
                    "solvent_radius": float(self.current_parameters[3]),
                }
            )

        return stats

    def get_solvent_parameters(self) -> Dict[str, float]:
        """Get current solvent parameters as a dictionary."""
        if self.current_parameters is not None:
            return {
                "k_sol": float(self.current_parameters[0]),
                "b_sol": float(self.current_parameters[1]),
                "grid_resolution_factor": float(self.current_parameters[2]),
                "solvent_radius": float(self.current_parameters[3]),
            }
        else:
            return {}
