"""
Processor responsible ONLY for geometry gradient calculations.

Input: macromolecule_data
Output: geometry_gradient_data

Does NOT know about:
- Target functions
- Structure factors
- Optimization
- Experimental data
"""

import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from agentbx.core.bundle_base import Bundle
from agentbx.schemas.generated import GeometryGradientDataBundle

from .base import SinglePurposeProcessor


logger = logging.getLogger(__name__)


class CctbxGeometryProcessor(SinglePurposeProcessor):
    """CCTBX-based geometry gradient processor."""

    def __init__(self, redis_manager: Any, processor_id: str) -> None:
        super().__init__(redis_manager, processor_id)
        self.logger = logging.getLogger(f"CctbxGeometryProcessor.{processor_id}")

    def define_input_bundle_types(self) -> List[str]:
        """Define the input bundle types for this processor."""
        return ["macromolecule_data"]

    def define_output_bundle_types(self) -> List[str]:
        """Define the output bundle types for this processor."""
        return ["geometry_gradient_data"]

    def process_bundles(self, input_bundles: Dict[str, Bundle]) -> Dict[str, Bundle]:
        """
        Calculate geometry gradients from macromolecule bundle.
        """
        return self.process_bundles_with_refresh(
            input_bundles, refresh_restraints=False
        )

    def process_bundles_with_refresh(
        self, input_bundles: Dict[str, Bundle], refresh_restraints: bool = False
    ) -> Dict[str, Bundle]:
        """Process macromolecule bundle to calculate geometry gradients with optional restraint refresh."""
        try:
            # Get macromolecule bundle
            macromolecule_bundle = input_bundles["macromolecule_data"]

            # Get model manager
            model_manager = macromolecule_bundle.get_asset("model_manager")

            # Get or create restraints manager
            if refresh_restraints:
                # Rebuild restraints from current model state
                restraints_manager = model_manager.get_restraints_manager()
                self.logger.info(
                    "🔄 Full restraint refresh enabled - restraints will be rebuilt on each iteration"
                )
            else:
                # Use existing restraints
                restraints_manager = macromolecule_bundle.get_asset("restraint_manager")
                self.logger.info("📋 Using existing restraints (no refresh requested)")

            # Calculate geometry gradients
            geometry_gradients = self._calculate_geometry_gradients(
                restraints_manager, model_manager
            )

            # Create gradient bundle
            gradient_bundle = self._create_gradient_bundle(
                geometry_gradients, macromolecule_bundle.bundle_id
            )

            return {"geometry_gradients": gradient_bundle}

        except Exception as e:
            self.logger.error(f"Failed to process geometry gradients: {e}")
            raise

    def _calculate_geometry_gradients(
        self, geometry_restraints_manager: Any, model_manager: Any
    ) -> Any:
        """
        Calculate geometry gradients using the geometry restraints manager.

        Args:
            geometry_restraints_manager: The geometry restraints manager.
            model_manager: The model manager.

        Returns:
            Tuple of (gradient tensor, geometry gradient bundle ID)

        Raises:
            RuntimeError: If there is an error in the geometry restraints manager.
        """
        try:
            pass

            sites_cart = model_manager.get_sites_cart()
            if sites_cart.size() == 0:
                raise RuntimeError("No atoms found in model_manager")
            try:
                current_restraints_manager = model_manager.get_restraints_manager()
                if current_restraints_manager is not None:
                    geometry_restraints = current_restraints_manager.geometry
                    call_args = {"sites_cart": sites_cart, "compute_gradients": True}
                    energies_and_gradients = geometry_restraints.energies_sites(
                        **call_args
                    )
                    total_geometry_energy = energies_and_gradients.target
                    logger.info(
                        f"Total geometry energy: {total_geometry_energy:.6f} (existing restraints manager)"
                    )
                    coordinate_gradients = energies_and_gradients.gradients
                    if coordinate_gradients.size() != sites_cart.size():
                        raise RuntimeError(
                            f"Gradient size mismatch: {coordinate_gradients.size()} vs {sites_cart.size()}"
                        )
                    logger.info(
                        f"Successfully calculated gradients for {coordinate_gradients.size()} atoms (existing restraints manager)"
                    )
                    return coordinate_gradients, total_geometry_energy
                else:
                    logger.warning(
                        "Existing restraints manager is None, will try to build a fresh one."
                    )
            except Exception as e:
                logger.warning(f"Existing restraints manager failed: {e}")
                logger.info(
                    "Falling back to building a fresh geometry restraints manager."
                )
            try:
                restraints_manager = model_manager.get_restraints_manager()
                if restraints_manager is None:
                    raise RuntimeError("No restraints manager found in model_manager")

                # Get geometry restraints
                geometry_restraints_manager = restraints_manager.geometry
                call_args = {"sites_cart": sites_cart, "compute_gradients": True}
                energies_and_gradients = geometry_restraints_manager.energies_sites(
                    **call_args
                )
                total_geometry_energy = energies_and_gradients.target
                logger.info(
                    f"Total geometry energy: {total_geometry_energy:.6f} (fresh restraints manager)"
                )
                coordinate_gradients = energies_and_gradients.gradients
                if coordinate_gradients.size() != sites_cart.size():
                    raise RuntimeError(
                        f"Gradient size mismatch: {coordinate_gradients.size()} vs {sites_cart.size()}"
                    )
                logger.info(
                    f"Successfully calculated gradients for {coordinate_gradients.size()} atoms (fresh restraints manager)"
                )
                return coordinate_gradients, total_geometry_energy
            except Exception as fresh_error:
                logger.warning(
                    f"Fresh geometry restraints manager failed: {fresh_error}"
                )
                logger.info(
                    "Unable to calculate gradients with either existing or fresh restraints manager."
                )
                raise RuntimeError(
                    f"Failed to calculate geometry gradients and energy: {fresh_error}"
                ) from fresh_error
        except Exception as e:
            logger.error(f"Error in _calculate_geometry_gradients: {e}")
            raise

    def _calculate_gradient_norm(self, geometry_gradients: Any) -> float:
        """Calculate the norm of geometry gradients."""
        try:
            if hasattr(geometry_gradients, "size"):
                # Single gradient array
                return geometry_gradients.norm()
            elif isinstance(geometry_gradients, tuple):
                # Tuple of (gradients, energy) - extract just the gradients
                gradients, _ = geometry_gradients
                return gradients.norm()
            else:
                self.logger.warning(
                    f"Unknown gradient type: {type(geometry_gradients)}"
                )
                return 0.0
        except Exception as e:
            self.logger.error(f"Error calculating gradient norm: {e}")
            return 0.0

    def _create_gradient_bundle(
        self, geometry_gradients: Any, parent_bundle_id: Optional[str]
    ) -> Bundle:
        """Create a gradient bundle from geometry gradients."""
        # Create geometry gradient bundle
        geo_bundle = Bundle(bundle_type="geometry_gradient_data")
        geo_bundle.add_asset("geometry_gradients", geometry_gradients)
        geo_bundle.add_asset(
            "gradient_norm", self._calculate_gradient_norm(geometry_gradients)
        )
        if isinstance(geometry_gradients, tuple):
            _, total_energy = geometry_gradients
            geo_bundle.add_asset("total_geometry_energy", total_energy)
        geo_bundle.add_metadata("geometry_type", "bond_length_angle_dihedral")
        geo_bundle.add_metadata("calculation_method", "analytical")
        geo_bundle.add_metadata("source_macromolecule", parent_bundle_id)
        geo_bundle.add_metadata("dialect", "cctbx")
        # Validate with schema
        GeometryGradientDataBundle(
            coordinates=None,  # Not available in this context
            geometric_gradients=geometry_gradients,
            restraint_energies={},
            restraint_counts={},
            geometry_metadata={},
        )
        print("[Schema Validation] GeometryGradientDataBundle validation successful.")
        return geo_bundle

    def calculate_geometry_gradients(self, macromolecule_bundle_id: str) -> str:
        """
        Calculate geometry gradients from macromolecule bundle.
        """
        # Process the macromolecule bundle
        output_ids = self.run({"macromolecule_data": macromolecule_bundle_id})
        return output_ids["geometry_gradient_data"]

    def get_computation_info(self) -> Dict[str, Any]:
        """
        Get information about this processor's computational requirements.
        """
        return {
            "processor_type": "geometry_processor",
            "input_types": ["macromolecule_data"],
            "output_types": ["geometry_gradient_data"],
            "memory_usage": "medium",
            "cpu_usage": "medium",
            "gpu_usage": "none",
        }
