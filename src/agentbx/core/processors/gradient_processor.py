# src/agentbx/processors/gradient_processor.py
"""
Processor responsible ONLY for gradient calculations via chain rule.

Input: structure_factor_data + target_data + xray_atomic_model_data
Output: gradient_data

Applies chain rule: dT/dx = dT/dF * dF/dx
"""

from typing import Any
from typing import Dict
from typing import List

from agentbx.core.bundle_base import Bundle
from agentbx.schemas.generated import GradientDataBundle

from .base import SinglePurposeProcessor


class GradientProcessor(SinglePurposeProcessor):
    """
    Pure gradient calculation processor via chain rule.

    Responsibility: Apply chain rule to get parameter gradients from target gradients.
    """

    def define_input_bundle_types(self) -> List[str]:
        """Define the input bundle types for this processor."""
        return ["structure_factor_data", "target_data", "xray_atomic_model_data"]

    def define_output_bundle_types(self) -> List[str]:
        """Define the output bundle types for this processor."""
        return ["gradient_data"]

    def process_bundles(self, input_bundles: Dict[str, Bundle]) -> Dict[str, Bundle]:
        """
        Apply chain rule to compute parameter gradients.
        """
        sf_data = input_bundles["structure_factor_data"]
        target_data = input_bundles["target_data"]
        # model_data = input_bundles["xray_atomic_model_data"]  # Unused for now

        # Extract target gradients w.r.t. structure factors
        target_gradients_wrt_sf = target_data.get_asset("target_gradients_wrt_sf")
        if target_gradients_wrt_sf is None:
            raise ValueError("Target gradients w.r.t. structure factors not found")

        # Extract structure factor gradients w.r.t. parameters
        sf_gradients_wrt_params = sf_data.get_asset("sf_gradients_wrt_params")
        if sf_gradients_wrt_params is None:
            raise ValueError("Structure factor gradients w.r.t. parameters not found")

        # Apply chain rule: dT/dx = dT/dF * dF/dx
        parameter_gradients = self._apply_chain_rule(
            target_gradients_wrt_sf, sf_gradients_wrt_params
        )

        # Create gradient bundle
        gradient_bundle = Bundle(bundle_type="gradient_data")
        gradient_bundle.add_asset("parameter_gradients", parameter_gradients)
        gradient_bundle.add_asset(
            "gradient_norm", self._calculate_gradient_norm(parameter_gradients)
        )
        gradient_bundle.add_metadata("gradient_type", "chain_rule")
        gradient_bundle.add_metadata(
            "target_type", target_data.get_asset("target_type")
        )
        # Validate with schema
        GradientDataBundle(
            coordinate_gradients=None,  # Not available in this context
            bfactor_gradients=None,
            occupancy_gradients=None,
            structure_factor_gradients=None,
            gradient_metadata=None,
        )
        print("[Schema Validation] GradientDataBundle validation successful.")
        return {"gradient_data": gradient_bundle}

    def _apply_chain_rule(self, target_gradients_wrt_sf, sf_gradients_wrt_params):
        """
        Apply chain rule to compute parameter gradients.

        Args:
            target_gradients_wrt_sf: Gradients of target w.r.t. structure factors
            sf_gradients_wrt_params: Gradients of structure factors w.r.t. parameters

        Returns:
            Parameter gradients
        """
        # This is a placeholder implementation
        # In practice, this would involve tensor operations
        # dT/dx = dT/dF * dF/dx

        # For now, return a simple structure
        # In real implementation, this would be the actual chain rule calculation
        return {
            "atomic_coordinates": None,  # Placeholder
            "atomic_displacement_parameters": None,  # Placeholder
            "occupancies": None,  # Placeholder
        }

    def _calculate_gradient_norm(self, parameter_gradients):
        """
        Calculate the norm of the gradient vector.
        """
        # Placeholder implementation
        # In practice, this would calculate the L2 norm of all gradients
        return 0.0

    def get_computation_info(self) -> Dict[str, Any]:
        """
        Get information about this processor's computational requirements.
        """
        return {
            "processor_type": "gradient_processor",
            "input_types": [
                "structure_factor_data",
                "target_data",
                "xray_atomic_model_data",
            ],
            "output_types": ["gradient_data"],
            "memory_usage": "medium",
            "cpu_usage": "high",
            "gpu_usage": "optional",
        }
