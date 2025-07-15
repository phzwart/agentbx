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

from ..core.bundle_base import Bundle
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
        model_data = input_bundles["xray_atomic_model_data"]

        # Extract target gradients w.r.t. structure factors
        target_gradients_wrt_sf = target_data.get_asset("target_gradients_wrt_sf")
        if target_gradients_wrt_sf is None:
            raise ValueError("Target gradients w.r.t. structure factors not found")

        # Extract structure factor gradients w.r.t. parameters
        sf_gradients_wrt_params = sf_data.get_asset("sf_gradients_wrt_params")
        if sf_gradients_wrt_params is None:
            raise ValueError("Structure factor gradients w.r.t. parameters not found")

        # Apply chain rule: dT/dx = dT/dF * dF/ 