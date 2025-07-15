# src/agentbx/processors/target_processor.py
"""
Processor responsible ONLY for target function calculations.

Input: structure_factor_data + experimental_data
Output: target_data

Does NOT know about:
- Atomic coordinates
- Gradients w.r.t. atomic parameters
- Optimization algorithms
"""

from typing import Any
from typing import Dict
from typing import List

from ..core.bundle_base import Bundle
from .base import SinglePurposeProcessor


class TargetProcessor(SinglePurposeProcessor):
    """
    Pure target function calculation processor.

    Responsibility: Compute target values from structure factors and experimental data.
    """

    def define_input_bundle_types(self) -> List[str]:
        """Define the input bundle types for this processor."""
        return ["experimental_data", "structure_factor_data"]

    def define_output_bundle_types(self) -> List[str]:
        """Define the output bundle types for this processor."""
        return ["target_data"]

    def process_bundles(self, input_bundles: Dict[str, Bundle]) -> Dict[str, Bundle]:
        """
        Pure target function calculation.
        """
        sf_data = input_bundles["structure_factor_data"]
        exp_data = input_bundles["experimental_data"]

        # Extract structure factors and experimental data
        f_model = sf_data.get_asset("f_model")  # or f_calc if no bulk solvent
        if f_model is None:
            f_model = sf_data.get_asset("f_calc")

        f_obs = exp_data.get_asset("f_obs")

        # Get target type preference
        target_preferences = exp_data.get_metadata("target_preferences", {})
        target_type = target_preferences.get("default_target", "maximum_likelihood")

        # Calculate target function
        if target_type == "maximum_likelihood":
            result = self._calculate_ml_target(f_model, f_obs, exp_data)
        elif target_type == "least_squares":
            result = self._calculate_ls_target(f_model, f_obs, exp_data)
        elif target_type == "least_squares_f":
            result = self._calculate_lsf_target(f_model, f_obs, exp_data)
        else:
            raise ValueError(f"Unknown target type: {target_type}")

        # Create output bundle
        target_bundle = Bundle(bundle_type="target_data")
        target_bundle.add_asset("target_value", result["target_value"])
        target_bundle.add_asset("target_type", target_type)
        target_bundle.add_asset("r_factors", result["r_factors"])

        # Add target-specific results
        if "likelihood_parameters" in result 