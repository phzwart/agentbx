# src/agentbx/agents/target_agent.py
"""
Agent responsible ONLY for target function calculations.

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
from .base import SinglePurposeAgent


class TargetAgent(SinglePurposeAgent):
    """
    Pure target function calculation agent.

    Responsibility: Compute target values from structure factors and experimental data.
    """

    def define_input_bundle_types(self) -> List[str]:
        """Define the input bundle types for this agent."""
        return ["experimental_data", "structure_factor_data"]

    def define_output_bundle_types(self) -> List[str]:
        """Define the output bundle types for this agent."""
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
        if "likelihood_parameters" in result:
            target_bundle.add_asset(
                "likelihood_parameters", result["likelihood_parameters"]
            )

        if "target_gradients_wrt_sf" in result:
            target_bundle.add_asset(
                "target_gradients_wrt_sf", result["target_gradients_wrt_sf"]
            )

        return {"target_data": target_bundle}

    def _calculate_ml_target(
        self, f_model: Any, f_obs: Any, exp_data: Any
    ) -> Dict[str, Any]:
        """
        Maximum likelihood target calculation.
        """
        from mmtbx.refinement import targets

        # Get R_free flags if available
        r_free_flags = exp_data.get_asset("r_free_flags")
        if r_free_flags is None:
            # Create dummy R_free flags
            r_free_flags = f_obs.array(data=f_obs.data() * 0.0).as_bool()

        # Common reflections
        f_obs_common, f_model_common = f_obs.common_sets(
            f_model, assert_is_similar_symmetry=False
        )
        r_free_common = r_free_flags.common_set(f_obs_common)

        # Create ML target functor
        target_functor = targets.target_functor(
            f_obs=f_obs_common, compute_gradients=True
        )

        # Calculate target
        target_result = target_functor.target_and_gradients(f_model=f_model_common)

        # Calculate R-factors
        r_work = self._calculate_r_factor(
            f_obs_common.select(~r_free_common.data()),
            f_model_common.select(~r_free_common.data()),
        )
        r_free = self._calculate_r_factor(
            f_obs_common.select(r_free_common.data()),
            f_model_common.select(r_free_common.data()),
        )

        return {
            "target_value": target_result.target(),
            "r_factors": {
                "r_work": r_work,
                "r_free": r_free,
                "number_reflections_work": (~r_free_common.data()).count(True),
                "number_reflections_free": r_free_common.data().count(True),
            },
            "likelihood_parameters": {
                "alpha": target_result.alpha(),
                "beta": target_result.beta(),
            },
            "target_gradients_wrt_sf": target_result.gradients(),
        }

    def _calculate_ls_target(
        self, f_model: Any, f_obs: Any, exp_data: Any
    ) -> Dict[str, Any]:
        """
        Least squares target calculation.
        """

        # Get R_free flags
        r_free_flags = exp_data.get_asset("r_free_flags")
        if r_free_flags is None:
            r_free_flags = f_obs.array(data=f_obs.data() * 0.0).as_bool()

        # Common reflections
        f_obs_common, f_model_common = f_obs.common_sets(
            f_model, assert_is_similar_symmetry=False
        )
        r_free_common = r_free_flags.common_set(f_obs_common)

        # Calculate least squares target
        diff = f_obs_common.data() - f_model_common.amplitudes().data()
        target_value = (diff * diff).sum()

        # Calculate R-factors
        r_work = self._calculate_r_factor(
            f_obs_common.select(~r_free_common.data()),
            f_model_common.select(~r_free_common.data()),
        )
        r_free = self._calculate_r_factor(
            f_obs_common.select(r_free_common.data()),
            f_model_common.select(r_free_common.data()),
        )

        # Gradients w.r.t. structure factors (for chain rule)
        gradients = -2.0 * diff
        gradient_array = f_model_common.customized_copy(data=gradients)

        return {
            "target_value": target_value,
            "r_factors": {
                "r_work": r_work,
                "r_free": r_free,
                "number_reflections_work": (~r_free_common.data()).count(True),
                "number_reflections_free": r_free_common.data().count(True),
            },
            "target_gradients_wrt_sf": gradient_array,
        }

    def _calculate_lsf_target(
        self, f_model: Any, f_obs: Any, exp_data: Any
    ) -> Dict[str, Any]:
        """
        Least squares on F target calculation.
        """
        # Similar to LS but on structure factor amplitudes
        return self._calculate_ls_target(f_model, f_obs, exp_data)

    def _calculate_r_factor(self, f_obs: Any, f_model: Any) -> float:
        """
        Calculate crystallographic R-factor.

        R = Σ|F_obs - F_model| / Σ|F_obs|
        """
        if f_obs.size() == 0:
            return 0.0

        f_model_amp = f_model.amplitudes() if f_model.is_complex_array() else f_model

        numerator = (f_obs.data() - f_model_amp.data()).norm()
        denominator = f_obs.data().norm()

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def get_computation_info(self) -> Dict[str, Any]:
        """Return information about this agent's computation."""
        return {
            "agent_type": "TargetAgent",
            "responsibility": "Target function calculation",
            "supported_targets": [
                "maximum_likelihood",
                "least_squares",
                "least_squares_f",
            ],
            "cctbx_modules": ["mmtbx.refinement.targets"],
        }
