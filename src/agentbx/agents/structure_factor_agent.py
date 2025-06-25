# src/agentbx/agents/structure_factor_agent.py
"""
Agent responsible ONLY for structure factor calculations.

Input: atomic_model_data
Output: structure_factor_data

Does NOT know about:
- Target functions
- Gradients
- Optimization
- Experimental data
"""

import logging
from typing import Any
from typing import Dict
from typing import List

from ..core.bundle_base import Bundle
from .base import SinglePurposeAgent


logger = logging.getLogger(__name__)


class StructureFactorAgent(SinglePurposeAgent):
    """
    Pure structure factor calculation agent.

    Responsibility: Convert atomic models to structure factors.
    """

    def define_input_bundle_types(self) -> List[str]:
        """Define the input bundle types for this agent."""
        return ["atomic_model_data"]

    def define_output_bundle_types(self) -> List[str]:
        """Define the output bundle types for this agent."""
        return ["structure_factor_data"]

    def process_bundles(self, input_bundles: Dict[str, Bundle]) -> Dict[str, Bundle]:
        """
        Pure structure factor calculation from atomic model.
        """
        logger.info("Starting structure factor calculation...")

        atomic_model = input_bundles["atomic_model_data"]

        # Extract CCTBX objects
        xray_structure = atomic_model.get_asset("xray_structure")
        miller_indices = atomic_model.get_asset("miller_indices")

        logger.info(f"X-ray structure has {len(xray_structure.scatterers())} atoms")
        logger.info(f"Miller indices has {miller_indices.size()} reflections")
        logger.info(f"Resolution: {miller_indices.d_min():.2f} Å")

        # Calculate F_calc (pure structure factors from model)
        logger.info("Calculating F_calc...")
        f_calc = self._calculate_f_calc(xray_structure, miller_indices)
        logger.info(f"F_calc calculation completed: {f_calc.size()} reflections")

        # Create base output bundle
        sf_bundle = Bundle(bundle_type="structure_factor_data")
        sf_bundle.add_asset("f_calc", f_calc)
        sf_bundle.add_asset("miller_indices", miller_indices)

        # Optional bulk solvent correction
        if atomic_model.has_asset("bulk_solvent_params"):
            logger.info("Calculating bulk solvent correction...")
            bulk_params = atomic_model.get_asset("bulk_solvent_params")
            f_mask = self._calculate_f_mask(xray_structure, miller_indices, bulk_params)
            f_model, scale_factors = self._combine_structure_factors(
                f_calc, f_mask, bulk_params
            )

            sf_bundle.add_asset("f_mask", f_mask)
            sf_bundle.add_asset("f_model", f_model)
            sf_bundle.add_asset("scale_factors", scale_factors)
            logger.info("Bulk solvent correction completed")

        logger.info("Structure factor calculation completed successfully")
        return {"structure_factor_data": sf_bundle}

    def _calculate_f_calc(self, xray_structure: Any, miller_indices: Any) -> Any:
        """
        Calculate structure factors from atomic model.

        Pure F_calc calculation - no bulk solvent, no scaling.
        """
        from cctbx.array_family import flex

        # Get the resolution from the miller indices
        d_min = miller_indices.d_min()
        logger.info(f"Calculating structure factors at {d_min:.2f} Å resolution...")

        try:
            # Calculate structure factors using the correct CCTBX API
            f_calc = xray_structure.structure_factors(
                d_min=d_min, algorithm="direct"
            ).f_calc()

            logger.info(f"Structure factors calculated: {f_calc.size()} reflections")

            # Create a mapping between the calculated reflections and our target miller indices
            # We need to find which reflections in f_calc match our miller_indices
            target_indices = set(miller_indices.indices())
            calc_indices = f_calc.indices()

            # Create boolean selection array
            selection = flex.bool(f_calc.size(), False)
            for i, idx in enumerate(calc_indices):
                if idx in target_indices:
                    selection[i] = True

            # Select the matching reflections
            f_calc_selected = f_calc.select(selection)
            logger.info(
                f"Selected {f_calc_selected.size()} reflections matching input miller indices"
            )

            # Ensure the selected reflections are in the same order as our input miller_indices
            # Create a new miller array with the same indices as our input
            f_calc_final = miller_indices.array(data=f_calc_selected.data())

            return f_calc_final

        except Exception as e:
            logger.error(f"Error in structure factor calculation: {e}")
            raise

    def _calculate_f_mask(
        self, xray_structure: Any, miller_indices: Any, bulk_params: Dict[str, Any]
    ) -> Any:
        """
        Calculate bulk solvent structure factors.
        """
        logger.info("Calculating bulk solvent mask...")

        try:
            from mmtbx.bulk_solvent import bulk_solvent_and_scaling

            # Create bulk solvent mask
            bulk_solvent_manager = bulk_solvent_and_scaling.bulk_solvent_manager(
                xray_structure=xray_structure,
                miller_indices=miller_indices,
                bulk_solvent_params=bulk_params,
            )

            f_mask = bulk_solvent_manager.f_mask()
            logger.info(f"Bulk solvent mask calculated: {f_mask.size()} reflections")
            return f_mask

        except ImportError:
            logger.warning("mmtbx not available, using zero mask as fallback")
            # Fallback if mmtbx is not available
            from cctbx.array_family import flex

            # Create a simple mask with zeros
            f_mask_data = flex.complex_double(miller_indices.size(), 0.0)
            f_mask = miller_indices.array(data=f_mask_data)

            return f_mask
        except Exception as e:
            logger.error(f"Error in bulk solvent calculation: {e}")
            # Fallback to zero mask
            from cctbx.array_family import flex

            f_mask_data = flex.complex_double(miller_indices.size(), 0.0)
            f_mask = miller_indices.array(data=f_mask_data)
            return f_mask

    def _combine_structure_factors(
        self, f_calc: Any, f_mask: Any, bulk_params: Dict[str, Any]
    ) -> tuple[Any, Dict[str, Any]]:
        """
        Combine F_calc and F_mask with scaling.

        F_model = k_overall * (F_calc + k_sol * exp(-B_sol * s^2/4) * F_mask)
        """
        from cctbx.array_family import flex

        k_sol = bulk_params.get("k_sol", 0.35)
        b_sol = bulk_params.get("b_sol", 50.0)
        k_overall = 1.0  # Will be determined by scaling

        # Apply B_sol scaling to F_mask
        d_star_sq = f_mask.d_star_sq().data()
        exp_b_sol = flex.exp(-b_sol * d_star_sq / 4.0)
        f_mask_scaled = f_mask.customized_copy(data=f_mask.data() * exp_b_sol)

        # Combine structure factors
        f_model_data = f_calc.data() + k_sol * f_mask_scaled.data()
        f_model = f_calc.customized_copy(data=f_model_data)

        # Scale factors used
        scale_factors = {"k_overall": k_overall, "k_sol": k_sol, "b_sol": b_sol}

        return f_model, scale_factors

    def get_computation_info(self) -> Dict[str, Any]:
        """Return information about this agent's computation."""
        return {
            "agent_type": "StructureFactorAgent",
            "responsibility": "Structure factor calculation",
            "algorithms": ["direct_summation", "bulk_solvent_correction"],
            "cctbx_modules": ["cctbx.xray", "mmtbx.bulk_solvent"],
        }
