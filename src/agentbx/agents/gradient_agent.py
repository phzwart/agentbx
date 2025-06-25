# src/agentbx/agents/gradient_agent.py
"""
Agent responsible ONLY for gradient calculations via chain rule.

Input: structure_factor_data + target_data + atomic_model_data
Output: gradient_data

Applies chain rule: dT/dx = dT/dF * dF/dx
"""

from typing import Any
from typing import Dict
from typing import List

from ..core.bundle_base import Bundle
from .base import SinglePurposeAgent


class GradientAgent(SinglePurposeAgent):
    """
    Pure gradient calculation agent via chain rule.

    Responsibility: Apply chain rule to get parameter gradients from target gradients.
    """

    def define_input_bundle_types(self) -> List[str]:
        """Define the input bundle types for this agent."""
        return ["structure_factor_data", "target_data", "atomic_model_data"]

    def define_output_bundle_types(self) -> List[str]:
        """Define the output bundle types for this agent."""
        return ["gradient_data"]

    def process_bundles(self, input_bundles: Dict[str, Bundle]) -> Dict[str, Bundle]:
        """
        Apply chain rule to compute parameter gradients.

        dT/dx = dT/dF * dF/dx (chain rule)
        """
        sf_data = input_bundles["structure_factor_data"]
        target_data = input_bundles["target_data"]
        atomic_data = input_bundles["atomic_model_data"]

        # Get target gradients w.r.t. structure factors (dT/dF)
        target_sf_grads = target_data.get_asset("target_gradients_wrt_sf")
        if target_sf_grads is None:
            raise ValueError(
                "Target data must contain gradients w.r.t. structure factors"
            )

        # Get structure factor gradients w.r.t. coordinates (dF/dx)
        sf_coord_grads = self._calculate_sf_gradients_wrt_coords(sf_data, atomic_data)

        # Apply chain rule: dT/dx = dT/dF * dF/dx
        coordinate_gradients = self._apply_chain_rule_coordinates(
            target_sf_grads, sf_coord_grads
        )

        # Optional: B-factor gradients
        bfactor_gradients = None
        occupancy_gradients = None

        # Create output bundle
        grad_bundle = Bundle(bundle_type="gradient_data")
        grad_bundle.add_asset("coordinate_gradients", coordinate_gradients)
        grad_bundle.add_asset("structure_factor_gradients", target_sf_grads)

        if bfactor_gradients is not None:
            grad_bundle.add_asset("bfactor_gradients", bfactor_gradients)

        if occupancy_gradients is not None:
            grad_bundle.add_asset("occupancy_gradients", occupancy_gradients)

        # Add gradient metadata
        grad_metadata = self._compute_gradient_metadata(coordinate_gradients)
        grad_bundle.add_asset("gradient_metadata", grad_metadata)

        return {"gradient_data": grad_bundle}

    def _calculate_sf_gradients_wrt_coords(self, sf_data: Any, atomic_data: Any) -> Any:
        """
        Calculate dF/dx - structure factor gradients w.r.t. coordinates.
        """

        xray_structure = atomic_data.get_asset("xray_structure")
        miller_indices = sf_data.get_asset("miller_indices")

        # Get current structure factors
        sf_data.get_asset("f_calc")

        # Calculate gradients of structure factors w.r.t. atomic coordinates
        # This uses CCTBX's automatic differentiation
        sf_derivatives = xray_structure.structure_factor_gradients(
            miller_indices=miller_indices.indices(),
            site_gradients=True,
            u_iso_gradients=False,
            u_aniso_gradients=False,
            occupancy_gradients=False,
        )

        return sf_derivatives.d_target_d_site_cart()

    def _apply_chain_rule_coordinates(self, dt_df: Any, df_dx: Any) -> Any:
        """
        Apply chain rule: dT/dx = dT/dF * dF/dx

        Args:
            dt_df: Gradients w.r.t. structure factors (complex array)
            df_dx: Gradients of structure factors w.r.t. coordinates

        Returns:
            Coordinate gradients for each atom
        """
        from cctbx.array_family import flex

        # This is where the actual chain rule mathematics happens
        # dt_df is a miller array with complex gradients
        # df_dx is coordinate gradients for each atom
        # The chain rule involves summing over all reflections:
        # dT/dx_i = Σ_h (dT/dF_h * dF_h/dx_i + dT/dF_h* * dF_h*/dx_i)

        coordinate_gradients = flex.vec3_double()

        for i_atom in range(df_dx.n_scatterers()):
            grad_x = grad_y = grad_z = 0.0

            # Sum over all reflections
            for i_refl in range(dt_df.size()):
                dt_df_val = dt_df.data()[i_refl]  # Complex gradient dT/dF
                df_dx_val = df_dx.d_target_d_site_cart()[i_atom]  # Real 3-vector dF/dx

                # Real part: Re(dT/dF) * dF_real/dx + Im(dT/dF) * dF_imag/dx
                # Imaginary part: Re(dT/dF) * dF_imag/dx - Im(dT/dF) * dF_real/dx

                grad_x += dt_df_val.real * df_dx_val[0] + dt_df_val.imag * df_dx_val[0]
                grad_y += dt_df_val.real * df_dx_val[1] + dt_df_val.imag * df_dx_val[1]
                grad_z += dt_df_val.real * df_dx_val[2] + dt_df_val.imag * df_dx_val[2]

            coordinate_gradients.append((grad_x, grad_y, grad_z))

        return coordinate_gradients

    def _compute_gradient_metadata(self, coordinate_gradients: Any) -> Dict[str, Any]:
        """
        Compute metadata about the gradients.
        """
        from cctbx.array_family import flex

        # Convert to flex array for calculations
        grad_norms = flex.double()
        for grad in coordinate_gradients:
            norm = (grad[0] ** 2 + grad[1] ** 2 + grad[2] ** 2) ** 0.5
            grad_norms.append(norm)

        return {
            "gradient_norm": grad_norms.norm(),
            "max_gradient": grad_norms.max_element(),
            "mean_gradient": grad_norms.mean(),
            "n_atoms": len(coordinate_gradients),
            "computation_method": "chain_rule",
        }

    def _finite_difference_test(
        self, atomic_data: Any, target_data: Any, delta: float = 1e-6
    ) -> None:
        """
        Verify gradients using finite differences (for testing).
        """
        # Implementation would perturb coordinates slightly and
        # compute numerical derivatives for comparison

    def get_computation_info(self) -> Dict[str, Any]:
        """Return information about this agent's computation."""
        return {
            "agent_type": "GradientAgent",
            "responsibility": "Chain rule gradient calculation",
            "algorithms": ["automatic_differentiation", "chain_rule"],
            "cctbx_modules": ["cctbx.examples.structure_factor_derivatives"],
            "mathematics": "dT/dx = dT/dF * dF/dx",
        }
