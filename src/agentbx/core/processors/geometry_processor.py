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
import os
import time
from typing import Any
from typing import Dict
from typing import List

from ..bundle_base import Bundle
from .base import SinglePurposeProcessor


logger = logging.getLogger(__name__)


class CctbxGeometryProcessor(SinglePurposeProcessor):
    """
    Pure geometry gradient calculation processor.

    Responsibility: Compute geometry gradients from macromolecule bundles using CCTBX.
    """

    def __init__(self, redis_manager: Any, processor_id: str) -> None:
        """
        Initialize the geometry processor.
        """
        super().__init__(redis_manager, processor_id)

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
        macromolecule_data = input_bundles["macromolecule_data"]

        # Get model_manager and xray_structure from macromolecule bundle
        model_manager = macromolecule_data.get_asset("model_manager")
        xray_structure = macromolecule_data.get_asset("xray_structure")
        
        # Get geometry restraints manager from model_manager (like in working example)
        
        # Ensure model manager is processed with restraints
        try:
            model_manager.process(make_restraints=True)
        except Exception as process_error:
            logger.warning(f"Model manager processing failed: {process_error}")
        
        restraints_manager = model_manager.get_restraints_manager()
        
        if restraints_manager is None:
            restraints_manager = macromolecule_data.get_asset("restraint_manager")
        
        geometry_restraints_manager = restraints_manager.geometry
        
        # Calculate geometry gradients
        geometry_gradients = self._calculate_geometry_gradients(geometry_restraints_manager, model_manager)

        # Create geometry gradient bundle
        geo_bundle = Bundle(bundle_type="geometry_gradient_data")
        geo_bundle.add_asset("geometry_gradients", geometry_gradients)
        geo_bundle.add_asset("gradient_norm", self._calculate_gradient_norm(geometry_gradients))

        # Add metadata
        geo_bundle.add_metadata("geometry_type", "bond_length_angle_dihedral")
        geo_bundle.add_metadata("calculation_method", "analytical")
        geo_bundle.add_metadata("source_macromolecule", macromolecule_data.bundle_id)

        return {"geometry_gradient_data": geo_bundle}

    def _calculate_geometry_gradients(self, geometry_restraints_manager: Any, model_manager: Any) -> Any:
        """
        Calculate geometry gradients using CCTBX geometry restraints.
        """
        try:
            from cctbx.array_family import flex
            
            # Get the current coordinates from the model_manager (like in working example)
            sites_cart = model_manager.get_sites_cart()
            
            # Safety check: ensure we have valid coordinates
            if sites_cart.size() == 0:
                logger.warning("No atoms found in model_manager")
                return None
                
            # Try to create a fresh geometry restraints manager
            try:
                # Get the restraints manager from the model manager
                fresh_restraints_manager = model_manager.get_restraints_manager()
                if fresh_restraints_manager is not None:
                    fresh_geometry_restraints = fresh_restraints_manager.geometry
                    
                    # Try with the fresh manager
                    call_args = {
                        'sites_cart': sites_cart,
                        'compute_gradients': True
                    }
                    
                    fresh_energies_and_gradients = fresh_geometry_restraints.energies_sites(**call_args)
                    
                    # Access the total energy
                    total_geometry_energy = fresh_energies_and_gradients.target
                    
                    # Access the gradients (forces) on each atom
                    coordinate_gradients = fresh_energies_and_gradients.gradients
                    
                    # Safety check: ensure gradients have the right size
                    if coordinate_gradients.size() != sites_cart.size():
                        logger.warning(f"Gradient size mismatch: {coordinate_gradients.size()} vs {sites_cart.size()}")
                        return None
                    
                    logger.info(f"Successfully calculated gradients for {coordinate_gradients.size()} atoms")
                    logger.info(f"Total geometry energy: {total_geometry_energy:.6f}")
                    
                    return coordinate_gradients
                else:
                    logger.warning("Fresh restraints manager is None, falling back to stored one")
            except Exception as fresh_error:
                logger.warning(f"Fresh geometry restraints manager failed: {fresh_error}")
                logger.info("Falling back to stored geometry restraints manager")
            
            # Fallback to stored geometry restraints manager
            call_args = {
                'sites_cart': sites_cart,
                'compute_gradients': True
            }
            
            energies_and_gradients = geometry_restraints_manager.energies_sites(**call_args)
            
            # Access the total energy
            total_geometry_energy = energies_and_gradients.target
            
            # Access the gradients (forces) on each atom
            coordinate_gradients = energies_and_gradients.gradients
            
            # Safety check: ensure gradients have the right size
            if coordinate_gradients.size() != sites_cart.size():
                logger.warning(f"Gradient size mismatch: {coordinate_gradients.size()} vs {sites_cart.size()}")
                return None
            
            logger.info(f"Successfully calculated gradients for {coordinate_gradients.size()} atoms")
            logger.info(f"Total geometry energy: {total_geometry_energy:.6f}")
            
            return coordinate_gradients
                
        except Exception as e:
            logger.error(f"Error calculating geometry gradients: {e}")
            import traceback
            traceback.print_exc()
            # Return zero gradients as fallback
            try:
                from cctbx.array_family import flex
                # Get number of atoms from the model_manager
                n_atoms = model_manager.get_sites_cart().size()
                if n_atoms > 0:
                    zero_gradients = flex.vec3_double(n_atoms, (0.0, 0.0, 0.0))
                    logger.warning("Using zero gradients as fallback")
                    return zero_gradients
                else:
                    logger.warning("No atoms found, returning None")
                    return None
            except Exception as fallback_error:
                logger.error(f"Fallback gradient calculation also failed: {fallback_error}")
                return None

    def _calculate_gradient_norm(self, geometry_gradients: Any) -> float:
        """
        Calculate the norm of the geometry gradient vector.
        """
        if geometry_gradients is None:
            return 0.0
            
        try:
            import math
            from cctbx.array_family import flex
            
            # Convert vec3_double to double array for norm calculation
            # Create a new flex.double array and populate it manually
            grad_array = flex.double()
            for i in range(geometry_gradients.size()):
                grad_array.append(geometry_gradients[i][0])  # x component
                grad_array.append(geometry_gradients[i][1])  # y component
                grad_array.append(geometry_gradients[i][2])  # z component
            
            # Calculate L2 norm using Python math.sqrt
            sum_sq = flex.sum_sq(grad_array)
            norm = math.sqrt(float(sum_sq))
            return norm
            
        except Exception as e:
            logger.error(f"Error calculating gradient norm: {e}")
            return 0.0

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
