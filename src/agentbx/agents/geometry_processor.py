"""
Processor responsible ONLY for geometry gradient calculations.

Input: xray_atomic_model_data
Output: geometry_gradient_data

Does NOT know about:
- Target functions
- Structure factors
- Optimization
- Experimental data
"""

import logging
import time
import os
from typing import Any
from typing import Dict
from typing import List

from ..core.bundle_base import Bundle
from .base import SinglePurposeProcessor


logger = logging.getLogger(__name__)


class CctbxGeometryProcessor(SinglePurposeProcessor):
    """
    Pure geometry gradient calculation processor.

    Responsibility: Compute geometry gradients from atomic models using CCTBX.
    """

    def define_input_bundle_types(self) -> List[str]:
        """Define the input bundle types for this processor."""
        return ["xray_atomic_model_data"]

    def define_output_bundle_types(self) -> List[str]:
        """Define the output bundle types for this processor."""
        return ["geometry_gradient_data"]

    def process_bundles(self, input_bundles: Dict[str, Bundle]) -> Dict[str, Bundle]:
        """
        Pure geometry gradient calculation from atomic model.
        """
        start_time = time.time()
        logger.info("Starting geometry gradient calculation...")

        atomic_model = input_bundles["xray_atomic_model_data"]

        # Extract CCTBX objects
        xray_structure = atomic_model.get_asset("xray_structure")
        pdb_file = atomic_model.get_asset("pdb_file") if atomic_model.has_asset("pdb_file") else None
        
        logger.info(f"X-ray structure has {len(xray_structure.scatterers())} atoms")

        # Get coordinates
        coordinates = xray_structure.sites_cart()
        logger.info(f"Extracted coordinates for {coordinates.size()} atoms")

        # Create model manager and process for geometry restraints
        logger.info("Creating model manager and processing geometry restraints...")
        model, grm = self._create_geometry_restraints_manager(xray_structure, pdb_file)
        
        # Compute geometry gradients
        logger.info("Computing geometry gradients...")
        geometric_gradients = self._compute_geometry_gradients(grm, coordinates)
        logger.info(f"Geometry gradients computed for {geometric_gradients.size()} atoms")

        # Create output bundle
        geometry_bundle = Bundle(bundle_type="geometry_gradient_data")
        geometry_bundle.add_asset("coordinates", coordinates)
        geometry_bundle.add_asset("geometric_gradients", geometric_gradients)

        # Optional: Compute restraint energies and counts
        logger.info("Computing restraint energies and counts...")
        restraint_energies = self._compute_restraint_energies(grm)
        restraint_counts = self._compute_restraint_counts(grm)
        
        geometry_bundle.add_asset("restraint_energies", restraint_energies)
        geometry_bundle.add_asset("restraint_counts", restraint_counts)

        # Add metadata
        computation_time = time.time() - start_time
        geometry_metadata = self._compute_geometry_metadata(
            geometric_gradients, restraint_energies, restraint_counts, computation_time
        )
        geometry_bundle.add_asset("geometry_metadata", geometry_metadata)

        logger.info(f"Geometry gradient calculation completed in {computation_time:.2f}s")
        return {"geometry_gradient_data": geometry_bundle}

    def _create_geometry_restraints_manager(self, xray_structure: Any, pdb_file: str = None) -> tuple[Any, Any]:
        """
        Create a model manager and geometry restraints manager from xray structure or PDB file.
        
        Args:
            xray_structure: CCTBX xray structure object
            pdb_file: Path to the PDB file (optional)
            
        Returns:
            tuple: (model, grm) - Model manager and geometry restraints manager
        """
        import iotbx.pdb
        import mmtbx.model
        from libtbx.utils import null_out

        if pdb_file is not None and os.path.exists(pdb_file):
            pdb_inp = iotbx.pdb.input(file_name=pdb_file)
        else:
            # Fallback: try to use xray_structure (may not work for all cases)
            pdb_inp = iotbx.pdb.input(source_info=None, lines=[])
            pdb_inp._xray_structure = xray_structure

        # Create model manager
        model = mmtbx.model.manager(model_input=pdb_inp, log=null_out())
        
        # Process model to create geometry restraints
        model.process(make_restraints=True)
        
        # Get geometry restraints manager
        grm = model.get_restraints_manager().geometry
        
        return model, grm

    def _compute_geometry_gradients(self, grm: Any, coordinates: Any) -> Any:
        """
        Compute geometry gradients with respect to coordinates.
        
        Args:
            grm: Geometry restraints manager
            coordinates: Atomic coordinates
            
        Returns:
            Geometric gradients as flex.vec3_double
        """
        from cctbx.array_family import flex
        
        # Initialize gradients to zero
        gradients = flex.vec3_double(coordinates.size(), (0.0, 0.0, 0.0))
        
        # Compute gradients for each type of restraint
        # Note: This is a simplified approach. In practice, you might want to use
        # CCTBX's built-in gradient computation methods if available
        
        # For now, we'll compute a simple gradient based on restraint violations
        # This is a placeholder - you'll want to replace this with actual CCTBX gradient computation
        
        # Example: Compute bond length gradients
        if hasattr(grm, 'bond_proxies') and grm.bond_proxies.size() > 0:
            bond_gradients = self._compute_bond_gradients(grm, coordinates)
            gradients += bond_gradients
            
        # Example: Compute angle gradients  
        if hasattr(grm, 'angle_proxies') and grm.angle_proxies.size() > 0:
            angle_gradients = self._compute_angle_gradients(grm, coordinates)
            gradients += angle_gradients
            
        return gradients

    def _compute_bond_gradients(self, grm: Any, coordinates: Any) -> Any:
        """
        Compute gradients from bond length restraints.
        
        Args:
            grm: Geometry restraints manager
            coordinates: Atomic coordinates
            
        Returns:
            Bond gradients as flex.vec3_double
        """
        from cctbx.array_family import flex
        import math
        
        gradients = flex.vec3_double(coordinates.size(), (0.0, 0.0, 0.0))
        
        # This is a simplified bond gradient computation
        # In practice, you'd use CCTBX's actual gradient computation
        for proxy in grm.bond_proxies:
            i_seq, j_seq = proxy.i_seqs
            if i_seq < coordinates.size() and j_seq < coordinates.size():
                # Simple harmonic bond gradient
                r_vec = coordinates[j_seq] - coordinates[i_seq]
                r_length = math.sqrt(r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2)
                if r_length > 0:
                    # Simplified gradient (you'd want the actual CCTBX computation here)
                    force_constant = 1.0  # Placeholder
                    target_distance = 1.5  # Placeholder
                    gradient_magnitude = force_constant * (r_length - target_distance)
                    
                    # Apply gradient to both atoms
                    unit_vec = (r_vec[0]/r_length, r_vec[1]/r_length, r_vec[2]/r_length)
                    gradients[i_seq] = (gradients[i_seq][0] - gradient_magnitude * unit_vec[0],
                                      gradients[i_seq][1] - gradient_magnitude * unit_vec[1],
                                      gradients[i_seq][2] - gradient_magnitude * unit_vec[2])
                    gradients[j_seq] = (gradients[j_seq][0] + gradient_magnitude * unit_vec[0],
                                      gradients[j_seq][1] + gradient_magnitude * unit_vec[1],
                                      gradients[j_seq][2] + gradient_magnitude * unit_vec[2])
        
        return gradients

    def _compute_angle_gradients(self, grm: Any, coordinates: Any) -> Any:
        """
        Compute gradients from angle restraints.
        
        Args:
            grm: Geometry restraints manager
            coordinates: Atomic coordinates
            
        Returns:
            Angle gradients as flex.vec3_double
        """
        from cctbx.array_family import flex
        
        # Placeholder for angle gradient computation
        # In practice, you'd implement the actual angle gradient computation
        gradients = flex.vec3_double(coordinates.size(), (0.0, 0.0, 0.0))
        
        return gradients

    def _compute_restraint_energies(self, grm: Any) -> Dict[str, float]:
        """
        Compute restraint energies by type.
        
        Args:
            grm: Geometry restraints manager
            
        Returns:
            Dictionary of restraint energies
        """
        energies = {
            "bond_energy": 0.0,
            "angle_energy": 0.0,
            "dihedral_energy": 0.0,
            "chirality_energy": 0.0,
            "planarity_energy": 0.0,
            "total_energy": 0.0
        }
        
        # Placeholder energy computation
        # In practice, you'd compute actual energies from the restraints
        
        total_energy = sum(energies.values())
        energies["total_energy"] = total_energy
        
        return energies

    def _compute_restraint_counts(self, grm: Any) -> Dict[str, int]:
        """
        Compute restraint counts by type.
        
        Args:
            grm: Geometry restraints manager
            
        Returns:
            Dictionary of restraint counts
        """
        counts = {
            "bond_proxies": 0,
            "angle_proxies": 0,
            "dihedral_proxies": 0,
            "chirality_proxies": 0,
            "planarity_proxies": 0,
            "total_restraints": 0
        }
        
        # Count restraints by type
        if hasattr(grm, 'bond_proxies'):
            counts["bond_proxies"] = grm.bond_proxies.size()
        
        if hasattr(grm, 'angle_proxies'):
            counts["angle_proxies"] = grm.angle_proxies.size()
            
        if hasattr(grm, 'dihedral_proxies'):
            counts["dihedral_proxies"] = grm.dihedral_proxies.size()
            
        if hasattr(grm, 'chirality_proxies'):
            counts["chirality_proxies"] = grm.chirality_proxies.size()
            
        if hasattr(grm, 'planarity_proxies'):
            counts["planarity_proxies"] = grm.planarity_proxies.size()
        
        total_restraints = sum(counts.values())
        counts["total_restraints"] = total_restraints
        
        return counts

    def _compute_geometry_metadata(self, gradients: Any, energies: Dict[str, float], 
                                 counts: Dict[str, int], computation_time: float) -> Dict[str, Any]:
        """
        Compute metadata about the geometry computation.
        
        Args:
            gradients: Geometric gradients
            energies: Restraint energies
            counts: Restraint counts
            computation_time: Time taken for computation
            
        Returns:
            Dictionary of metadata
        """
        from cctbx.array_family import flex
        
        # Compute gradient statistics
        gradient_norms = flex.double()
        for grad in gradients:
            norm = (grad[0]**2 + grad[1]**2 + grad[2]**2)**0.5
            gradient_norms.append(norm)
        
        # Handle empty or single-value flex arrays robustly
        if gradient_norms.size() == 0:
            max_gradient = 0.0
            mean_gradient = 0.0
            gradient_norm = 0.0
        else:
            max_gradient = float(max(gradient_norms))
            mean_gradient = float(sum(gradient_norms) / gradient_norms.size())
            gradient_norm = float(sum(gradient_norms))
        
        metadata = {
            "computation_method": "cctbx_geometry_restraints",
            "restraint_types_used": [k for k, v in counts.items() if v > 0 and k != "total_restraints"],
            "gradient_norm": gradient_norm,
            "max_gradient": max_gradient,
            "mean_gradient": mean_gradient,
            "computation_time": computation_time,
            "n_atoms": gradients.size(),
            "total_restraints": counts["total_restraints"]
        }
        
        return metadata

    def get_computation_info(self) -> Dict[str, Any]:
        """Return information about this processor's computation."""
        return {
            "processor_type": "CctbxGeometryProcessor",
            "responsibility": "Geometry gradient calculation",
            "algorithms": ["cctbx_geometry_restraints", "gradient_computation"],
            "cctbx_modules": ["cctbx.geometry_restraints", "mmtbx.model"],
        }

    def get_required_input_bundle_types(self) -> List[str]:
        """Return the required input bundle types."""
        return ["xray_atomic_model_data"] 