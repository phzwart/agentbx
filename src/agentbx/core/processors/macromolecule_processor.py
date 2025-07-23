"""
Processor responsible for macromolecule data management.

Input: pdb_file
Output: macromolecule_data

This is the central processor that creates and manages macromolecule bundles
with PDB hierarchy, from which other processors derive xray_structure and geometry restraints.
"""

import logging
from typing import Any
from typing import Dict
from typing import List

from agentbx.core.bundle_base import Bundle
from agentbx.schemas.generated import MacromoleculeDataBundle

from .base import SinglePurposeProcessor


logger = logging.getLogger(__name__)


class MacromoleculeProcessor(SinglePurposeProcessor):
    """
    Central macromolecule processor.

    Responsibility: Create and manage macromolecule bundles with PDB hierarchy.
    """

    def __init__(self, redis_manager: Any, processor_id: str) -> None:
        """
        Initialize the macromolecule processor.
        """
        super().__init__(redis_manager, processor_id)

    def define_input_bundle_types(self) -> List[str]:
        """Define the input bundle types for this processor."""
        return []  # Takes PDB file directly, not a bundle

    def define_output_bundle_types(self) -> List[str]:
        """Define the output bundle types for this processor."""
        return ["macromolecule_data"]

    def create_macromolecule_bundle(self, pdb_file: str) -> str:
        """
        Create a macromolecule bundle from a PDB file.

        Args:
            pdb_file: Path to PDB file

        Returns:
            Bundle ID of the created macromolecule bundle

        Raises:
            Exception: If there is an error creating the macromolecule bundle.
        """
        try:
            import iotbx.pdb
            import mmtbx.model
            from libtbx.utils import null_out

            # Read PDB file and create hierarchy
            pdb_input = iotbx.pdb.input(file_name=pdb_file)
            pdb_hierarchy = pdb_input.construct_hierarchy()
            crystal_symmetry = pdb_input.crystal_symmetry()

            # Create model manager with geometry restraints
            model_manager = mmtbx.model.manager(model_input=pdb_input, log=null_out())
            model_manager.process(make_restraints=True)

            # Get restraint manager
            restraint_manager = model_manager.get_restraints_manager()

            # Create xray_structure from hierarchy
            xray_structure = pdb_hierarchy.extract_xray_structure(
                crystal_symmetry=crystal_symmetry
            )

            # Create macromolecule bundle
            macromolecule_bundle = Bundle(bundle_type="macromolecule_data")
            macromolecule_bundle.add_asset("pdb_hierarchy", pdb_hierarchy)
            macromolecule_bundle.add_asset("crystal_symmetry", crystal_symmetry)
            macromolecule_bundle.add_asset("xray_structure", xray_structure)
            macromolecule_bundle.add_asset("model_manager", model_manager)
            macromolecule_bundle.add_asset("restraint_manager", restraint_manager)

            # Add metadata
            macromolecule_bundle.add_metadata("source_file", pdb_file)
            macromolecule_bundle.add_metadata(
                "n_atoms", len(list(pdb_hierarchy.atoms()))
            )
            macromolecule_bundle.add_metadata(
                "n_chains", len(list(pdb_hierarchy.chains()))
            )
            macromolecule_bundle.add_metadata(
                "unit_cell", str(crystal_symmetry.unit_cell())
            )
            macromolecule_bundle.add_metadata(
                "space_group", str(crystal_symmetry.space_group())
            )
            macromolecule_bundle.add_metadata("dialect", "cctbx")

            # Validate with schema
            MacromoleculeDataBundle(
                pdb_hierarchy=pdb_hierarchy,
                crystal_symmetry=crystal_symmetry,
                model_manager=model_manager,
                restraint_manager=restraint_manager,
                xray_structure=xray_structure,
                macromolecule_metadata=macromolecule_bundle.metadata,
            )
            print("[Schema Validation] MacromoleculeDataBundle validation successful.")
            # Store bundle
            bundle_id = self.store_bundle(macromolecule_bundle)

            logger.info(
                f"Created macromolecule bundle {bundle_id} with {len(pdb_hierarchy.atoms())} atoms"
            )
            return bundle_id

        except Exception as e:
            logger.error(f"Error creating macromolecule bundle: {e}")
            raise

    def update_coordinates(
        self, macromolecule_bundle_id: str, new_coordinates: Any
    ) -> str:
        """
        Update coordinates in the macromolecule bundle.

        Args:
            macromolecule_bundle_id: ID of the macromolecule bundle
            new_coordinates: New coordinates (flex.vec3_double)

        Returns:
            Updated bundle ID

        Raises:
            Exception: If there is an error updating the coordinates.
        """
        try:
            # Get the macromolecule bundle
            macromolecule_bundle = self.get_bundle(macromolecule_bundle_id)

            # Update coordinates in all representations
            pdb_hierarchy = macromolecule_bundle.get_asset("pdb_hierarchy")
            model_manager = macromolecule_bundle.get_asset("model_manager")
            xray_structure = macromolecule_bundle.get_asset("xray_structure")

            # Update PDB hierarchy coordinates
            pdb_hierarchy.atoms().set_xyz(new_coordinates)

            # Update model manager coordinates
            model_manager.set_sites_cart(new_coordinates)

            # Update xray_structure coordinates
            xray_structure.set_sites_cart(new_coordinates)

            # Store updated bundle
            updated_bundle_id = self.store_bundle(macromolecule_bundle)

            logger.info(
                f"Updated coordinates in macromolecule bundle {updated_bundle_id}"
            )
            return updated_bundle_id

        except Exception as e:
            logger.error(f"Error updating coordinates: {e}")
            raise

    def get_xray_structure(self, macromolecule_bundle_id: str) -> Any:
        """
        Get xray_structure from macromolecule bundle.

        Args:
            macromolecule_bundle_id: ID of the macromolecule bundle

        Returns:
            X-ray structure object
        """
        macromolecule_bundle = self.get_bundle(macromolecule_bundle_id)
        return macromolecule_bundle.get_asset("xray_structure")

    def get_geometry_restraints(self, macromolecule_bundle_id: str) -> Any:
        """
        Get geometry restraints from macromolecule bundle.

        Args:
            macromolecule_bundle_id: ID of the macromolecule bundle

        Returns:
            Geometry restraints manager
        """
        macromolecule_bundle = self.get_bundle(macromolecule_bundle_id)
        return macromolecule_bundle.get_asset("restraint_manager").geometry

    def process_bundles(self, input_bundles: Dict[str, Bundle]) -> Dict[str, Bundle]:
        """
        Process bundles (not used for macromolecule processor).
        """
        raise NotImplementedError(
            "MacromoleculeProcessor uses create_macromolecule_bundle instead"
        )

    def get_computation_info(self) -> Dict[str, Any]:
        """
        Get information about this processor's computational requirements.
        """
        return {
            "processor_type": "macromolecule_processor",
            "input_types": ["pdb_file"],
            "output_types": ["macromolecule_data"],
            "memory_usage": "medium",
            "cpu_usage": "low",
            "gpu_usage": "none",
        }
