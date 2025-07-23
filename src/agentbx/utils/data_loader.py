"""
Clean infrastructure for reading macromolecule and MTZ data into Redis bundles.

This module provides simple, unified functions for loading crystallographic data
and getting bundle IDs back from Redis.
"""

import logging
import os
from typing import Any
from typing import Dict
from typing import Tuple

from agentbx.core.processors.experimental_data_processor import (
    ExperimentalDataProcessor,
)
from agentbx.core.processors.macromolecule_processor import MacromoleculeProcessor
from agentbx.core.redis_manager import RedisManager
from agentbx.schemas.generated import MacromoleculeDataBundle
from agentbx.utils.io.crystallographic_utils import CrystallographicFileHandler


logger = logging.getLogger(__name__)


class DataLoader:
    """
    Clean infrastructure for loading crystallographic data into Redis bundles.

    Provides simple functions to read PDB/MTZ files and get bundle IDs back.
    """

    def __init__(self, redis_manager: RedisManager):
        """
        Initialize data loader with Redis manager.

        Args:
            redis_manager: Redis manager instance
        """
        self.redis_manager = redis_manager
        self.file_handler = CrystallographicFileHandler()
        self.macromolecule_processor = MacromoleculeProcessor(
            redis_manager, "data_loader_macro"
        )
        self.experimental_processor = ExperimentalDataProcessor(
            redis_manager, "data_loader_exp"
        )

    def load_macromolecule(self, pdb_file: str) -> str:
        """
        Load macromolecule data from PDB file into Redis.

        Args:
            pdb_file: Path to PDB file

        Returns:
            Bundle ID of the created macromolecule bundle

        Raises:
            FileNotFoundError: If PDB file doesn't exist
            ValueError: If file cannot be read or processed
        """
        if not os.path.exists(pdb_file):
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")

        logger.info(f"Loading macromolecule from: {pdb_file}")

        try:
            # Use existing macromolecule processor
            bundle_id = self.macromolecule_processor.create_macromolecule_bundle(
                pdb_file
            )

            # Validate the bundle was created
            bundle = self.redis_manager.get_bundle(bundle_id)
            # Validate with schema
            MacromoleculeDataBundle(
                pdb_hierarchy=bundle.get_asset("pdb_hierarchy"),
                crystal_symmetry=bundle.get_asset("crystal_symmetry"),
                model_manager=bundle.get_asset("model_manager"),
                restraint_manager=bundle.get_asset("restraint_manager"),
                xray_structure=bundle.get_asset("xray_structure"),
                macromolecule_metadata=bundle.metadata,
            )
            print("[Schema Validation] MacromoleculeDataBundle validation successful.")
            logger.info(f"Created macromolecule bundle: {bundle_id}")
            logger.info(f"Bundle contains {len(bundle.assets)} assets")

            return bundle_id

        except Exception as e:
            logger.error(f"Failed to load macromolecule from {pdb_file}: {e}")
            raise ValueError(f"Failed to load macromolecule: {e}") from e

    def load_experimental_data(
        self,
        mtz_file: str,
        f_obs_label: str = "FP",
        sigma_label: str = "SIGFP",
        r_free_label: str = "FreeR_flag",
    ) -> str:
        """
        Load experimental data from MTZ file into Redis.

        Args:
            mtz_file: Path to MTZ file
            f_obs_label: Label for F_obs data in MTZ
            sigma_label: Label for sigma data in MTZ
            r_free_label: Label for R_free flags in MTZ

        Returns:
            Bundle ID of the created experimental data bundle

        Raises:
            FileNotFoundError: If MTZ file doesn't exist
            ValueError: If file cannot be read or processed
        """
        if not os.path.exists(mtz_file):
            raise FileNotFoundError(f"MTZ file not found: {mtz_file}")

        logger.info(f"Loading experimental data from: {mtz_file}")

        try:
            # Use existing experimental data processor
            bundle_id = self.experimental_processor.process_mtz_file(
                mtz_file, f_obs_label, sigma_label, r_free_label
            )

            # Validate the bundle was created
            bundle = self.redis_manager.get_bundle(bundle_id)
            logger.info(f"Created experimental data bundle: {bundle_id}")
            logger.info(f"Bundle contains {len(bundle.assets)} assets")

            return bundle_id

        except Exception as e:
            logger.error(f"Failed to load experimental data from {mtz_file}: {e}")
            raise ValueError(f"Failed to load experimental data: {e}") from e

    def load_intensity_data(
        self, hkl_file: str, i_obs_label: str = "I", sigma_label: str = "SIGI"
    ) -> str:
        """
        Load intensity data from HKL file into Redis.

        Args:
            hkl_file: Path to HKL file
            i_obs_label: Label for I_obs data in HKL
            sigma_label: Label for sigma data in HKL

        Returns:
            Bundle ID of the created experimental data bundle

        Raises:
            FileNotFoundError: If HKL file doesn't exist
            ValueError: If file cannot be read or processed
        """
        if not os.path.exists(hkl_file):
            raise FileNotFoundError(f"HKL file not found: {hkl_file}")

        logger.info(f"Loading intensity data from: {hkl_file}")

        try:
            # Use existing experimental data processor
            bundle_id = self.experimental_processor.process_intensity_file(
                hkl_file, i_obs_label, sigma_label
            )

            # Validate the bundle was created
            bundle = self.redis_manager.get_bundle(bundle_id)
            logger.info(f"Created intensity data bundle: {bundle_id}")
            logger.info(f"Bundle contains {len(bundle.assets)} assets")

            return bundle_id

        except Exception as e:
            logger.error(f"Failed to load intensity data from {hkl_file}: {e}")
            raise ValueError(f"Failed to load intensity data: {e}") from e

    def load_both_data(
        self,
        pdb_file: str,
        mtz_file: str,
        f_obs_label: str = "FP",
        sigma_label: str = "SIGFP",
        r_free_label: str = "FreeR_flag",
    ) -> Tuple[str, str]:
        """
        Load both macromolecule and experimental data.

        Args:
            pdb_file: Path to PDB file
            mtz_file: Path to MTZ file
            f_obs_label: Label for F_obs data in MTZ
            sigma_label: Label for sigma data in MTZ
            r_free_label: Label for R_free flags in MTZ

        Returns:
            Tuple of (macromolecule_bundle_id, experimental_bundle_id)

        """
        logger.info("Loading both macromolecule and experimental data")
        logger.info(f"PDB: {pdb_file}")
        logger.info(f"MTZ: {mtz_file}")

        # Load macromolecule data
        macro_bundle_id = self.load_macromolecule(pdb_file)

        # Load experimental data
        exp_bundle_id = self.load_experimental_data(
            mtz_file, f_obs_label, sigma_label, r_free_label
        )

        logger.info("Successfully loaded both datasets:")
        logger.info(f"  Macromolecule bundle: {macro_bundle_id}")
        logger.info(f"  Experimental bundle: {exp_bundle_id}")

        return macro_bundle_id, exp_bundle_id

    def validate_data_compatibility(
        self, macro_bundle_id: str, exp_bundle_id: str
    ) -> Dict[str, Any]:
        """
        Validate compatibility between macromolecule and experimental data.

        Args:
            macro_bundle_id: Macromolecule bundle ID
            exp_bundle_id: Experimental data bundle ID

        Returns:
            Dictionary with validation results
        """
        logger.info("Validating data compatibility...")

        try:
            # Get bundles
            macro_bundle = self.redis_manager.get_bundle(macro_bundle_id)
            exp_bundle = self.redis_manager.get_bundle(exp_bundle_id)

            # Extract key data
            xray_structure = macro_bundle.get_asset("xray_structure")
            f_obs = exp_bundle.get_asset("f_obs")

            # Check crystal symmetry compatibility
            macro_symmetry = xray_structure.crystal_symmetry()
            exp_symmetry = f_obs.crystal_symmetry()

            compatibility = {
                "unit_cell_match": str(macro_symmetry.unit_cell())
                == str(exp_symmetry.unit_cell()),
                "space_group_match": str(macro_symmetry.space_group())
                == str(exp_symmetry.space_group()),
                "macro_atoms": len(xray_structure.scatterers()),
                "exp_reflections": f_obs.size(),
                "exp_resolution": f_obs.d_min(),
                "compatible": True,
            }

            # Check for mismatches
            if not compatibility["unit_cell_match"]:
                logger.warning(
                    "Unit cell mismatch between macromolecule and experimental data"
                )
                compatibility["compatible"] = False

            if not compatibility["space_group_match"]:
                logger.warning(
                    "Space group mismatch between macromolecule and experimental data"
                )
                compatibility["compatible"] = False

            logger.info("Data compatibility validation completed")
            return compatibility

        except Exception as e:
            logger.error(f"Failed to validate data compatibility: {e}")
            return {"compatible": False, "error": str(e)}

    def get_bundle_info(self, bundle_id: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a bundle.

        Args:
            bundle_id: Bundle ID to inspect

        Returns:
            Dictionary with bundle information
        """
        try:
            bundle = self.redis_manager.get_bundle(bundle_id)
            metadata = self.redis_manager.get_bundle_metadata(bundle_id)

            info = {
                "bundle_id": bundle_id,
                "bundle_type": bundle.bundle_type,
                "created_at": metadata.get("created_at"),
                "size_bytes": metadata.get("size_bytes"),
                "assets": list(bundle.assets.keys()),
                "metadata_keys": list(bundle.metadata.keys()),
            }

            # Add type-specific information
            if bundle.bundle_type == "macromolecule_data":
                if bundle.has_asset("xray_structure"):
                    xray_structure = bundle.get_asset("xray_structure")
                    info["n_atoms"] = len(xray_structure.scatterers())
                    info["unit_cell"] = str(xray_structure.unit_cell())
                    info["space_group"] = str(xray_structure.space_group())

            elif bundle.bundle_type == "experimental_data":
                if bundle.has_asset("f_obs"):
                    f_obs = bundle.get_asset("f_obs")
                    info["n_reflections"] = f_obs.size()
                    info["resolution"] = f_obs.d_min()
                    info["unit_cell"] = str(f_obs.unit_cell())
                    info["space_group"] = str(f_obs.space_group())

            return info

        except Exception as e:
            logger.error(f"Failed to get bundle info for {bundle_id}: {e}")
            return {"error": str(e)}

    def cleanup_bundles(self, bundle_ids: list[str]) -> Dict[str, bool]:
        """
        Clean up bundles from Redis.

        Args:
            bundle_ids: List of bundle IDs to delete

        Returns:
            Dictionary mapping bundle IDs to deletion success status
        """
        results = {}

        for bundle_id in bundle_ids:
            try:
                success = self.redis_manager.delete_bundle(bundle_id)
                results[bundle_id] = success
                if success:
                    logger.info(f"Deleted bundle: {bundle_id}")
                else:
                    logger.warning(f"Bundle not found for deletion: {bundle_id}")
            except Exception as e:
                logger.error(f"Failed to delete bundle {bundle_id}: {e}")
                results[bundle_id] = False

        return results


# Convenience functions for simple usage
def load_macromolecule(redis_manager: RedisManager, pdb_file: str) -> str:
    """
    Simple function to load macromolecule data.

    Args:
        redis_manager: Redis manager instance
        pdb_file: Path to PDB file

    Returns:
        Bundle ID
    """
    loader = DataLoader(redis_manager)
    return loader.load_macromolecule(pdb_file)


def load_experimental_data(
    redis_manager: RedisManager,
    mtz_file: str,
    f_obs_label: str = "FP",
    sigma_label: str = "SIGFP",
    r_free_label: str = "FreeR_flag",
) -> str:
    """
    Simple function to load experimental data.

    Args:
        redis_manager: Redis manager instance
        mtz_file: Path to MTZ file
        f_obs_label: Label for F_obs data
        sigma_label: Label for sigma data
        r_free_label: Label for R_free flags

    Returns:
        Bundle ID
    """
    loader = DataLoader(redis_manager)
    return loader.load_experimental_data(
        mtz_file, f_obs_label, sigma_label, r_free_label
    )


def load_both_data(
    redis_manager: RedisManager,
    pdb_file: str,
    mtz_file: str,
    f_obs_label: str = "FP",
    sigma_label: str = "SIGFP",
    r_free_label: str = "FreeR_flag",
) -> Tuple[str, str]:
    """
    Simple function to load both macromolecule and experimental data.

    Args:
        redis_manager: Redis manager instance
        pdb_file: Path to PDB file
        mtz_file: Path to MTZ file
        f_obs_label: Label for F_obs data
        sigma_label: Label for sigma data
        r_free_label: Label for R_free flags

    Returns:
        Tuple of (macromolecule_bundle_id, experimental_bundle_id)
    """
    loader = DataLoader(redis_manager)
    return loader.load_both_data(
        pdb_file, mtz_file, f_obs_label, sigma_label, r_free_label
    )
