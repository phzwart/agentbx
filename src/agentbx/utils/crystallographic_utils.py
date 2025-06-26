"""
Utilities for handling crystallographic data files and creating bundles.
"""

import logging
import os
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple


logger = logging.getLogger(__name__)


class CrystallographicFileHandler:
    """
    Handles reading and processing of crystallographic data files.
    """

    def __init__(self) -> None:
        self.supported_pdb_extensions = {".pdb", ".ent", ".cif"}
        self.supported_mtz_extensions = {".mtz"}
        self.supported_reflection_extensions = {".hkl", ".sca", ".mtz"}

    def read_pdb_file(self, pdb_file: str) -> Any:
        """
        Read PDB file and return CCTBX xray_structure object.

        Args:
            pdb_file: Path to PDB file

        Returns:
            CCTBX xray_structure object

        Raises:
            ImportError: If CCTBX is not available
            FileNotFoundError: If PDB file doesn't exist
            ValueError: If file cannot be read
        """
        if not os.path.exists(pdb_file):
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")

        try:
            from iotbx import pdb

            logger.info(f"Reading PDB file: {pdb_file}")
            pdb_input = pdb.input(file_name=pdb_file)
            xray_structure = pdb_input.xray_structure_simple()

            logger.info(
                f"Successfully read PDB with {len(xray_structure.scatterers())} atoms"
            )
            return xray_structure

        except ImportError as e:
            raise ImportError(f"CCTBX not available: {e}") from e
        except Exception as e:
            raise ValueError(f"Error reading PDB file {pdb_file}: {e}") from e

    def read_mtz_file(self, mtz_file: str, array_index: Optional[int] = None) -> Any:
        """
        Read MTZ file and return miller array.

        Args:
            mtz_file: Path to MTZ file
            array_index: Index of miller array to return (None for auto-selection)

        Returns:
            CCTBX miller array

        Raises:
            ImportError: If CCTBX is not available
            FileNotFoundError: If MTZ file doesn't exist
            ValueError: If file cannot be read or no suitable array found
        """
        if not os.path.exists(mtz_file):
            raise FileNotFoundError(f"MTZ file not found: {mtz_file}")

        try:
            from iotbx import mtz

            logger.info(f"Reading MTZ file: {mtz_file}")
            mtz_object = mtz.object(file_name=mtz_file)
            miller_arrays = mtz_object.as_miller_arrays()

            logger.info(f"Found {len(miller_arrays)} miller arrays in MTZ file")

            if array_index is not None:
                if 0 <= array_index < len(miller_arrays):
                    selected_array = miller_arrays[array_index]
                    logger.info(
                        f"Selected array {array_index}: {selected_array.info().labels}"
                    )
                    return selected_array
                else:
                    raise ValueError(
                        f"Array index {array_index} out of range (0-{len(miller_arrays)-1})"
                    )

            # Auto-select suitable array
            for i, array in enumerate(miller_arrays):
                if array.is_complex_array() or array.is_real_array():
                    logger.info(f"Auto-selected array {i}: {array.info().labels}")
                    return array

            raise ValueError("No suitable miller array found in MTZ file")

        except ImportError as e:
            raise ImportError(f"CCTBX not available: {e}") from e
        except Exception as e:
            raise ValueError(f"Error reading MTZ file {mtz_file}: {e}") from e

    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get basic information about a crystallographic file.

        Args:
            file_path: Path to file

        Returns:
            Dict[str, Any]: Dictionary with file information

        Raises:
            FileNotFoundError: If file does not exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_info = {
            "path": file_path,
            "size_bytes": os.path.getsize(file_path),
            "extension": Path(file_path).suffix.lower(),
            "exists": True,
        }

        # Try to get crystallographic-specific info
        try:
            if file_info["extension"] in self.supported_pdb_extensions:
                xray_structure = self.read_pdb_file(file_path)
                file_info.update(
                    {
                        "type": "pdb",
                        "n_atoms": len(xray_structure.scatterers()),
                        "unit_cell": str(xray_structure.unit_cell()),
                        "space_group": str(xray_structure.space_group()),
                        "n_chains": len(xray_structure.chains()),
                    }
                )
            elif file_info["extension"] in self.supported_mtz_extensions:
                miller_array = self.read_mtz_file(file_path)
                file_info.update(
                    {
                        "type": "mtz",
                        "n_reflections": miller_array.size(),
                        "d_min": miller_array.d_min(),
                        "labels": miller_array.info().labels,
                        "is_complex": miller_array.is_complex_array(),
                        "is_real": miller_array.is_real_array(),
                    }
                )
            else:
                file_info["type"] = "unknown"

        except Exception as e:
            file_info["error"] = str(e)
            file_info["type"] = "error"

        return file_info


def create_atomic_model_bundle(
    pdb_file: str,
    mtz_file: Optional[str] = None,
    bulk_solvent_params: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Create an atomic model bundle from PDB and optional MTZ files.

    Args:
        pdb_file: Path to PDB file
        mtz_file: Optional path to MTZ file for miller indices
        bulk_solvent_params: Optional bulk solvent parameters
        metadata: Optional additional metadata

    Returns:
        Bundle with atomic model data
    """
    from ..core.bundle_base import Bundle

    handler = CrystallographicFileHandler()

    # Read PDB file
    xray_structure = handler.read_pdb_file(pdb_file)

    # Read MTZ file or create synthetic miller indices
    if mtz_file and os.path.exists(mtz_file):
        miller_indices = handler.read_mtz_file(mtz_file)
    else:
        logger.warning("No MTZ file provided, creating synthetic miller indices")
        miller_indices = _create_synthetic_miller_indices(xray_structure)

    # Create bundle
    bundle = Bundle(bundle_type="atomic_model_data")
    bundle.add_asset("xray_structure", xray_structure)
    bundle.add_asset("miller_indices", miller_indices)

    # Add default bulk solvent parameters if not provided
    if bulk_solvent_params is None:
        bulk_solvent_params = {
            "k_sol": 0.35,
            "b_sol": 50.0,
            "grid_resolution_factor": 1 / 3,
            "solvent_radius": 1.11,
        }
    bundle.add_asset("bulk_solvent_params", bulk_solvent_params)

    # Add metadata
    if metadata is None:
        metadata = {}

    metadata.update(
        {
            "pdb_file": pdb_file,
            "mtz_file": mtz_file,
            "unit_cell": str(xray_structure.unit_cell()),
            "space_group": str(xray_structure.space_group()),
            "n_atoms": len(xray_structure.scatterers()),
            "n_reflections": miller_indices.size(),
            "d_min": miller_indices.d_min(),
        }
    )

    for key, value in metadata.items():
        bundle.add_metadata(key, value)

    logger.info(
        f"Created atomic model bundle with {len(xray_structure.scatterers())} atoms"
    )
    logger.info(f"Unit cell: {xray_structure.unit_cell()}")
    logger.info(f"Space group: {xray_structure.space_group()}")
    logger.info(f"Miller indices: {miller_indices.size()} reflections")

    return bundle


def _create_synthetic_miller_indices(xray_structure: Any, d_min: float = 2.0) -> Any:
    """
    Create synthetic miller indices for testing.

    Args:
        xray_structure: CCTBX xray_structure object
        d_min: Minimum resolution in Angstroms

    Returns:
        Any: CCTBX miller array with synthetic indices

    Raises:
        ImportError: If CCTBX is not available.
        RuntimeError: If synthetic data generation fails.
    """
    try:
        import random  # nosec - Used for generating synthetic test data

        from cctbx import crystal
        from cctbx import miller
        from cctbx.array_family import flex

        # Generate synthetic miller indices
        unit_cell = xray_structure.unit_cell()
        space_group = xray_structure.space_group()

        # Create miller set with reasonable resolution
        miller_set = miller.build_set(
            crystal_symmetry=crystal.symmetry(
                unit_cell=unit_cell, space_group=space_group
            ),
            anomalous_flag=False,
            d_min=d_min,
        )

        # Generate synthetic F_obs data
        f_obs_data = flex.double(miller_set.size())
        for i in range(miller_set.size()):
            f_obs_data[i] = random.uniform(0.1, 100.0)  # nosec - Synthetic test data

        # Create miller array
        f_obs = miller_set.array(data=f_obs_data)

        # Add some metadata
        f_obs.set_info(
            miller.array_info(source="synthetic", labels=["F_obs"], wavelength=1.0)
        )

        logger.info(
            f"Generated synthetic miller indices: {f_obs.size()} reflections, d_min={d_min}A"
        )
        return f_obs

    except ImportError as e:
        raise ImportError(
            f"CCTBX not available for synthetic data generation: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed to generate synthetic miller indices: {e}") from e


def validate_crystallographic_files(
    pdb_file: str, mtz_file: Optional[str] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate crystallographic files and return status and information.

    Args:
        pdb_file: Path to PDB file
        mtz_file: Optional path to MTZ file

    Returns:
        Tuple of (is_valid, file_info_dict)
    """
    handler = CrystallographicFileHandler()
    validation_results: Dict[str, Any] = {
        "pdb_file": None,
        "mtz_file": None,
        "compatibility": None,
        "errors": [],
    }

    try:
        # Validate PDB file
        pdb_info = handler.get_file_info(pdb_file)
        validation_results["pdb_file"] = pdb_info

        if pdb_info.get("type") == "error":
            validation_results["errors"].append(
                f"PDB file error: {pdb_info.get('error')}"
            )

    except Exception as e:
        validation_results["errors"].append(f"PDB file validation failed: {e}")

    try:
        # Validate MTZ file if provided
        if mtz_file and os.path.exists(mtz_file):
            mtz_info = handler.get_file_info(mtz_file)
            validation_results["mtz_file"] = mtz_info

            if mtz_info.get("type") == "error":
                validation_results["errors"].append(
                    f"MTZ file error: {mtz_info.get('error')}"
                )

            # Check compatibility
            pdb_file_info = validation_results["pdb_file"]
            mtz_file_info = validation_results["mtz_file"]
            if (
                pdb_file_info
                and mtz_file_info
                and pdb_file_info.get("type") == "pdb"
                and mtz_file_info.get("type") == "mtz"
            ):

                validation_results["compatibility"] = {
                    "pdb_atoms": pdb_file_info.get("n_atoms"),
                    "mtz_reflections": mtz_file_info.get("n_reflections"),
                    "mtz_d_min": mtz_file_info.get("d_min"),
                }

    except Exception as e:
        validation_results["errors"].append(f"MTZ file validation failed: {e}")

    is_valid = len(validation_results["errors"]) == 0
    return is_valid, validation_results
