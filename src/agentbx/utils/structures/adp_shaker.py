import logging

import numpy as np
from cctbx.array_family import flex


def shake_adps_in_bundle(macromolecule_bundle, magnitude=2.0):
    """
    Randomly perturb (shake) the ADPs (B-factors) in a macromolecule bundle.

    Args:
        macromolecule_bundle: Bundle containing 'xray_structure' asset
        magnitude: Maximum change in B-factor (in A^2)

    Returns:
        The updated bundle with shaken ADPs
    """
    logger = logging.getLogger(__name__)
    xray_structure = macromolecule_bundle.get_asset("xray_structure")
    b_factors = xray_structure.extract_b_iso_or_u_iso() * 1.0  # Copy
    n_atoms = b_factors.size()
    # Generate random perturbations as a flex.double
    perturbations = flex.double(np.random.normal(scale=magnitude, size=n_atoms))
    shaken_b = b_factors + perturbations
    # Ensure B-factors are non-negative
    shaken_b = flex.double([max(0.0, b) for b in shaken_b])
    xray_structure.set_b_iso(shaken_b)
    macromolecule_bundle.add_asset("xray_structure", xray_structure)
    logger.info(f"Shook {n_atoms} ADPs (B-factors) with magnitude {magnitude}")
    return macromolecule_bundle
