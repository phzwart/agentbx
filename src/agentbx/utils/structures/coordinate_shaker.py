import logging

import numpy as np
from cctbx.array_family import flex


def shake_coordinates_in_bundle(macromolecule_bundle, magnitude=0.2):
    """
    Randomly perturb (shake) the atomic coordinates in a macromolecule bundle.

    Args:
        macromolecule_bundle: Bundle containing 'xray_structure' asset
        magnitude: Maximum displacement in Angstroms

    Returns:
        The updated bundle with shaken coordinates
    """
    logger = logging.getLogger(__name__)
    xray_structure = macromolecule_bundle.get_asset("xray_structure")
    sites_cart = xray_structure.sites_cart()
    n_atoms = sites_cart.size()
    # Generate random displacements as a flex.vec3_double
    displacements = flex.vec3_double(
        np.random.normal(scale=magnitude, size=(n_atoms, 3))
    )
    # Vectorized addition
    shaken_sites = sites_cart + displacements
    xray_structure.set_sites_cart(shaken_sites)
    macromolecule_bundle.add_asset("xray_structure", xray_structure)
    logger.info(f"Shook {n_atoms} coordinates with magnitude {magnitude}")
    return macromolecule_bundle
