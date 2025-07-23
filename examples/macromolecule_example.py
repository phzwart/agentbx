#!/usr/bin/env python3
"""
Example demonstrating the new macromolecule paradigm.

This example shows how to:
1. Create a macromolecule bundle from a PDB file
2. Get geometry gradients from the macromolecule bundle
3. Update coordinates and recalculate gradients
"""

import logging
import os

from agentbx.core.redis_manager import RedisManager
from agentbx.processors.geometry_processor import CctbxGeometryProcessor
from agentbx.processors.macromolecule_processor import MacromoleculeProcessor


# Add src to path for imports


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def dosomething(obj):
    """Mock function to simulate usage."""
    # Mock function to simulate usage
    pass


def main():
    """Demonstrate macromolecule paradigm."""
    logger.info("Starting Macromolecule Example")

    # Check for input files
    pdb_file = "examples/input.pdb"

    if not os.path.exists(pdb_file):
        logger.error(f"PDB file not found: {pdb_file}")
        logger.info("Please run: python examples/download_pdb_data.py 1ubq")
        return

    # Initialize Redis manager
    try:
        redis_manager = RedisManager(host="localhost", port=6379)
        logger.info("Redis manager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Redis manager: {e}")
        return

    # Create macromolecule processor
    macromolecule_processor = MacromoleculeProcessor(redis_manager, "macro_processor")

    # Create macromolecule bundle from PDB file
    logger.info("Creating macromolecule bundle...")
    macromolecule_bundle_id = macromolecule_processor.create_macromolecule_bundle(
        pdb_file
    )
    logger.info(f"Macromolecule bundle created: {macromolecule_bundle_id}")

    # Get the macromolecule bundle to inspect it
    macromolecule_bundle = redis_manager.get_bundle(macromolecule_bundle_id)
    logger.info(f"Bundle type: {macromolecule_bundle.bundle_type}")
    logger.info(f"Available assets: {list(macromolecule_bundle.assets.keys())}")

    # Get xray_structure from macromolecule bundle
    xray_structure = macromolecule_processor.get_xray_structure(macromolecule_bundle_id)
    logger.info(f"X-ray structure has {len(xray_structure.scatterers())} atoms")

    # Get geometry restraints from macromolecule bundle
    geometry_restraints = macromolecule_processor.get_geometry_restraints(
        macromolecule_bundle_id
    )
    dosomething(geometry_restraints)
    logger.info("Geometry restraints available")

    # Create geometry processor
    geo_processor = CctbxGeometryProcessor(redis_manager, "geo_processor")

    # Calculate geometry gradients
    logger.info("Calculating geometry gradients...")
    geometry_bundle_id = geo_processor.calculate_geometry_gradients(
        macromolecule_bundle_id
    )
    logger.info(f"Geometry gradients calculated: {geometry_bundle_id}")

    # Get the geometry bundle to inspect results
    geometry_bundle = redis_manager.get_bundle(geometry_bundle_id)
    geometry_gradients = geometry_bundle.get_asset("geometry_gradients")
    gradient_norm = geometry_bundle.get_asset("gradient_norm")

    logger.info(f"Geometry gradients shape: {geometry_gradients.size()}")
    logger.info(f"Gradient norm: {gradient_norm}")

    # Demonstrate coordinate update
    logger.info("Demonstrating coordinate update...")

    # Get current coordinates
    current_coordinates = xray_structure.sites_cart()

    # Create a small perturbation (for demonstration)
    import random

    from cctbx.array_family import flex

    # Add small random perturbation to coordinates
    perturbation = flex.vec3_double(current_coordinates.size())
    for i in range(current_coordinates.size()):
        perturbation[i] = (
            random.uniform(-0.1, 0.1),  # Small perturbation in X
            random.uniform(-0.1, 0.1),  # Small perturbation in Y
            random.uniform(-0.1, 0.1),  # Small perturbation in Z
        )

    new_coordinates = current_coordinates + perturbation

    # Update coordinates in macromolecule bundle
    updated_macromolecule_id = macromolecule_processor.update_coordinates(
        macromolecule_bundle_id, new_coordinates
    )
    logger.info(f"Updated macromolecule bundle: {updated_macromolecule_id}")

    # Recalculate geometry gradients with updated coordinates
    updated_geometry_bundle_id = geo_processor.calculate_geometry_gradients(
        updated_macromolecule_id
    )
    logger.info(f"Updated geometry gradients: {updated_geometry_bundle_id}")

    # Compare gradient norms
    updated_geometry_bundle = redis_manager.get_bundle(updated_geometry_bundle_id)
    updated_gradient_norm = updated_geometry_bundle.get_asset("gradient_norm")

    logger.info(f"Original gradient norm: {gradient_norm}")
    logger.info(f"Updated gradient norm: {updated_gradient_norm}")

    # Clean up
    try:
        redis_manager.close()
        logger.info("Redis connection closed")
    except Exception as e:
        logger.error(f"Error closing Redis connection: {e}")

    logger.info("Macromolecule example completed successfully!")


if __name__ == "__main__":
    main()
