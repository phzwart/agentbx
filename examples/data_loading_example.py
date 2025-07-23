#!/usr/bin/env python3
"""
Example demonstrating the clean data loading infrastructure.

This example shows how to:
1. Load macromolecule data from PDB file
2. Load experimental data from MTZ file
3. Validate data compatibility
4. Get bundle information
"""

import logging
import os

from agentbx.core.redis_manager import RedisManager
from agentbx.utils.data_loader import DataLoader
from agentbx.utils.data_loader import load_experimental_data
from agentbx.utils.data_loader import load_macromolecule


# Add src to path for imports


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate clean data loading infrastructure."""
    logger.info("Starting Data Loading Example")

    # Check for input files
    pdb_file = "examples/input.pdb"
    mtz_file = "examples/input.mtz"

    if not os.path.exists(pdb_file):
        logger.error(f"PDB file not found: {pdb_file}")
        logger.info("Please run: python examples/download_pdb_data.py 1ubq")
        return

    if not os.path.exists(mtz_file):
        logger.error(f"MTZ file not found: {mtz_file}")
        logger.info("Please run: python examples/download_pdb_data.py 1ubq")
        return

    # Initialize Redis manager
    try:
        redis_manager = RedisManager(host="localhost", port=6379)
        logger.info("Redis manager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Redis manager: {e}")
        return

    # Create data loader
    data_loader = DataLoader(redis_manager)

    try:
        # Method 1: Load macromolecule data only
        logger.info("\n--- Method 1: Load macromolecule data ---")
        macro_bundle_id = data_loader.load_macromolecule(pdb_file)
        logger.info(f"Macromolecule bundle ID: {macro_bundle_id}")

        # Get bundle information
        macro_info = data_loader.get_bundle_info(macro_bundle_id)
        logger.info(f"Bundle info: {macro_info}")

        # Method 2: Load experimental data only
        logger.info("\n--- Method 2: Load experimental data ---")
        exp_bundle_id = data_loader.load_experimental_data(mtz_file)
        logger.info(f"Experimental bundle ID: {exp_bundle_id}")

        # Get bundle information
        exp_info = data_loader.get_bundle_info(exp_bundle_id)
        logger.info(f"Bundle info: {exp_info}")

        # Method 3: Load both datasets together
        logger.info("\n--- Method 3: Load both datasets ---")
        macro_id, exp_id = data_loader.load_both_data(pdb_file, mtz_file)
        logger.info(f"Macromolecule bundle ID: {macro_id}")
        logger.info(f"Experimental bundle ID: {exp_id}")

        # Validate data compatibility
        logger.info("\n--- Data Compatibility Validation ---")
        compatibility = data_loader.validate_data_compatibility(macro_id, exp_id)
        logger.info(f"Compatibility results: {compatibility}")

        # Method 4: Use convenience functions
        logger.info("\n--- Method 4: Convenience functions ---")
        macro_id_simple = load_macromolecule(redis_manager, pdb_file)
        exp_id_simple = load_experimental_data(redis_manager, mtz_file)
        logger.info(f"Simple macromolecule ID: {macro_id_simple}")
        logger.info(f"Simple experimental ID: {exp_id_simple}")

        # List all bundles
        logger.info("\n--- All Bundles in Redis ---")
        all_bundles = redis_manager.list_bundles_with_metadata()
        for bundle_info in all_bundles:
            logger.info(
                f"Bundle: {bundle_info['bundle_id']} ({bundle_info['bundle_type']})"
            )

        logger.info("\n--- Success ---")
        logger.info("Data loading example completed successfully!")

    except Exception as e:
        logger.error(f"Error during data loading: {e}")
        raise

    finally:
        # Close Redis connection
        redis_manager.close()
        logger.info("Redis connection closed")


if __name__ == "__main__":
    main()
