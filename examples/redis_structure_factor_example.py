"""
Example demonstrating Redis manager with StructureFactorAgent using real PDB and MTZ files.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agentbx.core.redis_manager import RedisManager
from agentbx.agents.structure_factor_agent import StructureFactorAgent
from agentbx.utils.crystallographic_utils import create_atomic_model_bundle, validate_crystallographic_files
from agentbx.utils.data_analysis_utils import analyze_bundle, print_analysis_summary
from agentbx.utils.workflow_utils import execute_structure_factor_workflow
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main example function."""
    logger.info("Starting Redis Structure Factor Agent Example with Real Data")
    
    # Check for input files
    pdb_file = "examples/input.pdb"
    mtz_file = "examples/input.mtz"
    
    if not os.path.exists(pdb_file):
        logger.error(f"PDB file not found: {pdb_file}")
        logger.info("Download data first: python examples/download_pdb_data.py <pdb_code>")
        return
    
    if not os.path.exists(mtz_file):
        logger.error(f"MTZ file not found: {mtz_file}")
        logger.info("Download data first: python examples/download_pdb_data.py <pdb_code>")
        return
    
    # Validate input files
    logger.info("Validating input files...")
    is_valid, validation_results = validate_crystallographic_files(pdb_file, mtz_file)
    
    if not is_valid:
        logger.error("File validation failed:")
        for error in validation_results["errors"]:
            logger.error(f"  - {error}")
        return
    
    logger.info("✅ Input files validated successfully")
    
    # Initialize Redis manager
    try:
        redis_manager = RedisManager(
            host="localhost",
            port=6379,
            db=0,
            default_ttl=3600  # 1 hour
        )
        logger.info("Redis manager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Redis manager: {e}")
        logger.info("Make sure Redis is running: redis-server")
        return
    
    # Check Redis health
    if not redis_manager.is_healthy():
        logger.error("Redis connection is not healthy")
        return
    
    # Option 1: Use the workflow utility (recommended)
    logger.info("\n=== Using Workflow Utility ===")
    try:
        output_bundle_ids = execute_structure_factor_workflow(redis_manager, pdb_file, mtz_file)
        logger.info(f"Workflow completed successfully. Output bundle IDs: {output_bundle_ids}")
        
        # Analyze the results
        sf_bundle_id = output_bundle_ids["structure_factor_data"]
        sf_bundle = redis_manager.get_bundle(sf_bundle_id)
        
        logger.info("\n=== Structure Factor Bundle Analysis ===")
        analysis = analyze_bundle(sf_bundle)
        print_analysis_summary(analysis)
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        return
    
    # Option 2: Manual execution (for comparison)
    logger.info("\n=== Manual Execution (for comparison) ===")
    try:
        # Create structure factor agent
        agent = StructureFactorAgent(redis_manager, "sf_agent_manual")
        logger.info(f"Created agent: {agent.get_client_info()}")
        
        # Create atomic model bundle using utility
        atomic_model_bundle = create_atomic_model_bundle(pdb_file, mtz_file)
        logger.info(f"Created atomic model bundle using utility")
        
        # Store input bundle in Redis
        input_bundle_id = redis_manager.store_bundle(atomic_model_bundle)
        logger.info(f"Stored atomic model bundle with ID: {input_bundle_id}")
        
        # Run the agent
        input_bundle_ids = {"atomic_model_data": input_bundle_id}
        output_bundle_ids = agent.run(input_bundle_ids)
        logger.info(f"Manual execution completed. Output bundle IDs: {output_bundle_ids}")
        
    except Exception as e:
        logger.error(f"Manual execution failed: {e}")
        return
    
    # List all bundles in Redis
    try:
        all_bundles = redis_manager.list_bundles()
        logger.info(f"\nAll bundles in Redis: {all_bundles}")
        
        # List bundles by type
        atomic_bundles = redis_manager.list_bundles("atomic_model_data")
        sf_bundles = redis_manager.list_bundles("structure_factor_data")
        logger.info(f"Atomic model bundles: {len(atomic_bundles)}")
        logger.info(f"Structure factor bundles: {len(sf_bundles)}")
        
    except Exception as e:
        logger.error(f"Failed to list bundles: {e}")
    
    # Clean up
    try:
        redis_manager.close()
        logger.info("Redis connection closed")
    except Exception as e:
        logger.error(f"Error closing Redis connection: {e}")


if __name__ == "__main__":
    main() 