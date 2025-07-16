"""
Example script showing how to inspect bundle content and metadata in Redis.
"""

import logging
import sys
from pprint import pprint

# Add src to path for imports
sys.path.insert(0, "src")

from agentbx.core.redis_manager import RedisManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def inspect_all_bundles():
    """Inspect all bundles in Redis with their metadata."""
    redis_manager = RedisManager(host="localhost", port=6379)
    
    try:
        # Method 1: List bundles with metadata
        logger.info("=== Listing all bundles with metadata ===")
        bundles_with_metadata = redis_manager.list_bundles_with_metadata()
        
        if not bundles_with_metadata:
            logger.info("No bundles found in Redis")
            return
            
        for bundle_info in bundles_with_metadata:
            logger.info(f"Bundle ID: {bundle_info['bundle_id']}")
            logger.info(f"  Type: {bundle_info['bundle_type']}")
            logger.info(f"  Created: {bundle_info['created_at']}")
            logger.info(f"  Size: {bundle_info['size_bytes']} bytes")
            logger.info(f"  Checksum: {bundle_info['checksum']}")
            logger.info("---")
            
    except Exception as e:
        logger.error(f"Error inspecting bundles: {e}")


def inspect_bundle_by_type(bundle_type: str):
    """Inspect bundles of a specific type."""
    redis_manager = RedisManager(host="localhost", port=6379)
    
    try:
        logger.info(f"=== Inspecting {bundle_type} bundles ===")
        bundles_info = redis_manager.list_bundles_with_metadata(bundle_type)
        
        if not bundles_info:
            logger.info(f"No {bundle_type} bundles found")
            return
            
        for bundle_info in bundles_info:
            logger.info(f"Bundle ID: {bundle_info['bundle_id']}")
            logger.info(f"  Created: {bundle_info['created_at']}")
            logger.info(f"  Size: {bundle_info['size_bytes']} bytes")
            
    except Exception as e:
        logger.error(f"Error inspecting {bundle_type} bundles: {e}")


def detailed_bundle_inspection(bundle_id: str):
    """Perform detailed inspection of a specific bundle."""
    redis_manager = RedisManager(host="localhost", port=6379)
    
    try:
        logger.info(f"=== Detailed inspection of bundle {bundle_id} ===")
        inspection = redis_manager.inspect_bundle(bundle_id)
        
        logger.info("Metadata:")
        pprint(inspection['metadata'])
        
        logger.info("\nContent Analysis:")
        pprint(inspection['content_analysis'])
        
    except Exception as e:
        logger.error(f"Error inspecting bundle {bundle_id}: {e}")


def get_bundle_metadata_only(bundle_id: str):
    """Get just the metadata for a bundle without loading the full content."""
    redis_manager = RedisManager(host="localhost", port=6379)
    
    try:
        logger.info(f"=== Metadata for bundle {bundle_id} ===")
        metadata = redis_manager.get_bundle_metadata(bundle_id)
        pprint(metadata)
        
    except Exception as e:
        logger.error(f"Error getting metadata for bundle {bundle_id}: {e}")


def main():
    """Main example function."""
    logger.info("Starting Bundle Inspection Example")
    
    # Check if Redis is available
    redis_manager = RedisManager(host="localhost", port=6379)
    if not redis_manager.is_healthy():
        logger.error("Redis connection is not healthy. Please start Redis server.")
        return
    
    # Method 1: Inspect all bundles
    inspect_all_bundles()
    
    # Method 2: Inspect bundles by type
    inspect_bundle_by_type("experimental_data")
    inspect_bundle_by_type("structure_factor_data")
    
    # Method 3: Get metadata for a specific bundle (if you know the ID)
    # Uncomment and replace with actual bundle ID:
    # get_bundle_metadata_only("your_bundle_id_here")
    
    # Method 4: Detailed inspection of a specific bundle (if you know the ID)
    # Uncomment and replace with actual bundle ID:
    # detailed_bundle_inspection("your_bundle_id_here")


if __name__ == "__main__":
    main() 