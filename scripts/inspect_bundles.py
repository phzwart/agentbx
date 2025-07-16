#!/usr/bin/env python3
"""
Standalone script to inspect bundles in Redis.
Run this script to see what bundles are stored and their metadata.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentbx.core.redis_manager import RedisManager


def main():
    """Main function to inspect bundles."""
    print("🔍 Inspecting bundles in Redis...")
    
    # Initialize Redis manager
    redis_manager = RedisManager(host="localhost", port=6379)
    
    if not redis_manager.is_healthy():
        print("❌ Redis connection failed. Please start Redis server.")
        return
    
    try:
        # Get all bundles with metadata
        bundles_info = redis_manager.list_bundles_with_metadata()
        
        if not bundles_info:
            print("📭 No bundles found in Redis")
            return
        
        print(f"📦 Found {len(bundles_info)} bundles:")
        print("=" * 50)
        
        for bundle_info in bundles_info:
            print(f"Bundle ID: {bundle_info['bundle_id']}")
            print(f"  Type: {bundle_info['bundle_type']}")
            print(f"  Created: {bundle_info['created_at']}")
            print(f"  Size: {bundle_info['size_bytes']:,} bytes")
            print(f"  Checksum: {bundle_info['checksum']}")
            print("-" * 30)
        
        # Show summary by type
        print("\n📊 Summary by bundle type:")
        type_counts = {}
        for bundle_info in bundles_info:
            bundle_type = bundle_info['bundle_type']
            type_counts[bundle_type] = type_counts.get(bundle_type, 0) + 1
        
        for bundle_type, count in type_counts.items():
            print(f"  {bundle_type}: {count} bundles")
        
        # Offer to inspect specific bundle
        if len(bundles_info) == 1:
            bundle_id = bundles_info[0]['bundle_id']
            print(f"\n🔍 Would you like to inspect bundle {bundle_id}? (y/n): ", end="")
            if input().lower().startswith('y'):
                inspection = redis_manager.inspect_bundle(bundle_id)
                print("\nDetailed inspection:")
                print(json.dumps(inspection, indent=2, default=str))
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 