#!/usr/bin/env python3
"""
Simple test to check model_manager serialization issue.
"""

import os
import sys

from agentbx.core.processors.macromolecule_processor import MacromoleculeProcessor
from agentbx.core.redis_manager import RedisManager


# Add src to path for imports


def test_model_manager_serialization():
    """Test if model_manager is properly serialized and deserialized."""
    print("=== Testing Model Manager Serialization ===")

    # Initialize Redis manager
    redis_manager = RedisManager(host="localhost", port=6379, db=0)

    if not redis_manager.is_healthy():
        print("❌ Redis connection failed. Please ensure Redis is running.")
        return False

    print("✅ Redis connection established")

    # Initialize processor
    processor = MacromoleculeProcessor(redis_manager, "test_processor")

    # Test file
    pdb_file = "../data/small.pdb"
    if not os.path.exists(pdb_file):
        print(f"❌ Test file not found: {pdb_file}")
        return False

    try:
        # Step 1: Create macromolecule bundle
        print("1. Creating macromolecule bundle...")
        bundle_id = processor.create_macromolecule_bundle(pdb_file)
        print(f"   Created bundle: {bundle_id}")

        # Step 2: Get bundle using processor
        print("2. Getting bundle using processor...")
        bundle_from_processor = processor.get_bundle(bundle_id)
        model_manager_from_processor = bundle_from_processor.get_asset("model_manager")
        print(f"   Model manager from processor: {model_manager_from_processor}")
        print(
            f"   Model manager type from processor: {type(model_manager_from_processor)}"
        )

        # Step 3: Get bundle using redis_manager directly
        print("3. Getting bundle using redis_manager directly...")
        bundle_from_redis = redis_manager.get_bundle(bundle_id)
        model_manager_from_redis = bundle_from_redis.get_asset("model_manager")
        print(f"   Model manager from redis: {model_manager_from_redis}")
        print(f"   Model manager type from redis: {type(model_manager_from_redis)}")

        # Step 4: Compare
        print("4. Comparing results...")
        if model_manager_from_processor is None and model_manager_from_redis is None:
            print("   ❌ Both model managers are None")
            return False
        elif model_manager_from_processor is None:
            print("   ❌ Model manager from processor is None, but from redis is not")
            return False
        elif model_manager_from_redis is None:
            print("   ❌ Model manager from redis is None, but from processor is not")
            return False
        else:
            print("   ✅ Both model managers are not None")

        # Step 5: Check if they're the same object
        if model_manager_from_processor is model_manager_from_redis:
            print("   ✅ Same object reference")
        else:
            print("   ⚠ Different object references (expected after serialization)")

        # Step 6: Test if we can use the model manager
        print("6. Testing model manager functionality...")
        try:
            sites_cart = model_manager_from_redis.get_sites_cart()
            print(f"   ✅ Got sites_cart: {len(sites_cart)} sites")

            restraints_manager = model_manager_from_redis.get_restraints_manager()
            print(f"   ✅ Got restraints_manager: {restraints_manager}")

            print("   ✅ Model manager is functional")
            return True

        except Exception as e:
            print(f"   ❌ Model manager is not functional: {e}")
            return False

    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Cleanup
        print("\nCleaning up...")
        redis_manager.close()


if __name__ == "__main__":
    success = test_model_manager_serialization()

    if success:
        print("\n🎉 Test passed!")
        sys.exit(0)
    else:
        print("\n💥 Test failed!")
        sys.exit(1)
