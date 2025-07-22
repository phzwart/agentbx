#!/usr/bin/env python3
"""
Test script to verify geometry minimizer consumer group setup.
"""

import asyncio
import json
import logging

# Add src to path for imports
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agentbx.core.clients.geometry_minimizer import GeometryMinimizer
from agentbx.core.processors.macromolecule_processor import MacromoleculeProcessor
from agentbx.core.redis_manager import RedisManager
from agentbx.utils.structures.coordinate_shaker import shake_coordinates_in_bundle


def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


async def test_minimizer_consumer():
    """Test the geometry minimizer consumer group setup."""
    print("=== Test Geometry Minimizer Consumer ===")

    # Initialize Redis manager
    print("1. Initializing Redis manager...")
    redis_manager = RedisManager(host="localhost", port=6379, db=0)

    if not redis_manager.is_healthy():
        print("❌ Redis connection failed. Please ensure Redis is running.")
        return False

    print("✅ Redis connection established")

    # Initialize processor and create test bundle
    print("2. Creating test macromolecule bundle...")
    processor = MacromoleculeProcessor(redis_manager, "test_processor")

    # Test file
    pdb_file = "../input.pdb"
    try:
        bundle_id = processor.create_macromolecule_bundle(pdb_file)
        print(f"   Created bundle: {bundle_id}")

        # Shake the coordinates
        bundle = processor.get_bundle(bundle_id)
        shaken_bundle = shake_coordinates_in_bundle(bundle, magnitude=0.02)
        shaken_bundle_id = redis_manager.store_bundle(shaken_bundle)
        print(f"   Shaken bundle: {shaken_bundle_id}")

    except Exception as e:
        print(f"❌ Failed to create test bundle: {e}")
        return False

    # Initialize geometry minimizer
    print("3. Initializing geometry minimizer...")
    minimizer = GeometryMinimizer(
        redis_manager=redis_manager,
        macromolecule_bundle_id=shaken_bundle_id,
        learning_rate=0.1,
        optimizer="adam",
        max_iterations=1,  # Just one iteration for testing
        convergence_threshold=1e-6,
        timeout_seconds=30.0,
    )

    print("   ✅ Geometry minimizer initialized")

    # Test the forward pass (geometry calculation)
    print("4. Testing forward pass (geometry calculation)...")
    try:
        gradients_tensor, geometry_bundle_id = await minimizer.forward()
        print(f"   ✅ Forward pass successful")
        print(f"   ✅ Geometry bundle ID: {geometry_bundle_id}")
        print(f"   ✅ Gradients tensor shape: {gradients_tensor.shape}")

        # Verify bundle exists
        geometry_bundle = redis_manager.get_bundle(geometry_bundle_id)
        print(f"   ✅ Geometry bundle retrieved successfully")

        # Check bundle contents
        assets = list(geometry_bundle.assets.keys())
        print(f"   ✅ Geometry bundle assets: {assets}")

        return True

    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run the test."""
    setup_logging()

    success = await test_minimizer_consumer()

    if success:
        print("\n🎉 Test passed!")
        sys.exit(0)
    else:
        print("\n💥 Test failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
