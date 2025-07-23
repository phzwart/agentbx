#!/usr/bin/env python3
"""
Test script to verify geometry minimizer consumer group setup.
"""

import argparse
import asyncio
import logging

# Add src to path for imports
import sys

from agentbx.core.agents.async_geometry_agent import AsyncGeometryAgent
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


def parse_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Test Geometry Minimizer Consumer")
    parser.add_argument(
        "--pdbfile",
        type=str,
        default="../data/small.pdb",
        help="Path to the input PDB file (default: small.pdb)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="Optimizer to use (adam or sgd, default: adam)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for optimizer (default: 0.1)",
    )
    parser.add_argument(
        "--request-stream-name",
        type=str,
        default="geometry_requests",
        help="Redis request stream name (default: geometry_requests)",
    )
    parser.add_argument(
        "--response-stream-name",
        type=str,
        default="geometry_requests_responses",
        help="Redis response stream name (default: geometry_requests_responses)",
    )
    parser.add_argument(
        "--consumer-group",
        type=str,
        default="minimizer_consumer",
        help="Redis consumer group (default: minimizer_consumer)",
    )
    parser.add_argument(
        "--no-agent",
        action="store_true",
        help="Do NOT start an AsyncGeometryAgent in the background",
    )
    return parser.parse_args()


async def test_minimizer_consumer(
    pdb_file,
    optimizer,
    learning_rate,
    request_stream_name,
    response_stream_name,
    consumer_group,
    start_agent,
):
    """Test the geometry minimizer consumer group setup."""
    print("=== Test Geometry Minimizer Consumer ===")

    # Initialize Redis manager
    print("1. Initializing Redis manager...")
    redis_manager = RedisManager(host="localhost", port=6379, db=0)

    if not redis_manager.is_healthy():
        print("❌ Redis connection failed. Please ensure Redis is running.")
        return False

    print("✅ Redis connection established")

    agent = None
    if start_agent:
        print("Starting AsyncGeometryAgent in the background...")
        agent = AsyncGeometryAgent(
            agent_id="test_agent",
            redis_manager=redis_manager,
            stream_name=request_stream_name,
            consumer_group="geometry_agents",
        )
        await agent.initialize()
        asyncio.create_task(agent.start())
        await asyncio.sleep(2)  # Give agent time to start
        print("✅ AsyncGeometryAgent started")

    # Initialize processor and create test bundle
    print("2. Creating test macromolecule bundle...")
    processor = MacromoleculeProcessor(redis_manager, "test_processor")

    # Test file
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

    # Select optimizer factory
    if optimizer == "adam":
        optimizer_factory = __import__("torch").optim.Adam
    elif optimizer == "sgd":
        optimizer_factory = __import__("torch").optim.SGD
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    optimizer_kwargs = {"lr": learning_rate}

    # Initialize geometry minimizer
    print("3. Initializing geometry minimizer...")
    minimizer = GeometryMinimizer(
        redis_manager=redis_manager,
        macromolecule_bundle_id=shaken_bundle_id,
        optimizer_factory=optimizer_factory,
        optimizer_kwargs=optimizer_kwargs,
        max_iterations=1,  # Just one iteration for testing
        convergence_threshold=1e-6,
        timeout_seconds=30.0,
        request_stream_name=request_stream_name,
        response_stream_name=response_stream_name,
        consumer_group=consumer_group,
    )

    print("   ✅ Geometry minimizer initialized")

    # Test the forward pass (geometry calculation)
    print("4. Testing forward pass (geometry calculation)...")
    try:
        gradients_tensor, geometry_bundle_id = await minimizer.forward()
        print("   ✅ Forward pass successful")
        print(f"   ✅ Geometry bundle ID: {geometry_bundle_id}")
        print(f"   ✅ Gradients tensor shape: {gradients_tensor.shape}")

        # Verify bundle exists
        geometry_bundle = redis_manager.get_bundle(geometry_bundle_id)
        print("   ✅ Geometry bundle retrieved successfully")

        # Check bundle contents
        assets = list(geometry_bundle.assets.keys())
        print(f"   ✅ Geometry bundle assets: {assets}")

        return True

    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        if agent is not None:
            print("Stopping AsyncGeometryAgent...")
            await agent.stop()
            print("✅ AsyncGeometryAgent stopped")


async def main():
    """Run the test."""
    setup_logging()
    args = parse_args()
    success = await test_minimizer_consumer(
        args.pdbfile,
        args.optimizer,
        args.learning_rate,
        args.request_stream_name,
        args.response_stream_name,
        args.consumer_group,
        not args.no_agent,
    )

    if success:
        print("\n🎉 Test passed!")
        sys.exit(0)
    else:
        print("\n💥 Test failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
