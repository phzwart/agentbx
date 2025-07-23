#!/usr/bin/env python3
"""
Debug script for async geometry agent.

This script tests the async geometry agent directly to debug the bundle storage issue.
"""

import asyncio
import json
import logging

# Add src to path for imports
import sys
import time
import uuid

from agentbx.core.agents.async_geometry_agent import AsyncGeometryAgent
from agentbx.core.agents.async_geometry_agent import GeometryRequest
from agentbx.core.processors.macromolecule_processor import MacromoleculeProcessor
from agentbx.core.redis_manager import RedisManager
from agentbx.utils.structures.coordinate_shaker import shake_coordinates_in_bundle


def setup_logging():
    """Setup logging for the debug."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


async def debug_geometry_agent():
    """Debug the async geometry agent directly."""
    print("=== Debug Geometry Agent ===")

    # Initialize Redis manager
    print("1. Initializing Redis manager...")
    redis_manager = RedisManager(host="localhost", port=6379, db=0)

    if not redis_manager.is_healthy():
        print("❌ Redis connection failed. Please ensure Redis is running.")
        return False

    print("✅ Redis connection established")

    # Initialize processor and create test bundle
    print("2. Creating test macromolecule bundle...")
    processor = MacromoleculeProcessor(redis_manager, "debug_processor")

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

    # Initialize async geometry agent
    print("3. Initializing async geometry agent...")
    agent = AsyncGeometryAgent(
        agent_id="debug_agent",
        redis_manager=redis_manager,
        stream_name="geometry_requests",
        consumer_group="geometry_agents",
    )

    try:
        await agent.initialize()
        print("   ✅ Agent initialized")
    except Exception as e:
        print(f"   ❌ Agent initialization failed: {e}")
        return False

    # Test direct geometry calculation
    print("4. Testing direct geometry calculation...")
    try:
        # Get the shaken bundle
        shaken_bundle = redis_manager.get_bundle(shaken_bundle_id)

        # Process with geometry processor directly
        from agentbx.core.processors.geometry_processor import CctbxGeometryProcessor

        geometry_processor = CctbxGeometryProcessor(
            redis_manager, "debug_geo_processor"
        )

        output_bundles = geometry_processor.process_bundles(
            {"macromolecule_data": shaken_bundle}
        )

        geometry_bundle = output_bundles["geometry_gradient_data"]
        geometry_bundle_id = redis_manager.store_bundle(geometry_bundle)

        print(f"   ✅ Direct geometry calculation successful: {geometry_bundle_id}")

        # Verify bundle exists
        retrieved_bundle = redis_manager.get_bundle(geometry_bundle_id)
        print(f"   ✅ Bundle retrieval successful: {type(retrieved_bundle)}")

        # Check bundle contents
        assets = list(retrieved_bundle.assets.keys())
        print(f"   ✅ Bundle assets: {assets}")

    except Exception as e:
        print(f"   ❌ Direct geometry calculation failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test async request/response
    print("5. Testing async request/response...")
    try:
        # Start the agent
        asyncio.create_task(agent.start())

        # Give agent time to start
        await asyncio.sleep(2)

        # Create request
        request = GeometryRequest(
            request_id=str(uuid.uuid4()),
            macromolecule_bundle_id=shaken_bundle_id,
            priority=1,
        )

        print(f"   Sending request: {request.request_id}")

        # Send request to stream
        stream_name = "geometry_requests"
        message = {
            "request": json.dumps(request.__dict__, default=str),
            "timestamp": time.time(),
            "source": "debug_script",
        }

        # Use sync Redis client for sending
        import redis

        sync_redis = redis.Redis(
            host="localhost", port=6379, db=0, decode_responses=True
        )
        sync_redis.xadd(stream_name, message)

        print("   ✅ Request sent to stream")

        # Wait for response
        response_stream = "geometry_requests_responses"
        consumer_group = "debug_consumer"
        consumer_name = "debug_1"

        # Create consumer group
        try:
            sync_redis.xgroup_create(response_stream, consumer_group, mkstream=True)
        except Exception:
            pass  # Group already exists

        # Wait for response
        start_time = time.time()
        timeout = 30
        while time.time() - start_time < timeout:
            messages = sync_redis.xreadgroup(
                consumer_group,
                consumer_name,
                {response_stream: ">"},
                count=1,
                block=1000,
            )

            if messages:
                for stream, message_list in messages:
                    for message_id, fields in message_list:
                        response_data = fields.get("response", "{}")
                        response_dict = json.loads(response_data)

                        print(f"   ✅ Received response: {response_dict}")

                        bundle_id = response_dict.get("geometry_bundle_id")
                        if bundle_id:
                            # Verify bundle exists
                            try:
                                retrieved_bundle = redis_manager.get_bundle(bundle_id)
                                print(f"   ✅ Response bundle exists: {bundle_id}")
                                print(f"   ✅ Bundle type: {type(retrieved_bundle)}")
                                return True
                            except Exception as e:
                                print(f"   ❌ Response bundle not found: {bundle_id}")
                                print(f"   ❌ Error: {e}")
                                return False

                        # Acknowledge message
                        sync_redis.xack(response_stream, consumer_group, message_id)

            await asyncio.sleep(0.1)

        print("   ❌ Timeout waiting for response")
        return False

    except Exception as e:
        print(f"   ❌ Async test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Stop agent
        await agent.stop()


async def main():
    """Run the debug test."""
    setup_logging()

    success = await debug_geometry_agent()

    if success:
        print("\n🎉 Debug test passed!")
        sys.exit(0)
    else:
        print("\n💥 Debug test failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
