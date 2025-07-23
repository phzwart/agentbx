#!/usr/bin/env python3
"""
Test script to demonstrate the refresh functionality in the async geometry agent.
"""

import asyncio
import logging
import os

from agentbx.core.agents.async_geometry_agent import AsyncGeometryAgent
from agentbx.core.clients.geometry_minimizer import GeometryMinimizer
from agentbx.core.config import RedisConfig
from agentbx.core.processors.macromolecule_processor import MacromoleculeProcessor
from agentbx.core.redis_manager import RedisManager
from agentbx.utils.structures.coordinate_shaker import shake_coordinates_in_bundle


# Configure logging
logging.basicConfig(level=logging.INFO)


async def test_refresh_functionality():
    """Test the refresh functionality."""
    print("=== Testing Refresh Functionality ===")

    # Initialize Redis manager
    redis_config = RedisConfig.from_env()
    redis_manager = RedisManager(
        host=redis_config.host,
        port=redis_config.port,
        db=redis_config.db,
        password=redis_config.password,
        max_connections=redis_config.max_connections,
        socket_timeout=redis_config.socket_timeout,
        socket_connect_timeout=redis_config.socket_connect_timeout,
        retry_on_timeout=redis_config.retry_on_timeout,
        health_check_interval=redis_config.health_check_interval,
        default_ttl=redis_config.default_ttl,
    )

    # Initialize the async geometry agent
    print("1. Initializing async geometry agent...")
    agent = AsyncGeometryAgent(
        agent_id="refresh_test_agent",
        redis_manager=redis_manager,
        stream_name="geometry_requests",
        consumer_group="geometry_agents",
    )

    await agent.initialize()
    print("   ✅ Agent initialized")

    # Start the agent in the background
    asyncio.create_task(agent.start())
    print("   ✅ Agent started")

    # Give the agent a moment to start up
    await asyncio.sleep(2)

    with redis_manager:
        print("Connected to Redis.")

        # 2. Read in a PDB file and load as a macromolecule bundle
        pdb_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../input.pdb")
        )
        processor = MacromoleculeProcessor(redis_manager, "refresh_test_processor")
        macromolecule_bundle_id = processor.create_macromolecule_bundle(pdb_path)
        print(f"Loaded macromolecule bundle: {macromolecule_bundle_id}")

        # 3. Shake the coordinates
        macromolecule_bundle = redis_manager.get_bundle(macromolecule_bundle_id)
        shaken_bundle = shake_coordinates_in_bundle(
            macromolecule_bundle, magnitude=0.02
        )
        shaken_bundle_id = redis_manager.store_bundle(shaken_bundle)
        print(f"Shaken coordinates stored in bundle: {shaken_bundle_id}")

        # 4. Test without refresh (should use existing restraints)
        print("\n4. Testing WITHOUT refresh (should use existing restraints)...")
        minimizer_no_refresh = GeometryMinimizer(
            redis_manager=redis_manager,
            macromolecule_bundle_id=shaken_bundle_id,
            learning_rate=0.1,
            optimizer="adam",
            max_iterations=3,
            convergence_threshold=1e-6,
            timeout_seconds=30.0,
        )
        print("Geometry minimizer client initialized (no refresh).")

        results_no_refresh = await minimizer_no_refresh.minimize(
            refresh_restraints=False
        )
        print("Minimization results (no refresh):")
        print(f"  Final gradient norm: {results_no_refresh['final_gradient_norm']}")
        print(f"  Iterations: {results_no_refresh['iterations']}")
        print(f"  Converged: {results_no_refresh['converged']}")

        # 5. Test with refresh (should rebuild restraints)
        print("\n5. Testing WITH refresh (should rebuild restraints)...")
        minimizer_with_refresh = GeometryMinimizer(
            redis_manager=redis_manager,
            macromolecule_bundle_id=shaken_bundle_id,
            learning_rate=0.1,
            optimizer="adam",
            max_iterations=3,
            convergence_threshold=1e-6,
            timeout_seconds=30.0,
        )
        print("Geometry minimizer client initialized (with refresh).")

        results_with_refresh = await minimizer_with_refresh.minimize(
            refresh_restraints=True
        )
        print("Minimization results (with refresh):")
        print(f"  Final gradient norm: {results_with_refresh['final_gradient_norm']}")
        print(f"  Iterations: {results_with_refresh['iterations']}")
        print(f"  Converged: {results_with_refresh['converged']}")

        # 6. Compare results
        print("\n6. Comparing results...")
        no_refresh_norm = results_no_refresh["final_gradient_norm"]
        with_refresh_norm = results_with_refresh["final_gradient_norm"]

        print(f"  No refresh final gradient norm: {no_refresh_norm}")
        print(f"  With refresh final gradient norm: {with_refresh_norm}")

        if abs(no_refresh_norm - with_refresh_norm) < 1e-6:
            print("  ✅ Results are very similar (expected for small changes)")
        else:
            print("  ⚠ Results differ (refresh may have different behavior)")

    # Stop the agent
    await agent.stop()
    print("\n✅ Refresh functionality test completed!")


if __name__ == "__main__":

    asyncio.run(test_refresh_functionality())
