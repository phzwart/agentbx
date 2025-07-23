import argparse
import asyncio
import logging
import os

import torch

from agentbx.core.agents.async_geometry_agent import AsyncGeometryAgent
from agentbx.core.clients.geometry_minimizer import GeometryMinimizer
from agentbx.core.config import RedisConfig
from agentbx.core.processors.macromolecule_processor import MacromoleculeProcessor
from agentbx.core.redis_manager import RedisManager
from agentbx.utils.structures.coordinate_shaker import shake_coordinates_in_bundle


# Configure logging for demonstration
logging.basicConfig(level=logging.INFO)

# Stream configuration - centralized configuration for all components
STREAM_CONFIG = {
    "request_stream_name": "geometry_requests",
    "response_stream_name": "geometry_requests_responses",
    "agent_consumer_group": "geometry_agents",
    "minimizer_consumer_group": "minimizer_consumer",
}

# 0. Open a Redis session
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


def parse_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Geometry minimization workflow")
    parser.add_argument(
        "--pdbfile", type=str, required=True, help="Path to the input PDB file"
    )
    parser.add_argument(
        "--shake-magnitude",
        type=float,
        default=0.5,
        help="Magnitude for coordinate shaking (default: 0.5)",
    )
    return parser.parse_args()


async def main(pdbfile, shake_magnitude):
    """Main async workflow."""
    # 1. Initialize the async geometry agent
    print("1. Initializing async geometry agent...")
    agent = AsyncGeometryAgent(
        agent_id="test_agent",
        redis_manager=redis_manager,
        stream_name=STREAM_CONFIG["request_stream_name"],
        consumer_group=STREAM_CONFIG["agent_consumer_group"],
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
        pdb_path = os.path.abspath(pdbfile)
        processor = MacromoleculeProcessor(redis_manager, "workflow_processor")
        macromolecule_bundle_id = processor.create_macromolecule_bundle(pdb_path)
        print(f"Loaded macromolecule bundle: {macromolecule_bundle_id}")

        # 3. Load and prepare macromolecule data
        macromolecule_bundle = redis_manager.get_bundle(macromolecule_bundle_id)
        shaken_bundle = shake_coordinates_in_bundle(
            macromolecule_bundle, magnitude=shake_magnitude
        )
        shaken_bundle_id = redis_manager.store_bundle(shaken_bundle)

        # Construct optimizer and scheduler first
        minimizer = GeometryMinimizer(
            redis_manager=redis_manager,
            macromolecule_bundle_id=shaken_bundle_id,
            optimizer_factory=torch.optim.Adam,
            optimizer_kwargs={"lr": 0.33},
            scheduler_factory=torch.optim.lr_scheduler.CosineAnnealingLR,
            scheduler_kwargs={"T_max": 1000, "eta_min": 0.0},
            max_iterations=100,
            convergence_threshold=1e-6,
            timeout_seconds=3.0,
            # Stream configuration - no more magic constants!
            request_stream_name=STREAM_CONFIG["request_stream_name"],
            response_stream_name=STREAM_CONFIG["response_stream_name"],
            consumer_group=STREAM_CONFIG["minimizer_consumer_group"],
            consumer_name=None,  # Auto-generated
        )

        # 5. Run the geometry minimization loop
        results = await minimizer.minimize(refresh_restraints=False)
        print("Minimization results:")
        print(results)

    # Stop the agent
    await agent.stop()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.pdbfile, args.shake_magnitude))
