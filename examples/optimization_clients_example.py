#!/usr/bin/env python3
"""
Optimization Clients Example

This example demonstrates how to use the different optimization clients
for coordinate, B-factor, and solvent parameter optimization.
"""

import asyncio
import logging
import os
import time

from agentbx.core.agents.async_geometry_agent import AsyncGeometryAgent
from agentbx.core.clients.bfactor_optimizer import BFactorOptimizer
from agentbx.core.clients.coordinate_optimizer import CoordinateOptimizer
from agentbx.core.clients.solvent_optimizer import SolventOptimizer
from agentbx.core.redis_manager import RedisManager
from agentbx.processors.macromolecule_processor import MacromoleculeProcessor


# Add src to path for imports


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_macromolecule_bundle(
    redis_manager: RedisManager, pdb_file: str
) -> str:
    """Create a macromolecule bundle from a PDB file."""
    try:
        macromolecule_processor = MacromoleculeProcessor(
            redis_manager, "example_processor"
        )
        bundle_id = macromolecule_processor.create_macromolecule_bundle(pdb_file)
        logger.info(f"Created macromolecule bundle: {bundle_id}")
        return bundle_id
    except Exception as e:
        logger.error(f"Failed to create macromolecule bundle: {e}")
        raise


async def start_geometry_agent(redis_manager: RedisManager) -> AsyncGeometryAgent:
    """Start the async geometry agent."""
    try:
        agent = AsyncGeometryAgent(
            redis_manager=redis_manager,
            agent_id="example_agent",
            stream_name="geometry",
        )

        # Start the agent
        await agent.start()
        logger.info("Geometry agent started successfully")
        return agent

    except Exception as e:
        logger.error(f"Failed to start geometry agent: {e}")
        raise


async def demonstrate_coordinate_optimization(
    redis_manager: RedisManager, macromolecule_bundle_id: str
):
    """Demonstrate coordinate optimization."""
    logger.info("\n=== Coordinate Optimization ===")

    try:
        # Create coordinate optimizer
        coord_optimizer = CoordinateOptimizer(
            redis_manager=redis_manager,
            macromolecule_bundle_id=macromolecule_bundle_id,
            learning_rate=0.01,
            optimizer="adam",
            max_iterations=5,  # Small number for demo
            convergence_threshold=1e-4,
            timeout_seconds=30.0,
        )

        # Run optimization
        logger.info("Starting coordinate optimization...")
        start_time = time.time()
        results = await coord_optimizer.optimize()
        total_time = time.time() - start_time

        # Display results
        logger.info(f"Coordinate optimization completed in {total_time:.2f}s")
        logger.info(f"Converged: {results['converged']}")
        logger.info(f"Final gradient norm: {results['final_gradient_norm']:.6f}")
        logger.info(f"Best gradient norm: {results['best_gradient_norm']:.6f}")
        logger.info(f"Iterations: {results['iterations']}")

        # Get coordinate-specific stats
        coord_stats = coord_optimizer.get_coordinate_stats()
        logger.info(f"Number of atoms: {coord_stats['n_atoms']}")
        logger.info(f"Coordinate range: {coord_stats['coordinate_range']}")

        # Save optimized coordinates
        coord_optimizer.save_parameters("optimized_coordinates.pt")
        logger.info("Optimized coordinates saved to optimized_coordinates.pt")

        return results

    except Exception as e:
        logger.error(f"Coordinate optimization failed: {e}")
        raise


async def demonstrate_bfactor_optimization(
    redis_manager: RedisManager, macromolecule_bundle_id: str
):
    """Demonstrate B-factor optimization."""
    logger.info("\n=== B-Factor Optimization ===")

    try:
        # Create B-factor optimizer
        bfactor_optimizer = BFactorOptimizer(
            redis_manager=redis_manager,
            macromolecule_bundle_id=macromolecule_bundle_id,
            learning_rate=0.01,
            optimizer="adam",
            max_iterations=5,  # Small number for demo
            convergence_threshold=1e-4,
            timeout_seconds=30.0,
        )

        # Run optimization
        logger.info("Starting B-factor optimization...")
        start_time = time.time()
        results = await bfactor_optimizer.optimize()
        total_time = time.time() - start_time

        # Display results
        logger.info(f"B-factor optimization completed in {total_time:.2f}s")
        logger.info(f"Converged: {results['converged']}")
        logger.info(f"Final gradient norm: {results['final_gradient_norm']:.6f}")
        logger.info(f"Best gradient norm: {results['best_gradient_norm']:.6f}")
        logger.info(f"Iterations: {results['iterations']}")

        # Get B-factor-specific stats
        bfactor_stats = bfactor_optimizer.get_bfactor_stats()
        logger.info(f"Number of atoms: {bfactor_stats['n_atoms']}")
        logger.info(f"B-factor range: {bfactor_stats['b_factor_range']}")

        # Save optimized B-factors
        bfactor_optimizer.save_parameters("optimized_b_factors.pt")
        logger.info("Optimized B-factors saved to optimized_b_factors.pt")

        return results

    except Exception as e:
        logger.error(f"B-factor optimization failed: {e}")
        raise


async def demonstrate_solvent_optimization(
    redis_manager: RedisManager, macromolecule_bundle_id: str
):
    """Demonstrate solvent parameter optimization."""
    logger.info("\n=== Solvent Parameter Optimization ===")

    try:
        # Create solvent optimizer
        solvent_optimizer = SolventOptimizer(
            redis_manager=redis_manager,
            macromolecule_bundle_id=macromolecule_bundle_id,
            learning_rate=0.01,
            optimizer="adam",
            max_iterations=5,  # Small number for demo
            convergence_threshold=1e-4,
            timeout_seconds=30.0,
        )

        # Run optimization
        logger.info("Starting solvent parameter optimization...")
        start_time = time.time()
        results = await solvent_optimizer.optimize()
        total_time = time.time() - start_time

        # Display results
        logger.info(f"Solvent optimization completed in {total_time:.2f}s")
        logger.info(f"Converged: {results['converged']}")
        logger.info(f"Final gradient norm: {results['final_gradient_norm']:.6f}")
        logger.info(f"Best gradient norm: {results['best_gradient_norm']:.6f}")
        logger.info(f"Iterations: {results['iterations']}")

        # Get solvent-specific stats
        solvent_stats = solvent_optimizer.get_solvent_stats()
        logger.info(f"k_sol: {solvent_stats['k_sol']:.3f}")
        logger.info(f"b_sol: {solvent_stats['b_sol']:.3f}")
        logger.info(
            f"grid_resolution_factor: {solvent_stats['grid_resolution_factor']:.3f}"
        )
        logger.info(f"solvent_radius: {solvent_stats['solvent_radius']:.3f}")

        # Save optimized solvent parameters
        solvent_optimizer.save_parameters("optimized_solvent_params.pt")
        logger.info("Optimized solvent parameters saved to optimized_solvent_params.pt")

        return results

    except Exception as e:
        logger.error(f"Solvent optimization failed: {e}")
        raise


async def demonstrate_sequential_optimization(
    redis_manager: RedisManager, macromolecule_bundle_id: str
):
    """Demonstrate sequential optimization of different parameters."""
    logger.info("\n=== Sequential Parameter Optimization ===")

    try:
        # Step 1: Optimize coordinates
        logger.info("Step 1: Optimizing coordinates...")
        coord_results = await demonstrate_coordinate_optimization(
            redis_manager, macromolecule_bundle_id
        )

        # Step 2: Optimize B-factors
        logger.info("Step 2: Optimizing B-factors...")
        bfactor_results = await demonstrate_bfactor_optimization(
            redis_manager, macromolecule_bundle_id
        )

        # Step 3: Optimize solvent parameters
        logger.info("Step 3: Optimizing solvent parameters...")
        solvent_results = await demonstrate_solvent_optimization(
            redis_manager, macromolecule_bundle_id
        )

        # Summary
        logger.info("\n=== Sequential Optimization Summary ===")
        logger.info(
            f"Coordinate optimization: {coord_results['iterations']} iterations, "
            f"gradient_norm = {coord_results['final_gradient_norm']:.6f}"
        )
        logger.info(
            f"B-factor optimization: {bfactor_results['iterations']} iterations, "
            f"gradient_norm = {bfactor_results['final_gradient_norm']:.6f}"
        )
        logger.info(
            f"Solvent optimization: {solvent_results['iterations']} iterations, "
            f"gradient_norm = {solvent_results['final_gradient_norm']:.6f}"
        )

        total_time = (
            coord_results["total_time"]
            + bfactor_results["total_time"]
            + solvent_results["total_time"]
        )
        logger.info(f"Total optimization time: {total_time:.2f}s")

    except Exception as e:
        logger.error(f"Sequential optimization failed: {e}")
        raise


async def demonstrate_parameter_inspection(
    redis_manager: RedisManager, macromolecule_bundle_id: str
):
    """Demonstrate inspection of optimized parameters."""
    logger.info("\n=== Parameter Inspection ===")

    try:
        # Get the final macromolecule bundle
        final_bundle = redis_manager.get_bundle(macromolecule_bundle_id)

        # Inspect xray_structure
        xray_structure = final_bundle.get_asset("xray_structure")
        logger.info(
            f"Final xray_structure has {len(xray_structure.scatterers())} scatterers"
        )

        # Inspect coordinates
        sites_cart = xray_structure.sites_cart()
        logger.info(f"Coordinate range: {sites_cart.min_max_mean().as_tuple()}")

        # Inspect B-factors
        b_factors = xray_structure.extract_u_iso_or_u_equiv() * 8 * 3.14159**2
        logger.info(f"B-factor range: {b_factors.min_max_mean().as_tuple()}")

        # Inspect solvent parameters
        solvent_params = final_bundle.get_asset("solvent_parameters")
        if solvent_params:
            logger.info(f"Solvent parameters: {solvent_params}")
        else:
            logger.info("No solvent parameters found in bundle")

        # Bundle metadata
        logger.info(f"Bundle ID: {final_bundle.bundle_id}")
        logger.info(f"Bundle type: {final_bundle.bundle_type}")
        logger.info(f"Available assets: {list(final_bundle.assets.keys())}")

    except Exception as e:
        logger.error(f"Parameter inspection failed: {e}")
        raise


async def main():
    """Main function."""
    logger.info("Optimization Clients Example")

    # Check for input files
    pdb_file = "examples/input.pdb"
    if not os.path.exists(pdb_file):
        logger.error(f"PDB file not found: {pdb_file}")
        logger.info("Please run: python examples/download_pdb_data.py 1ubq")
        return

    # Initialize Redis manager
    try:
        redis_manager = RedisManager(host="localhost", port=6379)
        logger.info("Redis manager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Redis manager: {e}")
        return

    # Start geometry agent
    agent = await start_geometry_agent(redis_manager)

    try:
        # Create macromolecule bundle
        logger.info("Creating macromolecule bundle...")
        macromolecule_bundle_id = await create_macromolecule_bundle(
            redis_manager, pdb_file
        )

        # Demonstrate individual optimizations
        await demonstrate_coordinate_optimization(
            redis_manager, macromolecule_bundle_id
        )
        await demonstrate_bfactor_optimization(redis_manager, macromolecule_bundle_id)
        await demonstrate_solvent_optimization(redis_manager, macromolecule_bundle_id)

        # Demonstrate sequential optimization
        await demonstrate_sequential_optimization(
            redis_manager, macromolecule_bundle_id
        )

        # Demonstrate parameter inspection
        await demonstrate_parameter_inspection(redis_manager, macromolecule_bundle_id)

        logger.info("\n--- Success ---")
        logger.info("All optimization examples completed successfully!")

    except Exception as e:
        logger.error(f"Error during optimization examples: {e}")
        raise

    finally:
        # Stop the agent
        await agent.stop()
        logger.info("Geometry agent stopped")

        # Close Redis connection
        redis_manager.close()
        logger.info("Redis connection closed")


if __name__ == "__main__":
    asyncio.run(main())
