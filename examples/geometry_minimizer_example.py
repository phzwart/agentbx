#!/usr/bin/env python3
"""
Geometry Minimizer Example

This example demonstrates how to use the GeometryMinimizer to perform
coordinate optimization with proper bundle updates.
"""

import asyncio
import logging
import os
import sys
import time

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentbx.core.redis_manager import RedisManager
from agentbx.core.clients.geometry_minimizer import GeometryMinimizer
from agentbx.core.agents.async_geometry_agent import AsyncGeometryAgent
from agentbx.processors.macromolecule_processor import MacromoleculeProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_macromolecule_bundle(redis_manager: RedisManager, pdb_file: str) -> str:
    """Create a macromolecule bundle from a PDB file."""
    try:
        macromolecule_processor = MacromoleculeProcessor(redis_manager, "example_processor")
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
            stream_name="geometry"
        )
        
        # Start the agent
        await agent.start()
        logger.info("Geometry agent started successfully")
        return agent
        
    except Exception as e:
        logger.error(f"Failed to start geometry agent: {e}")
        raise


async def run_minimization_example():
    """Run the complete minimization example."""
    logger.info("Starting Geometry Minimizer Example")
    
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
        macromolecule_bundle_id = await create_macromolecule_bundle(redis_manager, pdb_file)
        
        # Inspect the initial bundle
        initial_bundle = redis_manager.get_bundle(macromolecule_bundle_id)
        xray_structure = initial_bundle.get_asset("xray_structure")
        initial_coordinates = xray_structure.sites_cart()
        logger.info(f"Initial bundle has {initial_coordinates.size()} coordinates")
        
        # Create geometry minimizer
        logger.info("Creating geometry minimizer...")
        minimizer = GeometryMinimizer(
            redis_manager=redis_manager,
            macromolecule_bundle_id=macromolecule_bundle_id,
            learning_rate=0.01,
            optimizer="adam",
            max_iterations=10,  # Small number for demo
            convergence_threshold=1e-4,
            timeout_seconds=30.0
        )
        
        # Run minimization
        logger.info("Starting minimization...")
        start_time = time.time()
        results = await minimizer.minimize()
        total_time = time.time() - start_time
        
        # Display results
        logger.info("\n--- Minimization Results ---")
        logger.info(f"Converged: {results['converged']}")
        logger.info(f"Final gradient norm: {results['final_gradient_norm']:.6f}")
        logger.info(f"Best gradient norm: {results['best_gradient_norm']:.6f}")
        logger.info(f"Iterations: {results['iterations']}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Final bundle ID: {results['final_bundle_id']}")
        
        # Verify coordinates were updated in bundle
        final_bundle = redis_manager.get_bundle(results['final_bundle_id'])
        final_xray_structure = final_bundle.get_asset("xray_structure")
        final_coordinates = final_xray_structure.sites_cart()
        
        logger.info(f"\n--- Bundle Verification ---")
        logger.info(f"Final bundle has {final_coordinates.size()} coordinates")
        logger.info(f"Bundle ID changed: {macromolecule_bundle_id} -> {results['final_bundle_id']}")
        
        # Compare initial and final coordinates
        import numpy as np
        initial_np = np.array(initial_coordinates)
        final_np = np.array(final_coordinates)
        coordinate_change = np.linalg.norm(final_np - initial_np)
        logger.info(f"Total coordinate change: {coordinate_change:.6f} A")
        
        # Get best coordinates from minimizer
        best_coordinates = minimizer.get_best_coordinates()
        logger.info(f"Best coordinates shape: {best_coordinates.shape}")
        
        # Save coordinates
        minimizer.save_coordinates("minimized_coordinates.pt")
        logger.info("Coordinates saved to minimized_coordinates.pt")
        
        # Get minimization statistics
        stats = minimizer.get_minimization_stats()
        logger.info(f"\n--- Minimization Stats ---")
        logger.info(f"Optimizer: {stats['optimizer']}")
        logger.info(f"Learning rate: {stats['learning_rate']}")
        logger.info(f"Converged: {stats['converged']}")
        
        # Show iteration history
        logger.info(f"\n--- Iteration History ---")
        for i, iteration in enumerate(results['iteration_history'][:5]):  # Show first 5
            logger.info(f"Iteration {iteration['iteration']}: gradient_norm = {iteration['gradient_norm']:.6f}")
        
        if len(results['iteration_history']) > 5:
            logger.info(f"... and {len(results['iteration_history']) - 5} more iterations")
        
        logger.info("\n--- Success ---")
        logger.info("Geometry minimization completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during minimization: {e}")
        raise
    
    finally:
        # Stop the agent
        await agent.stop()
        logger.info("Geometry agent stopped")
        
        # Close Redis connection
        redis_manager.close()
        logger.info("Redis connection closed")


async def demonstrate_coordinate_updates():
    """Demonstrate coordinate updates in bundles."""
    logger.info("\n=== Coordinate Update Demonstration ===")
    
    # Initialize Redis manager
    redis_manager = RedisManager(host="localhost", port=6379)
    
    try:
        # Create macromolecule processor
        macromolecule_processor = MacromoleculeProcessor(redis_manager, "demo_processor")
        
        # Create a test macromolecule bundle
        pdb_file = "examples/input.pdb"
        if os.path.exists(pdb_file):
            bundle_id = macromolecule_processor.create_macromolecule_bundle(pdb_file)
            
            # Get initial coordinates
            initial_bundle = redis_manager.get_bundle(bundle_id)
            xray_structure = initial_bundle.get_asset("xray_structure")
            initial_coordinates = xray_structure.sites_cart()
            
            logger.info(f"Initial coordinates: {initial_coordinates.size()} atoms")
            
            # Create a small perturbation
            from cctbx.array_family import flex
            import random
            
            perturbation = flex.vec3_double(initial_coordinates.size())
            for i in range(initial_coordinates.size()):
                perturbation[i] = (
                    random.uniform(-0.1, 0.1),
                    random.uniform(-0.1, 0.1),
                    random.uniform(-0.1, 0.1)
                )
            
            new_coordinates = initial_coordinates + perturbation
            
            # Update coordinates in bundle
            updated_bundle_id = macromolecule_processor.update_coordinates(bundle_id, new_coordinates)
            
            # Verify update
            updated_bundle = redis_manager.get_bundle(updated_bundle_id)
            updated_xray_structure = updated_bundle.get_asset("xray_structure")
            updated_coordinates = updated_xray_structure.sites_cart()
            
            logger.info(f"Updated coordinates: {updated_coordinates.size()} atoms")
            logger.info(f"Bundle ID changed: {bundle_id} -> {updated_bundle_id}")
            
            # Check that coordinates were actually updated
            import numpy as np
            initial_np = np.array(initial_coordinates)
            updated_np = np.array(updated_coordinates)
            change = np.linalg.norm(updated_np - initial_np)
            logger.info(f"Coordinate change: {change:.6f} A")
            
            if change > 0:
                logger.info("✓ Coordinates successfully updated in bundle")
            else:
                logger.warning("⚠ No coordinate change detected")
        
    except Exception as e:
        logger.error(f"Error in coordinate update demo: {e}")
    
    finally:
        redis_manager.close()


def main():
    """Main function."""
    logger.info("Geometry Minimizer Example")
    
    # Run the main example
    asyncio.run(run_minimization_example())
    
    # Run coordinate update demonstration
    asyncio.run(demonstrate_coordinate_updates())


if __name__ == "__main__":
    main() 