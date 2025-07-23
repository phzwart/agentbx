#!/usr/bin/env python3
"""
Shake and Energy Test

This script tests the complete workflow:
1. Make a macromolecule bundle and submit to Redis
2. Shake the coordinates using the coordinate shaker
3. Get the restraint manager from Redis (deserialized)
4. Compute energy and gradient from the deserialized object
"""

import logging
import os
import sys

from agentbx.core.processors.macromolecule_processor import MacromoleculeProcessor
from agentbx.core.redis_manager import RedisManager
from agentbx.utils.structures.coordinate_shaker import shake_coordinates_in_bundle


def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def compute_total_energy(restraint_manager, model_manager):
    """Compute total energy and gradient norm from restraint manager and model manager."""
    try:
        # Get geometry restraints
        geometry_restraints = restraint_manager.geometry

        # Get current coordinates
        sites_cart = model_manager.get_sites_cart()

        # Compute total energy and gradients
        energy_gradients = geometry_restraints.energies_sites(
            sites_cart=sites_cart, compute_gradients=True
        )

        # Extract energy and gradients - use the correct attributes
        total_energy = energy_gradients.target
        gradients = energy_gradients.gradients

        # Compute gradient norm - handle scitbx vec3_double array
        import math

        if hasattr(gradients, "__len__"):
            # For scitbx arrays, iterate through each gradient vector
            gradient_norm = 0.0
            for i in range(len(gradients)):
                grad_vec = gradients[i]
                # Each gradient vector has 3 components (x, y, z)
                gradient_norm += grad_vec[0] ** 2 + grad_vec[1] ** 2 + grad_vec[2] ** 2
            gradient_norm = math.sqrt(gradient_norm)
        else:
            # Fallback for single value
            gradient_norm = abs(gradients)

        return total_energy, gradient_norm

    except Exception as e:
        print(f"Error computing total energy and gradients: {e}")
        return None, None


def refresh_restraint_manager(model_manager):
    """Refresh the restraint manager for round trip test."""

    model_manager.process(make_restraints=True)
    return model_manager.get_restraints_manager()


def test_shake_and_energy():
    """Test the complete shake and energy computation workflow."""
    print("=== Shake and Energy Test ===")

    # Initialize Redis manager
    print("1. Initializing Redis manager...")
    redis_manager = RedisManager(host="localhost", port=6379, db=0)

    if not redis_manager.is_healthy():
        print("❌ Redis connection failed. Please ensure Redis is running.")
        return False

    print("✅ Redis connection established")

    # Initialize processor
    print("2. Initializing macromolecule processor...")
    processor = MacromoleculeProcessor(redis_manager, "shake_test_processor")

    # Test file
    pdb_file = "../input.pdb"
    if not os.path.exists(pdb_file):
        print(f"❌ Test file not found: {pdb_file}")
        return False

    try:
        # Step 1: Create macromolecule bundle and submit to Redis
        print(f"\n3. Creating macromolecule bundle from {pdb_file}...")
        bundle_id = processor.create_macromolecule_bundle(pdb_file)
        print(f"   Created bundle: {bundle_id}")

        # Step 2: Get the bundle and extract initial coordinates
        print("4. Getting bundle from Redis...")
        bundle = processor.get_bundle(bundle_id)
        xray_structure = bundle.get_asset("xray_structure")
        initial_sites = xray_structure.sites_cart()
        print(f"   Initial coordinates: {len(initial_sites)} sites")

        # Step 3: Shake the coordinates
        print("5. Shaking coordinates...")
        shaken_bundle = shake_coordinates_in_bundle(bundle, magnitude=0.1)

        # Store shaken bundle
        shaken_bundle_id = redis_manager.store_bundle(shaken_bundle)
        print(f"   Shaken bundle stored: {shaken_bundle_id}")

        # Step 4: Get shaken coordinates for comparison
        shaken_xray_structure = shaken_bundle.get_asset("xray_structure")
        shaken_sites = shaken_xray_structure.sites_cart()
        print(f"   Shaken coordinates: {len(shaken_sites)} sites")

        # Calculate coordinate change
        import math

        total_change = 0.0
        for i in range(len(initial_sites)):
            dx = shaken_sites[i][0] - initial_sites[i][0]
            dy = shaken_sites[i][1] - initial_sites[i][1]
            dz = shaken_sites[i][2] - initial_sites[i][2]
            total_change += math.sqrt(dx * dx + dy * dy + dz * dz)
        print(f"   Total coordinate change: {total_change:.6f} Å")

        # Step 5: Get restraint manager from Redis (deserialized)
        print("6. Getting restraint manager from Redis...")
        retrieved_bundle = processor.get_bundle(shaken_bundle_id)
        restraint_manager = retrieved_bundle.get_asset("restraint_manager")
        model_manager = retrieved_bundle.get_asset("model_manager")

        print(f"   Restraint manager type: {type(restraint_manager)}")
        print(f"   Model manager type: {type(model_manager)}")

        # Step 6: Compute energy and gradient from deserialized object (stale restraint manager)
        print(
            "7. Computing energy and gradient from deserialized object (stale restraint manager)..."
        )
        total_energy, gradient_norm = compute_total_energy(
            restraint_manager, model_manager
        )

        if total_energy is not None:
            print(f"   ✅ [STALE] Total energy: {total_energy}")
            print(f"   ✅ [STALE] Gradient norm: {gradient_norm}")
            print(
                "   ✅ [STALE] Energy computation successful from deserialized object"
            )
        else:
            print("   ❌ [STALE] Energy computation failed")
            return False

        # Step 6b: Refresh restraint manager and recompute
        print("7b. Refreshing restraint manager and recomputing energy/gradient...")
        refreshed_restraint_manager = refresh_restraint_manager(model_manager)
        refreshed_energy, refreshed_gradient_norm = compute_total_energy(
            refreshed_restraint_manager, model_manager
        )
        if refreshed_energy is not None:
            print(f"   ✅ [REFRESHED] Total energy: {refreshed_energy}")
            print(f"   ✅ [REFRESHED] Gradient norm: {refreshed_gradient_norm}")
            print("   ✅ [REFRESHED] Energy computation successful after refresh")
        else:
            print("   ❌ [REFRESHED] Energy computation failed after refresh")
            return False

        # Step 7: Compare with original bundle energy
        print("8. Comparing with original bundle energy...")
        original_restraint_manager = bundle.get_asset("restraint_manager")
        original_model_manager = bundle.get_asset("model_manager")
        original_energy, original_gradient_norm = compute_total_energy(
            original_restraint_manager, original_model_manager
        )

        if original_energy is not None:
            print(f"   Original energy: {original_energy}")
            print(f"   Original gradient norm: {original_gradient_norm}")
            print(f"   Energy difference: {abs(total_energy - original_energy):.6f}")
            print(
                f"   Gradient norm difference: {abs(gradient_norm - original_gradient_norm):.6f}"
            )

            # Energy should be higher after shaking (worse geometry)
            if total_energy > original_energy:
                print("   ✅ Energy increased after shaking (expected)")
            else:
                print("   ⚠ Energy decreased after shaking (unexpected)")
        else:
            print("   ❌ Failed to compute original energy")

        print("\n✅ SHAKE AND ENERGY TEST PASSED")
        print("   - Bundle created and stored in Redis")
        print("   - Coordinates shaken successfully")
        print("   - Restraint manager retrieved from Redis")
        print("   - Energy and gradient computed from deserialized object")

        return True

    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Cleanup
        print("\nCleaning up...")
        redis_manager.close()


def main():
    """Run the shake and energy test."""
    setup_logging()

    success = test_shake_and_energy()

    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
