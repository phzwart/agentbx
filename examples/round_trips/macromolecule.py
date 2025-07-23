#!/usr/bin/env python3
"""
PDB File Round Trip Test

This script tests the complete round trip of PDB files through Redis:
1. Load PDB file and create macromolecule bundle
2. Store bundle in Redis
3. Retrieve bundle from Redis
4. Validate that all components are properly preserved
5. Test coordinate updates and re-serialization

This ensures that CCTBX objects are properly serialized/deserialized.
"""

import logging
import os
import sys

from agentbx.core.processors.macromolecule_processor import MacromoleculeProcessor
from agentbx.core.redis_manager import RedisManager


# Add src to path for imports
# Remove: sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


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
        # Try to debug what attributes are available
        try:
            energy_gradients = geometry_restraints.energies_sites(
                sites_cart=sites_cart, compute_gradients=True
            )
            print(f"Available attributes: {dir(energy_gradients)}")
            print(f"Gradients type: {type(energy_gradients.gradients)}")
            if hasattr(energy_gradients.gradients, "__len__"):
                grad_shape = len(energy_gradients.gradients)
            else:
                grad_shape = "no len"
            print(f"Gradients shape/length: {grad_shape}")
        except Exception as e:
            print(f"Error debugging energy gradients: {e}")
        return None, None


def refresh_restraint_manager(model_manager):
    """Refresh the restraint manager for round trip test."""
    model_manager.process(make_restraints=True)
    return model_manager.get_restraints_manager()


def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def debug_space_group_serialization(redis_manager, processor, pdb_file):
    """Debug space group serialization issues."""
    print(f"\n=== Debugging Space Group Serialization for {pdb_file} ===")

    # Create bundle
    bundle_id = processor.create_macromolecule_bundle(pdb_file)
    original_bundle = redis_manager.get_bundle(bundle_id)
    original_crystal_symmetry = original_bundle.get_asset("crystal_symmetry")

    # Get original space group details
    orig_space_group = original_crystal_symmetry.space_group()
    print(f"Original space group type: {type(orig_space_group)}")
    print(f"Original space group: {orig_space_group}")
    print(f"Original space group info: {orig_space_group.info()}")
    print(f"Original space group type: {orig_space_group.type()}")
    print(f"Original space group type number: {orig_space_group.type().number()}")

    # Test direct serialization
    print("\nTesting direct serialization...")
    try:
        serialized = redis_manager._serialize(orig_space_group)
        print(f"Serialized size: {len(serialized)} bytes")

        deserialized = redis_manager._deserialize(serialized)
        print(f"Deserialized type: {type(deserialized)}")
        print(f"Deserialized: {deserialized}")
        print(f"Deserialized info: {deserialized.info()}")
        print(f"Deserialized type: {deserialized.type()}")
        print(f"Deserialized type number: {deserialized.type().number()}")

        # Compare actual properties instead of string representation
        orig_info = str(orig_space_group.info())
        deser_info = str(deserialized.info())
        orig_type_num = orig_space_group.type().number()
        deser_type_num = deserialized.type().number()

        print(f"   Comparing space group info: '{orig_info}' vs '{deser_info}'")
        print(f"   Comparing type numbers: {orig_type_num} vs {deser_type_num}")

        if orig_info == deser_info and orig_type_num == deser_type_num:
            print("✅ Direct serialization works")
        else:
            print("❌ Direct serialization failed")
            print(f"Original info: {orig_info}")
            print(f"Deserialized info: {deser_info}")
            print(f"Original type number: {orig_type_num}")
            print(f"Deserialized type number: {deser_type_num}")

            # Debug: check if there are hidden characters
            print(f"Original info bytes: {repr(orig_info)}")
            print(f"Deserialized info bytes: {repr(deser_info)}")

    except Exception as e:
        print(f"❌ Direct serialization failed with exception: {e}")
        import traceback

        traceback.print_exc()

    # Test crystal symmetry serialization
    print("\nTesting crystal symmetry serialization...")
    try:
        serialized = redis_manager._serialize(original_crystal_symmetry)
        print(f"Serialized crystal symmetry size: {len(serialized)} bytes")

        deserialized = redis_manager._deserialize(serialized)
        print(f"Deserialized crystal symmetry type: {type(deserialized)}")

        # Get space group from deserialized crystal symmetry
        retr_space_group = deserialized.space_group()
        print(f"Retrieved space group: {retr_space_group}")
        print(f"Retrieved space group info: {retr_space_group.info()}")
        print(f"Retrieved space group type: {retr_space_group.type()}")
        print(f"Retrieved space group type number: {retr_space_group.type().number()}")

        # Compare actual properties instead of string representation
        orig_info = str(orig_space_group.info())
        retr_info = str(retr_space_group.info())
        orig_type_num = orig_space_group.type().number()
        retr_type_num = retr_space_group.type().number()

        print(f"   Comparing space group info: '{orig_info}' vs '{retr_info}'")
        print(f"   Comparing type numbers: {orig_type_num} vs {retr_type_num}")

        if orig_info == retr_info and orig_type_num == retr_type_num:
            print("✅ Crystal symmetry serialization works")
        else:
            print("❌ Crystal symmetry serialization failed")
            print(f"Original info: {orig_info}")
            print(f"Retrieved info: {retr_info}")
            print(f"Original type number: {orig_type_num}")
            print(f"Retrieved type number: {retr_type_num}")

            # Debug: check if there are hidden characters
            print(f"Original info bytes: {repr(orig_info)}")
            print(f"Retrieved info bytes: {repr(retr_info)}")

    except Exception as e:
        print(f"❌ Crystal symmetry serialization failed with exception: {e}")
        import traceback

        traceback.print_exc()


def test_basic_round_trip(redis_manager, processor, pdb_file):
    """Test basic round trip of PDB file through Redis."""
    print(f"\n=== Testing Basic Round Trip for {pdb_file} ===")

    # Step 1: Create macromolecule bundle
    print("1. Creating macromolecule bundle...")
    bundle_id = processor.create_macromolecule_bundle(pdb_file)
    print(f"   Created bundle: {bundle_id}")

    # Step 2: Get original bundle and extract key components
    print("2. Extracting original components...")
    original_bundle = redis_manager.get_bundle(bundle_id)

    # Extract key components for comparison
    original_hierarchy = original_bundle.get_asset("pdb_hierarchy")
    original_xray_structure = original_bundle.get_asset("xray_structure")
    original_model_manager = original_bundle.get_asset("model_manager")
    original_restraint_manager = refresh_restraint_manager(original_model_manager)
    original_crystal_symmetry = original_bundle.get_asset("crystal_symmetry")

    # Compute original total energy
    print("\n--- Original Bundle Energy Computation ---")
    original_total_energy, original_gradient_norm = compute_total_energy(
        original_restraint_manager, original_model_manager
    )
    if original_total_energy is not None:
        print(f"Original total energy: {original_total_energy}")
        print(f"Original gradient norm: {original_gradient_norm}")
    else:
        print("Failed to compute original total energy")
    print("--- End Original Bundle Energy Computation ---")

    # Get original metadata
    original_metadata = {
        "n_atoms": original_bundle.get_metadata("n_atoms"),
        "n_chains": original_bundle.get_metadata("n_chains"),
        "unit_cell": original_bundle.get_metadata("unit_cell"),
        "space_group": original_bundle.get_metadata("space_group"),
        "source_file": original_bundle.get_metadata("source_file"),
    }

    print(f"   Original atoms: {original_metadata['n_atoms']}")
    print(f"   Original chains: {original_metadata['n_chains']}")
    print(f"   Original unit cell: {original_metadata['unit_cell']}")
    print(f"   Original space group: {original_metadata['space_group']}")

    # Step 3: Store bundle in Redis (this happens automatically in create_macromolecule_bundle)
    print("3. Bundle already stored in Redis during creation")

    # Step 4: Retrieve bundle from Redis
    print("4. Retrieving bundle from Redis...")
    retrieved_bundle = redis_manager.get_bundle(bundle_id)

    # Step 5: Extract retrieved components
    print("5. Extracting retrieved components...")
    retrieved_hierarchy = retrieved_bundle.get_asset("pdb_hierarchy")
    retrieved_xray_structure = retrieved_bundle.get_asset("xray_structure")
    retrieved_model_manager = retrieved_bundle.get_asset("model_manager")
    retrieved_restraint_manager = refresh_restraint_manager(retrieved_model_manager)
    retrieved_crystal_symmetry = retrieved_bundle.get_asset("crystal_symmetry")

    # Print model_manager and restraint_manager info
    print("\n--- Model and Restraint Manager Extraction ---")
    print(f"Model manager: {retrieved_model_manager}")
    print(f"Model manager type: {type(retrieved_model_manager)}")
    if hasattr(retrieved_model_manager, "get_restraints_manager"):
        print(
            f"Model manager restraints: {retrieved_model_manager.get_restraints_manager()}"
        )
    # Always refresh the restraint manager after deserialization
    retrieved_restraint_manager = refresh_restraint_manager(retrieved_model_manager)
    print(f"Restraint manager: {retrieved_restraint_manager}")
    print(f"Restraint manager type: {type(retrieved_restraint_manager)}")
    if hasattr(retrieved_restraint_manager, "geometry"):
        print(f"Restraint manager geometry: {retrieved_restraint_manager.geometry}")

    # Compute total energy
    print("\n--- Computing Total Energy ---")
    total_energy, gradient_norm = compute_total_energy(
        retrieved_restraint_manager, retrieved_model_manager
    )
    if total_energy is not None:
        print(f"Total energy: {total_energy}")
        print(f"Gradient norm: {gradient_norm}")
        print(f"Total energy type: {type(total_energy)}")

        # Compare with original energy if available
        if original_total_energy is not None:
            energy_diff = abs(total_energy - original_total_energy)
            gradient_diff = abs(gradient_norm - original_gradient_norm)
            print(f"Energy difference: {energy_diff}")
            print(f"Gradient norm difference: {gradient_diff}")
            if energy_diff < 1e-6 and gradient_diff < 1e-6:
                print(
                    "✅ Energy and gradient computation preserved through serialization"
                )
            else:
                print("❌ Energy or gradient computation changed through serialization")
    else:
        print("Failed to compute total energy")
    print("--- End Total Energy Computation ---")

    print("--- End Model and Restraint Manager Extraction ---\n")

    # Get retrieved metadata
    retrieved_metadata = {
        "n_atoms": retrieved_bundle.get_metadata("n_atoms"),
        "n_chains": retrieved_bundle.get_metadata("n_chains"),
        "unit_cell": retrieved_bundle.get_metadata("unit_cell"),
        "space_group": retrieved_bundle.get_metadata("space_group"),
        "source_file": retrieved_bundle.get_metadata("source_file"),
    }

    print(f"   Retrieved atoms: {retrieved_metadata['n_atoms']}")
    print(f"   Retrieved chains: {retrieved_metadata['n_chains']}")
    print(f"   Retrieved unit cell: {retrieved_metadata['unit_cell']}")
    print(f"   Retrieved space group: {retrieved_metadata['space_group']}")

    # Step 6: Validate round trip
    print("6. Validating round trip...")

    # Check metadata preservation
    metadata_passed = True
    for key in original_metadata:
        if original_metadata[key] != retrieved_metadata[key]:
            print(
                f"   ❌ Metadata mismatch for {key}: {original_metadata[key]} != {retrieved_metadata[key]}"
            )
            metadata_passed = False
        else:
            print(f"   ✅ Metadata preserved for {key}")

    # Check hierarchy preservation
    hierarchy_passed = True
    try:
        original_atoms = list(original_hierarchy.atoms())
        retrieved_atoms = list(retrieved_hierarchy.atoms())

        if len(original_atoms) != len(retrieved_atoms):
            print(
                f"   ❌ Atom count mismatch: {len(original_atoms)} != {len(retrieved_atoms)}"
            )
            hierarchy_passed = False
        else:
            print(f"   ✅ Atom count preserved: {len(original_atoms)}")

        # Check first few atom coordinates
        for i in range(min(3, len(original_atoms))):
            orig_coord = original_atoms[i].xyz
            retr_coord = retrieved_atoms[i].xyz
            if (
                abs(orig_coord[0] - retr_coord[0]) > 1e-6
                or abs(orig_coord[1] - retr_coord[1]) > 1e-6
                or abs(orig_coord[2] - retr_coord[2]) > 1e-6
            ):
                print(
                    f"   ❌ Coordinate mismatch for atom {i}: "
                    f"{orig_coord} != "
                    f"{retr_coord}"
                )
                hierarchy_passed = False
            else:
                print(f"   ✅ Coordinates preserved for atom {i}")

    except Exception as e:
        print(f"   ❌ Hierarchy validation failed: {e}")
        hierarchy_passed = False

    # Check xray_structure preservation
    xray_passed = True
    try:
        orig_sites = original_xray_structure.sites_cart()
        retr_sites = retrieved_xray_structure.sites_cart()

        if len(orig_sites) != len(retr_sites):
            print(
                f"   ❌ X-ray structure site count mismatch: {len(orig_sites)} != {len(retr_sites)}"
            )
            xray_passed = False
        else:
            print(f"   ✅ X-ray structure site count preserved: {len(orig_sites)}")

        # Check first few coordinates
        for i in range(min(3, len(orig_sites))):
            orig_site = orig_sites[i]
            retr_site = retr_sites[i]
            if (
                abs(orig_site[0] - retr_site[0]) > 1e-6
                or abs(orig_site[1] - retr_site[1]) > 1e-6
                or abs(orig_site[2] - retr_site[2]) > 1e-6
            ):
                print(
                    f"   ❌ X-ray structure coordinate mismatch for site {i}: {orig_site} != {retr_site}"
                )
                xray_passed = False
            else:
                print(f"   ✅ X-ray structure coordinates preserved for site {i}")

    except Exception as e:
        print(f"   ❌ X-ray structure validation failed: {e}")
        xray_passed = False

    # Check crystal symmetry preservation with detailed debugging
    symmetry_passed = True
    try:
        orig_unit_cell = original_crystal_symmetry.unit_cell()
        retr_unit_cell = retrieved_crystal_symmetry.unit_cell()

        if str(orig_unit_cell) != str(retr_unit_cell):
            print(f"   ❌ Unit cell mismatch: {orig_unit_cell} != {retr_unit_cell}")
            symmetry_passed = False
        else:
            print(f"   ✅ Unit cell preserved: {orig_unit_cell}")

        orig_space_group = original_crystal_symmetry.space_group()
        retr_space_group = retrieved_crystal_symmetry.space_group()

        print(f"   Original space group type: {type(orig_space_group)}")
        print(f"   Retrieved space group type: {type(retr_space_group)}")
        print(f"   Original space group: {orig_space_group}")
        print(f"   Retrieved space group: {retr_space_group}")

        # Compare actual space group properties instead of string representation
        orig_info = str(orig_space_group.info())
        retr_info = str(retr_space_group.info())
        orig_type_num = orig_space_group.type().number()
        retr_type_num = retr_space_group.type().number()

        print(f"   Comparing space group info: '{orig_info}' vs '{retr_info}'")
        print(f"   Comparing type numbers: {orig_type_num} vs {retr_type_num}")

        if orig_info == retr_info and orig_type_num == retr_type_num:
            print(f"   ✅ Space group preserved: {orig_info}")
        else:
            print(f"   ❌ Space group mismatch: {orig_info} != {retr_info}")
            print(f"   Original info bytes: {repr(orig_info)}")
            print(f"   Retrieved info bytes: {repr(retr_info)}")
            symmetry_passed = False

    except Exception as e:
        print(f"   ❌ Crystal symmetry validation failed: {e}")
        import traceback

        traceback.print_exc()
        symmetry_passed = False

    # Overall result
    overall_passed = (
        metadata_passed and hierarchy_passed and xray_passed and symmetry_passed
    )

    if overall_passed:
        print("   ✅ BASIC ROUND TRIP PASSED")
    else:
        print("   ❌ BASIC ROUND TRIP FAILED")

    return overall_passed, bundle_id


def test_coordinate_update_round_trip(redis_manager, processor, bundle_id):
    """Test coordinate update round trip."""
    print("\n=== Testing Coordinate Update Round Trip ===")

    # Get original bundle
    original_bundle = redis_manager.get_bundle(bundle_id)
    original_xray_structure = original_bundle.get_asset("xray_structure")
    original_sites = original_xray_structure.sites_cart()

    print(f"1. Original coordinates: {len(original_sites)} sites")

    # Create modified coordinates (small perturbation)
    import random

    random.seed(42)  # For reproducible test

    modified_sites = original_sites.deep_copy()
    for i in range(min(10, len(modified_sites))):  # Modify first 10 sites
        modified_sites[i] = (
            modified_sites[i][0] + random.uniform(-0.1, 0.1),
            modified_sites[i][1] + random.uniform(-0.1, 0.1),
            modified_sites[i][2] + random.uniform(-0.1, 0.1),
        )

    print(f"2. Modified {min(10, len(modified_sites))} coordinates")

    # Update coordinates
    print("3. Updating coordinates in bundle...")
    updated_bundle_id = processor.update_coordinates(bundle_id, modified_sites)

    # Retrieve updated bundle
    print("4. Retrieving updated bundle...")
    updated_bundle = redis_manager.get_bundle(updated_bundle_id)
    updated_xray_structure = updated_bundle.get_asset("xray_structure")
    updated_sites = updated_xray_structure.sites_cart()

    # Validate coordinate update
    print("5. Validating coordinate update...")
    update_passed = True

    for i in range(min(10, len(updated_sites))):
        # orig_site = original_sites[i]
        mod_site = modified_sites[i]
        retr_site = updated_sites[i]

        # Check that retrieved matches modified (not original)
        if (
            abs(mod_site[0] - retr_site[0]) > 1e-6
            or abs(mod_site[1] - retr_site[1]) > 1e-6
            or abs(mod_site[2] - retr_site[2]) > 1e-6
        ):
            print(
                f"   ❌ Updated coordinate mismatch for site {i}: {mod_site} != {retr_site}"
            )
            update_passed = False
        else:
            print(f"   ✅ Updated coordinates preserved for site {i}")

    if update_passed:
        print("   ✅ COORDINATE UPDATE ROUND TRIP PASSED")
    else:
        print("   ❌ COORDINATE UPDATE ROUND TRIP FAILED")

    # Use numpy for coordinate comparison
    import numpy as np

    initial_np = np.array(original_sites)
    updated_np = np.array(updated_sites)
    change = np.linalg.norm(updated_np - initial_np)
    print(f"Coordinate change: {change:.6f} A")

    return update_passed


def test_multiple_round_trips(redis_manager, processor, pdb_file, num_round_trips=3):
    """Test multiple round trips to ensure stability."""
    print(f"\n=== Testing Multiple Round Trips ({num_round_trips} iterations) ===")

    bundle_ids = []

    for i in range(num_round_trips):
        print(f"\nRound trip {i+1}/{num_round_trips}:")

        # Create bundle
        bundle_id = processor.create_macromolecule_bundle(pdb_file)
        bundle_ids.append(bundle_id)

        # Retrieve bundle
        bundle = redis_manager.get_bundle(bundle_id)
        hierarchy = bundle.get_asset("pdb_hierarchy")
        n_atoms = len(list(hierarchy.atoms()))

        print(f"   Bundle {bundle_id}: {n_atoms} atoms")

        # Compute energy and gradient norm
        model_manager = bundle.get_asset("model_manager")
        restraint_manager = refresh_restraint_manager(model_manager)
        total_energy, gradient_norm = compute_total_energy(
            restraint_manager, model_manager
        )

        if total_energy is not None:
            print(f"   Total energy: {total_energy}")
            print(f"   Gradient norm: {gradient_norm}")
        else:
            print(f"   Failed to compute energy for round trip {i+1}")

        # Validate basic properties
        if n_atoms == 0:
            print(f"   ❌ Round trip {i+1} failed: no atoms found")
            return False

    print(f"   ✅ All {num_round_trips} round trips completed successfully")
    return True


def test_bundle_inspection(redis_manager, processor, bundle_id):
    """Test bundle inspection capabilities."""
    print("\n=== Testing Bundle Inspection ===")

    # Test bundle listing
    print("1. Listing bundles...")
    bundles = redis_manager.list_bundles("macromolecule_data")
    print(f"   Found {len(bundles)} macromolecule bundles")

    # Test bundle metadata
    print("2. Getting bundle metadata...")
    metadata = redis_manager.get_bundle_metadata(bundle_id)
    print(f"   Bundle type: {metadata.get('bundle_type', 'unknown')}")
    print(f"   Created at: {metadata.get('created_at', 'unknown')}")
    print(f"   Size: {metadata.get('size_bytes', 0)} bytes")

    # Test bundle inspection
    print("3. Inspecting bundle...")
    inspection = redis_manager.inspect_bundle(bundle_id)
    print(f"   Assets: {list(inspection.get('assets', {}).keys())}")
    print(f"   Metadata keys: {list(inspection.get('metadata', {}).keys())}")

    return True


def main():
    """Run all round trip tests."""
    setup_logging()

    print("PDB File Round Trip Test")
    print("=" * 50)

    # Initialize Redis manager
    print("Initializing Redis manager...")
    redis_manager = RedisManager(host="localhost", port=6379, db=0)

    if not redis_manager.is_healthy():
        print("❌ Redis connection failed. Please ensure Redis is running.")
        return False

    print("✅ Redis connection established")

    # Initialize processor
    print("Initializing macromolecule processor...")
    processor = MacromoleculeProcessor(redis_manager, "test_processor")

    # Test files
    test_files = [
        "../input.pdb",
        "../small.pdb",
    ]

    all_tests_passed = True

    for pdb_file in test_files:
        if not os.path.exists(pdb_file):
            print(f"❌ Test file not found: {pdb_file}")
            continue

        print(f"\n{'='*60}")
        print(f"Testing file: {pdb_file}")
        print(f"{'='*60}")

        try:
            # Debug space group serialization first
            debug_space_group_serialization(redis_manager, processor, pdb_file)

            # Test 1: Basic round trip
            basic_passed, bundle_id = test_basic_round_trip(
                redis_manager, processor, pdb_file
            )
            all_tests_passed = all_tests_passed and basic_passed

            if basic_passed:
                # Test 2: Coordinate update round trip
                coord_passed = test_coordinate_update_round_trip(
                    redis_manager, processor, bundle_id
                )
                all_tests_passed = all_tests_passed and coord_passed

                # Test 3: Multiple round trips
                multi_passed = test_multiple_round_trips(
                    redis_manager, processor, pdb_file
                )
                all_tests_passed = all_tests_passed and multi_passed

                # Test 4: Bundle inspection
                inspect_passed = test_bundle_inspection(
                    redis_manager, processor, bundle_id
                )
                all_tests_passed = all_tests_passed and inspect_passed

        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            import traceback

            traceback.print_exc()
            all_tests_passed = False

    # Final results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")

    if all_tests_passed:
        print("✅ ALL ROUND TRIP TESTS PASSED")
        print("   PDB files are properly serialized and deserialized in Redis")
    else:
        print("❌ SOME ROUND TRIP TESTS FAILED")
        print("   There are issues with PDB file serialization/deserialization")

    # Cleanup
    print("\nCleaning up...")
    redis_manager.close()

    return all_tests_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
