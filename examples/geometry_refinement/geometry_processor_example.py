#!/usr/bin/env python3
"""
Example demonstrating the use of CctbxGeometryProcessor.

This example shows how to:
1. Read a PDB file
2. Create an atomic model bundle
3. Use the CctbxGeometryProcessor to compute geometry gradients
"""

import sys


# Add the src directory to the path so we can import agentbx modules


try:
    from agentbx.agents.geometry_processor import CctbxGeometryProcessor
    from agentbx.core.bundle_base import Bundle
    from agentbx.utils.io.crystallographic_utils import CrystallographicFileHandler
except ImportError:
    print(
        "Error: Could not import agentbx modules. Make sure you're in the correct directory."
    )
    sys.exit(1)


class MockRedisManager:
    """Simple mock Redis manager for testing without Redis."""

    def __init__(self):
        self.bundles = {}
        self.bundle_id_counter = 0

    def store_bundle(self, bundle):
        """Store a bundle and return its ID."""
        bundle_id = f"mock_bundle_{self.bundle_id_counter}"
        self.bundle_id_counter += 1
        self.bundles[bundle_id] = bundle
        return bundle_id

    def get_bundle(self, bundle_id):
        """Retrieve a bundle by ID."""
        return self.bundles.get(bundle_id)


def create_atomic_model_bundle(pdb_file: str) -> Bundle:
    """
    Create an atomic model bundle from a PDB file.

    Args:
        pdb_file: Path to the PDB file

    Returns:
        Bundle with atomic model data
    """
    handler = CrystallographicFileHandler()

    # Read PDB file
    xray_structure = handler.read_pdb_file(pdb_file)

    # Create synthetic miller indices (not used for geometry, but required by bundle)
    from cctbx import crystal
    from cctbx import miller

    # Create miller set with reasonable resolution
    miller_set = miller.build_set(
        crystal_symmetry=crystal.symmetry(
            unit_cell=xray_structure.unit_cell(),
            space_group=xray_structure.space_group(),
        ),
        anomalous_flag=False,
        d_min=2.0,
    )

    # Create bundle
    bundle = Bundle(bundle_type="xray_atomic_model_data")
    bundle.add_asset("xray_structure", xray_structure)
    bundle.add_asset("miller_indices", miller_set)
    bundle.add_asset("pdb_file", pdb_file)

    return bundle


def main():
    """Main function demonstrating geometry processor usage."""
    # Define the PDB file name
    pdb_file = "your_model.pdb"

    print(f"Reading PDB file: {pdb_file}")

    # Create atomic model bundle
    atomic_bundle = create_atomic_model_bundle(pdb_file)
    print(
        f"Created atomic model bundle with {len(atomic_bundle.get_asset('xray_structure').scatterers())} atoms"
    )

    # Create mock Redis manager
    mock_redis = MockRedisManager()

    # Create geometry processor
    geometry_processor = CctbxGeometryProcessor(
        mock_redis, "geometry_processor_example"
    )
    print("Created CctbxGeometryProcessor")

    # Process the bundle to compute geometry gradients
    print("Computing geometry gradients...")
    input_bundles = {"xray_atomic_model_data": atomic_bundle}
    output_bundles = geometry_processor.process_bundles(input_bundles)

    # Extract the geometry gradient bundle
    geometry_bundle = output_bundles["geometry_gradient_data"]

    # Display results
    print("\n--- Results ---")
    coordinates = geometry_bundle.get_asset("coordinates")
    gradients = geometry_bundle.get_asset("geometric_gradients")
    restraint_counts = geometry_bundle.get_asset("restraint_counts")
    geometry_metadata = geometry_bundle.get_asset("geometry_metadata")

    print(f"Coordinates shape: {coordinates.size()} atoms")
    print(f"Gradients shape: {gradients.size()} atoms")
    print(f"Restraint counts: {restraint_counts}")
    print(f"Gradient norm: {geometry_metadata['gradient_norm']:.6f}")
    print(f"Max gradient: {geometry_metadata['max_gradient']:.6f}")
    print(f"Mean gradient: {geometry_metadata['mean_gradient']:.6f}")
    print(f"Computation time: {geometry_metadata['computation_time']:.3f}s")

    print("\n--- Success ---")
    print("Geometry gradients computed successfully!")


if __name__ == "__main__":
    main()
