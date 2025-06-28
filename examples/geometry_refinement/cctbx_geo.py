#!/usr/bin/env python3
"""
Geometry Refinement Tool for CCTBX

This script processes PDB files and creates geometry restraints managers
for crystallographic refinement workflows.
"""

import argparse
import os
import sys

import iotbx.pdb
import mmtbx.model
from libtbx.utils import null_out  # Used to suppress verbose output during processing


def process_pdb_file(pdb_file_name, verbose=True):
    """
    Process a PDB file and create a geometry restraints manager.

    Args:
        pdb_file_name (str): Path to the PDB file
        verbose (bool): Whether to print progress messages

    Returns:
        tuple: (model, grm) - The model manager and geometry restraints manager

    Raises:
        FileNotFoundError: If the PDB file doesn't exist
    """
    # Validate input file
    if not os.path.exists(pdb_file_name):
        raise FileNotFoundError(f"PDB file not found: {pdb_file_name}")

    if verbose:
        print(f"Reading PDB file: {pdb_file_name}")

    # 1. Read the PDB file from disk:
    #    The iotbx.pdb.input class is used to parse the PDB file.
    #    It takes the file_name as an argument.
    pdb_inp = iotbx.pdb.input(file_name=pdb_file_name)

    if verbose:
        print("Creating model manager...")

    # 2. Create an mmtbx.model.manager object:
    #    The mmtbx.model.manager serves as a high-level container for model information,
    #    including scattering information and geometry restraints.
    #    The 'model_input' argument takes the pdb_inp object created in the previous step.
    #    'log=null_out()' is used here to suppress the often verbose output from the model manager's initialization.
    model = mmtbx.model.manager(model_input=pdb_inp, log=null_out())

    if verbose:
        print("Processing model and building restraints...")

    # 3. Process the model and build the geometry restraints manager:
    #    Calling the 'process()' method on the model manager is crucial.
    #    Setting 'make_restraints=True' instructs the model manager to automatically
    #    generate the necessary geometry restraints (e.g., bonds, angles, dihedrals)
    #    based on the model's chemical components and the monomer library.
    model.process(make_restraints=True)

    # 4. Access the geometry restraints manager (GRM):
    #    Once the model is processed with 'make_restraints=True', the geometry
    #    restraints manager can be retrieved from the model object.
    grm = model.get_restraints_manager().geometry

    if verbose:
        print("\n--- Success ---")
        print(f"Successfully processed PDB file '{pdb_file_name}'.")
        print(f"Geometry Restraints Manager (GRM) object created: {grm}")

    return model, grm


def print_grm_info(grm, verbose=True):
    """
    Print information about the geometry restraints manager.

    Args:
        grm: Geometry restraints manager object
        verbose (bool): Whether to print detailed information
    """
    if not verbose:
        return
    print("\n--- GRM Information ---")
    # List of (label, attribute) pairs to check
    proxy_info = [
        ("bond proxies", "bond_proxies"),
        ("angle proxies", "angle_proxies"),
        ("dihedral proxies", "dihedral_proxies"),
        ("chirality proxies", "chirality_proxies"),
        ("planarity proxies", "planarity_proxies"),
        ("parallelity proxies", "parallelity_proxies"),
        ("reference coordinate proxies", "reference_coordinate_proxies"),
        ("pair proxies (nonbonded)", "pair_proxies"),
    ]
    total_restraints = 0
    restraint_types = []
    for label, attr in proxy_info:
        proxy = getattr(grm, attr, None)
        if proxy is not None and not callable(proxy) and hasattr(proxy, "size"):
            count = proxy.size()
            print(f"Number of {label}: {count}")
            total_restraints += count
            restraint_types.append(f"{label}: {count}")
    # Ramachandran restraints (just presence)
    if (
        hasattr(grm, "ramachandran_manager")
        and getattr(grm, "ramachandran_manager") is not None
    ):
        print("Ramachandran restraints: Available")
    if total_restraints > 0:
        print(f"\nTotal restraints: {total_restraints}")
        print(f"Restraint types: {', '.join(restraint_types)}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Process PDB files and create geometry restraints managers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s your_model.pdb
  %(prog)s -q your_model.pdb
  %(prog)s --verbose your_model.pdb
        """,
    )

    parser.add_argument("pdb_file", help="Path to the PDB file to process")

    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress progress messages"
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed GRM information"
    )

    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")

    args = parser.parse_args()

    # Determine verbosity level
    verbose = not args.quiet

    try:
        # Process the PDB file
        model, grm = process_pdb_file(args.pdb_file, verbose=verbose)

        # Print additional information if requested
        if args.verbose:
            print_grm_info(grm, verbose=True)

        # Return success
        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error processing PDB file: {e}", file=sys.stderr)
        if verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
