"""
Helper script to prepare input files for the Redis structure factor example.
"""

import os
import shutil

import click


@click.command()
@click.option("--pdb", "pdb_file", help="Path to your PDB file")
@click.option("--mtz", "mtz_file", help="Path to your MTZ file")
def prepare_files(pdb_file, mtz_file):
    """Copy PDB and MTZ files to the examples directory."""

    examples_dir = "examples"

    # Create examples directory if it doesn't exist
    if not os.path.exists(examples_dir):
        os.makedirs(examples_dir)
        click.echo(f"Created directory: {examples_dir}")

    # Copy PDB file
    if pdb_file and os.path.exists(pdb_file):
        target_pdb = os.path.join(examples_dir, "input.pdb")
        shutil.copy2(pdb_file, target_pdb)
        click.echo(f"Copied {pdb_file} to {target_pdb}")
    else:
        click.echo("No PDB file provided or file not found")

    # Copy MTZ file
    if mtz_file and os.path.exists(mtz_file):
        target_mtz = os.path.join(examples_dir, "input.mtz")
        shutil.copy2(mtz_file, target_mtz)
        click.echo(f"Copied {mtz_file} to {target_mtz}")
    else:
        click.echo("No MTZ file provided or file not found")

    # Check if files are ready
    pdb_ready = os.path.exists(os.path.join(examples_dir, "input.pdb"))
    mtz_ready = os.path.exists(os.path.join(examples_dir, "input.mtz"))

    if pdb_ready and mtz_ready:
        click.echo("✅ Input files are ready!")
        click.echo("You can now run: python examples/redis_structure_factor_example.py")
    else:
        click.echo("❌ Input files are not ready yet.")
        if not pdb_ready:
            click.echo("  - Missing: examples/input.pdb")
        if not mtz_ready:
            click.echo("  - Missing: examples/input.mtz")


if __name__ == "__main__":
    prepare_files()
