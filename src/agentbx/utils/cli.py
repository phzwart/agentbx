"""
Command-line interface for agentbx utilities.
"""

import logging
from typing import Optional

import click

from ..core.redis_manager import RedisManager
from .crystallographic_utils import validate_crystallographic_files
from .data_analysis_utils import analyze_bundle
from .data_analysis_utils import print_analysis_summary


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
def cli() -> None:
    """Agentbx utilities command-line interface."""


@cli.command()
@click.argument("pdb_file")
@click.argument("mtz_file", required=False)
def validate(pdb_file: str, mtz_file: Optional[str]) -> None:
    """Validate crystallographic files."""
    is_valid, results = validate_crystallographic_files(pdb_file, mtz_file)

    if is_valid:
        click.echo("✅ Files are valid")

        if results["pdb_file"]:
            pdb_info = results["pdb_file"]
            click.echo(f"PDB: {pdb_info.get('n_atoms', 'N/A')} atoms")

        if results["mtz_file"]:
            mtz_info = results["mtz_file"]
            click.echo(f"MTZ: {mtz_info.get('n_reflections', 'N/A')} reflections")

        if results["compatibility"]:
            comp = results["compatibility"]
            click.echo(
                f"Compatibility: {comp['pdb_atoms']} atoms, {comp['mtz_reflections']} reflections"
            )
    else:
        click.echo("❌ Files are invalid:")
        for error in results["errors"]:
            click.echo(f"  - {error}")


@cli.command()
@click.argument("bundle_id")
@click.option("--host", default="localhost", help="Redis host")
@click.option("--port", default=6379, help="Redis port")
def analyze(bundle_id: str, host: str, port: int) -> None:
    """Analyze a bundle in Redis."""
    try:
        redis_manager = RedisManager(host=host, port=port)
        bundle = redis_manager.get_bundle(bundle_id)

        analysis = analyze_bundle(bundle)
        print_analysis_summary(analysis)

    except Exception as e:
        click.echo(f"❌ Error: {e}")


@cli.command()
@click.argument("pdb_file")
@click.argument("mtz_file", required=False)
@click.option("--host", default="localhost", help="Redis host")
@click.option("--port", default=6379, help="Redis port")
def workflow(pdb_file: str, mtz_file: Optional[str], host: str, port: int) -> None:
    """Execute structure factor calculation workflow."""
    try:
        from .workflow_utils import execute_structure_factor_workflow

        redis_manager = RedisManager(host=host, port=port)
        output_ids = execute_structure_factor_workflow(
            redis_manager, pdb_file, mtz_file
        )

        click.echo("✅ Workflow completed successfully")
        click.echo(f"Output bundle IDs: {output_ids}")

    except Exception as e:
        click.echo(f"❌ Workflow failed: {e}")


if __name__ == "__main__":
    cli()
