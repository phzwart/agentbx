"""
Command-line interface for agentbx utilities.
"""

import logging
from typing import Optional

import click

from agentbx.core.redis_manager import RedisManager

from .data_analysis_utils import analyze_bundle
from .data_analysis_utils import print_analysis_summary
from .io.crystallographic_utils import validate_crystallographic_files


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
def cli() -> None:
    """Agentbx utilities command-line interface."""


def main() -> None:
    """Main entry point for the CLI."""
    cli()


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


if __name__ == "__main__":
    main()
