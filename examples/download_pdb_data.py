"""
Script to download PDB files and associated data from the Protein Data Bank.
"""

import logging
import os
import sys
from urllib.parse import urljoin

import click
import requests


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDBDownloader:
    """Download PDB files and associated data from the Protein Data Bank."""

    def __init__(self):
        self.pdb_base_url = "https://files.rcsb.org/download/"
        self.pdb_archive_url = "https://data.rcsb.org/pub/pdb/data/structures/all/pdb/"
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "agentbx/1.0 (https://github.com/phzwart/agentbx)"}
        )

    def download_pdb_file(self, pdb_code: str, output_dir: str = "examples") -> str:
        """
        Download PDB file for a given PDB code.

        Args:
            pdb_code: 4-letter PDB code (e.g., '1ubq')
            output_dir: Directory to save the file

        Returns:
            Path to downloaded PDB file

        Raises:
            requests.exceptions.RequestException: If the download fails
        """
        pdb_code = pdb_code.lower()
        pdb_file = f"{pdb_code}.pdb"
        url = urljoin(self.pdb_base_url, pdb_file)

        output_path = os.path.join(output_dir, "input.pdb")

        logger.info(f"Downloading PDB file: {url}")
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            with open(output_path, "wb") as f:
                f.write(response.content)

            logger.info(f"Downloaded PDB file: {output_path}")
            return output_path

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download PDB file: {e}")
            raise

    def download_mtz_file(self, pdb_code: str, output_dir: str = "examples") -> str:
        """
        Download MTZ file for a given PDB code from PDB archive.

        Args:
            pdb_code: 4-letter PDB code (e.g., '1ubq')
            output_dir: Directory to save the file

        Returns:
            Path to downloaded MTZ file
        """
        pdb_code = pdb_code.lower()

        # Try different possible MTZ file names
        possible_mtz_names = [
            f"{pdb_code}_phases.mtz",
            f"{pdb_code}_reflections.mtz",
            f"{pdb_code}_data.mtz",
            f"{pdb_code}_sf.mtz",
        ]

        output_path = os.path.join(output_dir, "input.mtz")

        for mtz_name in possible_mtz_names:
            url = urljoin(self.pdb_archive_url, f"{pdb_code}/{mtz_name}")

            logger.info(f"Trying to download MTZ file: {url}")
            try:
                response = self.session.get(url, timeout=30)
                if response.status_code == 200:
                    # Create output directory if it doesn't exist
                    os.makedirs(output_dir, exist_ok=True)

                    with open(output_path, "wb") as f:
                        f.write(response.content)

                    logger.info(f"Downloaded MTZ file: {output_path}")
                    return output_path

            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to download {mtz_name}: {e}")
                continue

        # If no MTZ file found, try to generate one from PDB
        logger.warning("No MTZ file found in PDB archive. Generating synthetic data...")
        return self._generate_synthetic_mtz(pdb_code, output_dir)

    def _generate_synthetic_mtz(self, pdb_code: str, output_dir: str) -> str:
        """
        Generate synthetic MTZ file from PDB structure.

        Args:
            pdb_code: 4-letter PDB code
            output_dir: Directory to save the file

        Returns:
            Path to generated MTZ file

        Raises:
            RuntimeError: If CCTBX is not available or synthetic data generation fails
        """
        try:
            # Import CCTBX modules for synthetic data generation
            import random  # nosec - Used for generating synthetic test data

            from cctbx import crystal
            from cctbx import miller
            from cctbx.array_family import flex
            from iotbx import pdb

            # Read the PDB file we just downloaded
            pdb_file = os.path.join(output_dir, "input.pdb")
            pdb_input = pdb.input(file_name=pdb_file)
            xray_structure = pdb_input.xray_structure_simple()

            # Generate synthetic miller indices
            unit_cell = xray_structure.unit_cell()
            space_group = xray_structure.space_group()

            # Create miller set with reasonable resolution
            miller_set = miller.build_set(
                crystal_symmetry=crystal.symmetry(
                    unit_cell=unit_cell, space_group=space_group
                ),
                anomalous_flag=False,
                d_min=2.0,  # 2 Angstrom resolution
            )

            # Generate synthetic F_obs data
            f_obs_data = flex.double(miller_set.size())
            for i in range(miller_set.size()):
                # Generate realistic structure factor amplitudes
                f_obs_data[i] = random.uniform(
                    0.1, 100.0
                )  # nosec - Synthetic test data

            # Create miller array
            f_obs = miller_set.array(data=f_obs_data)

            # Add some metadata
            f_obs.set_info(
                miller.array_info(source="synthetic", labels=["F_obs"], wavelength=1.0)
            )

            # Write MTZ file
            output_path = os.path.join(output_dir, "input.mtz")
            f_obs.as_mtz_dataset(column_root_label="F_obs").mtz_object().write(
                output_path
            )

            logger.info(f"Generated synthetic MTZ file: {output_path}")
            logger.info(f"Synthetic data: {f_obs.size()} reflections, d_min=2.0A")

            return output_path

        except ImportError as e:
            logger.error(f"CCTBX not available for synthetic data generation: {e}")
            raise RuntimeError("Cannot generate synthetic MTZ without CCTBX") from e
        except Exception as e:
            logger.error(f"Failed to generate synthetic MTZ: {e}")
            raise

    def download_all_data(
        self, pdb_code: str, output_dir: str = "examples"
    ) -> tuple[str, str]:
        """
        Download both PDB and MTZ files for a given PDB code.

        Args:
            pdb_code: 4-letter PDB code (e.g., '1ubq')
            output_dir: Directory to save the files

        Returns:
            Tuple of (pdb_file_path, mtz_file_path)
        """
        logger.info(f"Downloading data for PDB code: {pdb_code.upper()}")

        # Download PDB file
        pdb_file = self.download_pdb_file(pdb_code, output_dir)

        # Download or generate MTZ file
        mtz_file = self.download_mtz_file(pdb_code, output_dir)

        return pdb_file, mtz_file

    def list_available_pdb_codes(self) -> list[str]:
        """
        Return a list of PDB codes that are known to work well with this example.

        Returns:
            List of PDB codes
        """
        return [
            "1ubq",  # Ubiquitin - small, well-determined structure
            "1lyd",  # Lysozyme - classic test case
            "1crn",  # Crambin - small protein
            "1hhb",  # Hemoglobin - medium size
            "1ake",  # Adenylate kinase - medium size
            "1pdb",  # Trypsin inhibitor - small
            "1a28",  # Alpha-lytic protease - medium
            "1bni",  # Barnase - small
            "1cbs",  # Cytochrome c - small
            "1d66",  # DNA-binding protein - small
            "1ee2",  # Endothiapepsin - medium size
            "2pdb",  # Trypsin inhibitor - small
            "3pdb",  # Trypsin inhibitor - small
        ]


def validate_pdb_code(pdb_code: str) -> bool:
    """
    Validate PDB code format.

    PDB codes are 4 characters long and can contain letters and numbers.
    Examples: 1ubq, 1ee2, 2pdb, 3pdb, 1lyd, etc.
    """
    if len(pdb_code) != 4:
        return False

    # PDB codes can contain letters and numbers, but must be alphanumeric
    if not pdb_code.isalnum():
        return False

    return True


@click.command()
@click.argument("pdb_code", required=False)
@click.option(
    "--output-dir",
    "-o",
    default="examples",
    help="Output directory for downloaded files",
)
@click.option("--list-available", "-l", is_flag=True, help="List available PDB codes")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing files")
def main(pdb_code, output_dir, list_available, force):
    """
    Download PDB files and associated data from the Protein Data Bank.

    PDB_CODE: 4-character PDB code (e.g., 1ubq, 1ee2, 2pdb)
    """

    downloader = PDBDownloader()

    if list_available:
        click.echo("Available PDB codes for testing:")
        for code in downloader.list_available_pdb_codes():
            click.echo(f"  - {code.upper()}")
        return

    if not pdb_code:
        click.echo("Please provide a PDB code or use --list-available to see options")
        click.echo("Example: python examples/download_pdb_data.py 1ubq")
        return

    # Validate PDB code
    pdb_code = pdb_code.lower()
    if not validate_pdb_code(pdb_code):
        click.echo(
            "Error: PDB code must be 4 alphanumeric characters (e.g., 1ubq, 1ee2, 2pdb)"
        )
        return

    # Check if files already exist
    pdb_file = os.path.join(output_dir, "input.pdb")
    mtz_file = os.path.join(output_dir, "input.mtz")

    if os.path.exists(pdb_file) or os.path.exists(mtz_file):
        if not force:
            click.echo("Files already exist. Use --force to overwrite.")
            return
        else:
            click.echo("Overwriting existing files...")

    try:
        # Download the data
        pdb_path, mtz_path = downloader.download_all_data(pdb_code, output_dir)

        click.echo("✅ Download completed successfully!")
        click.echo(f"PDB file: {pdb_path}")
        click.echo(f"MTZ file: {mtz_path}")
        click.echo("\nYou can now run the Redis structure factor example:")
        click.echo("python examples/redis_structure_factor_example.py")

    except Exception as e:
        click.echo(f"❌ Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
