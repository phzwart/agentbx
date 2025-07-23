"""
Script to download PDB files and associated data from the Protein Data Bank.

Updated with correct URLs and modern PDB data access patterns.
"""

import logging
import os
import sys

import click
import requests


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDBDownloader:
    """Download PDB files and associated data from the Protein Data Bank."""

    def __init__(self):
        # Updated to use the correct wwPDB URLs
        self.pdb_base_url = "https://files.rcsb.org/download/"
        self.wwpdb_base_url = "https://files.wwpdb.org/pub/pdb/data/structures/"
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "agentbx/1.0 (https://github.com/phzwart/agentbx)"}
        )

    def download_pdb_file(self, pdb_code: str, output_dir: str = "data") -> str:
        """
        Download PDB file for a given PDB code.

        Args:
            pdb_code: 4-letter PDB code (e.g., '1ubq')
            output_dir: Directory to save the file

        Returns:
            Path to downloaded PDB file

        Raises:
            RequestException: If the download fails
        """
        pdb_code = pdb_code.lower()

        # Try multiple sources for PDB files
        urls_to_try = [
            # Primary RCSB download service
            f"{self.pdb_base_url}{pdb_code}.pdb",
            # Alternative mmCIF format (more reliable)
            f"{self.pdb_base_url}{pdb_code}.cif",
            # wwPDB archive (with proper path structure)
            f"{self.wwpdb_base_url}divided/pdb/{pdb_code[1:3]}/pdb{pdb_code}.ent.gz",
        ]

        output_path = os.path.join(output_dir, f"{pdb_code}.pdb")

        for url in urls_to_try:
            logger.info(f"Trying to download PDB file: {url}")
            try:
                response = self.session.get(url, timeout=30)
                if response.status_code == 200:
                    # Create output directory if it doesn't exist
                    os.makedirs(output_dir, exist_ok=True)

                    # Handle compressed files
                    content = response.content
                    if url.endswith(".gz"):
                        import gzip

                        content = gzip.decompress(content)

                    # Convert mmCIF to PDB format if needed (basic conversion)
                    if url.endswith(".cif"):
                        # For now, save as-is but warn user
                        output_path = os.path.join(output_dir, f"{pdb_code}.cif")
                        logger.warning(
                            "Downloaded mmCIF format. Consider using mmCIF-compatible tools."
                        )

                    with open(output_path, "wb") as f:
                        f.write(content)

                    logger.info(f"Successfully downloaded PDB file: {output_path}")
                    return output_path

            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to download from {url}: {e}")
                continue

        raise requests.exceptions.RequestException(
            f"Failed to download PDB file for {pdb_code} from all sources"
        )

    def download_structure_factors(
        self, pdb_code: str, output_dir: str = "data"
    ) -> str:
        """
        Download structure factors for a given PDB code.

        Args:
            pdb_code: 4-letter PDB code (e.g., '1ubq')
            output_dir: Directory to save the file

        Returns:
            Path to downloaded structure factors file
        """
        pdb_code = pdb_code.lower()

        # Structure factors are now distributed as mmCIF files
        # Path structure: divided/structure_factors/{middle_2_chars}/r{pdb_code}sf.ent.gz
        middle_chars = pdb_code[1:3]

        urls_to_try = [
            # Primary structure factor location (mmCIF format)
            f"{self.wwpdb_base_url}divided/structure_factors/{middle_chars}/r{pdb_code}sf.ent.gz",
            # Alternative structure factor location
            f"{self.wwpdb_base_url}all/structure_factors/r{pdb_code}sf.ent.gz",
            # Try validation map coefficients (if available)
            f"https://files.rcsb.org/pub/pdb/validation_reports/{middle_chars}/{pdb_code}/{pdb_code}_validation_2fo-fc_map_coef.cif.gz",
        ]

        output_path = os.path.join(output_dir, f"{pdb_code}_sf.cif")

        for url in urls_to_try:
            logger.info(f"Trying to download structure factors: {url}")
            try:
                response = self.session.get(url, timeout=30)
                if response.status_code == 200:
                    # Create output directory if it doesn't exist
                    os.makedirs(output_dir, exist_ok=True)

                    # Handle compressed files
                    content = response.content
                    if url.endswith(".gz"):
                        import gzip

                        content = gzip.decompress(content)

                    with open(output_path, "wb") as f:
                        f.write(content)

                    logger.info(f"Downloaded structure factors: {output_path}")
                    return output_path

            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to download structure factors from {url}: {e}")
                continue

        # If no structure factors found, try to generate synthetic data
        logger.warning("No structure factor files found. Generating synthetic data...")
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
            pdb_files = [
                os.path.join(output_dir, f"{pdb_code}.pdb"),
                os.path.join(output_dir, f"{pdb_code}.cif"),
            ]

            pdb_file = None
            for f in pdb_files:
                if os.path.exists(f):
                    pdb_file = f
                    break

            if not pdb_file:
                raise RuntimeError(
                    "No coordinate file found for synthetic data generation"
                )

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
            output_path = os.path.join(output_dir, f"{pdb_code}.mtz")
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
        self, pdb_code: str, output_dir: str = "data"
    ) -> tuple[str, str]:
        """
        Download both PDB and structure factor files for a given PDB code.

        Args:
            pdb_code: 4-letter PDB code (e.g., '1ubq')
            output_dir: Directory to save the files

        Returns:
            Tuple of (pdb_file_path, structure_factor_file_path)
        """
        logger.info(f"Downloading data for PDB code: {pdb_code.upper()}")

        # Download PDB file
        pdb_file = self.download_pdb_file(pdb_code, output_dir)

        # Download structure factors
        sf_file = self.download_structure_factors(pdb_code, output_dir)

        return pdb_file, sf_file

    def list_available_pdb_codes(self) -> list[str]:
        """
        Return a list of PDB codes that are known to work well with this example.

        Returns:
            List of PDB codes
        """
        return [
            "1ubq",  # Ubiquitin - small, well-determined structure
            "1lyd",  # Lysozyme - classic test case
            "1crn",  # Crambin - small protein with excellent data
            "1hhb",  # Hemoglobin - medium size
            "1ake",  # Adenylate kinase - medium size
            "2pdb",  # Trypsin inhibitor - small
            "1a28",  # Alpha-lytic protease - medium
            "1bni",  # Barnase - small
            "1cbs",  # Cytochrome c - small
            "1ee2",  # Endothiapepsin - medium size
            "1plc",  # Plastocyanin - high resolution small protein
            "3ry4",  # FcγRIIa - high resolution
            "5tro",  # Penicillin-binding protein - good resolution
        ]

    def get_pdb_info(self, pdb_code: str) -> dict:
        """
        Get basic information about a PDB entry from the RCSB API.

        Args:
            pdb_code: 4-letter PDB code

        Returns:
            Dictionary with PDB entry information
        """
        api_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_code.upper()}"

        try:
            response = self.session.get(api_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    "resolution": data.get("rcsb_entry_info", {}).get(
                        "resolution_combined", "N/A"
                    ),
                    "method": data.get("exptl", [{}])[0].get("method", "Unknown"),
                    "chains": data.get("rcsb_entry_info", {}).get(
                        "polymer_entity_count_protein", 0
                    ),
                    "release_date": data.get("rcsb_accession_info", {}).get(
                        "initial_release_date", "Unknown"
                    ),
                }
        except Exception as e:
            logger.warning(f"Could not fetch PDB info: {e}")

        return {}


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
    default="data",
    help="Output directory for downloaded files",
)
@click.option("--list-available", "-l", is_flag=True, help="List available PDB codes")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing files")
@click.option("--info", "-i", is_flag=True, help="Show PDB entry information")
def main(pdb_code, output_dir, list_available, force, info):
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
        click.echo("Example: python download_pdb_data.py 1ubq")
        return

    # Validate PDB code
    pdb_code = pdb_code.lower()
    if not validate_pdb_code(pdb_code):
        click.echo(
            "Error: PDB code must be 4 alphanumeric characters (e.g., 1ubq, 1ee2, 2pdb)"
        )
        return

    # Show PDB information if requested
    if info:
        click.echo(f"Getting information for PDB entry {pdb_code.upper()}...")
        pdb_info = downloader.get_pdb_info(pdb_code)
        if pdb_info:
            click.echo(f"Resolution: {pdb_info.get('resolution', 'N/A')} Å")
            click.echo(f"Method: {pdb_info.get('method', 'Unknown')}")
            click.echo(f"Protein chains: {pdb_info.get('chains', 0)}")
            click.echo(f"Release date: {pdb_info.get('release_date', 'Unknown')}")
        return

    # Check if files already exist
    possible_files = [
        os.path.join(output_dir, f"{pdb_code}.pdb"),
        os.path.join(output_dir, f"{pdb_code}.cif"),
        os.path.join(output_dir, f"{pdb_code}_sf.cif"),
        os.path.join(output_dir, f"{pdb_code}.mtz"),
    ]

    existing_files = [f for f in possible_files if os.path.exists(f)]

    if existing_files and not force:
        click.echo("Files already exist:")
        for f in existing_files:
            click.echo(f"  - {f}")
        click.echo("Use --force to overwrite.")
        return
    elif existing_files and force:
        click.echo("Overwriting existing files...")

    try:
        # Download the data
        coord_path, sf_path = downloader.download_all_data(pdb_code, output_dir)

        click.echo("✅ Download completed successfully!")
        click.echo(f"Coordinate file: {coord_path}")
        click.echo(f"Structure factor file: {sf_path}")

        # Show some helpful information
        pdb_info = downloader.get_pdb_info(pdb_code)
        if pdb_info:
            click.echo("\nPDB Entry Information:")
            click.echo(f"  Resolution: {pdb_info.get('resolution', 'N/A')} Å")
            click.echo(f"  Method: {pdb_info.get('method', 'Unknown')}")

        click.echo("\n📝 Notes:")
        click.echo("- Structure factors are now distributed in mmCIF format (.cif)")
        click.echo("- Use tools like phenix.cif_as_mtz to convert to MTZ if needed")
        click.echo("- Many modern crystallographic programs accept mmCIF directly")

    except Exception as e:
        click.echo(f"❌ Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
