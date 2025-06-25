"""
Script to inspect downloaded PDB and MTZ data.
"""

import os
import sys
import click
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def inspect_pdb_file(pdb_file: str):
    """Inspect PDB file and show basic information."""
    try:
        from iotbx import pdb
        
        pdb_input = pdb.input(file_name=pdb_file)
        xray_structure = pdb_input.xray_structure_simple()
        
        click.echo("📁 PDB File Information:")
        click.echo(f"  File: {pdb_file}")
        click.echo(f"  Unit cell: {xray_structure.unit_cell()}")
        click.echo(f"  Space group: {xray_structure.space_group()}")
        click.echo(f"  Number of atoms: {len(xray_structure.scatterers())}")
        click.echo(f"  Number of chains: {len(xray_structure.chains())}")
        
        # Show some atom information
        scatterers = xray_structure.scatterers()
        if scatterers:
            click.echo(f"  First atom: {scatterers[0].label}")
            click.echo(f"  Last atom: {scatterers[-1].label}")
        
        return xray_structure
        
    except ImportError:
        click.echo("❌ CCTBX not available - cannot inspect PDB file")
        return None
    except Exception as e:
        click.echo(f"❌ Error inspecting PDB file: {e}")
        return None


def inspect_mtz_file(mtz_file: str):
    """Inspect MTZ file and show basic information."""
    try:
        from iotbx import mtz
        
        mtz_object = mtz.object(file_name=mtz_file)
        miller_arrays = mtz_object.as_miller_arrays()
        
        click.echo("\n📊 MTZ File Information:")
        click.echo(f"  File: {mtz_file}")
        click.echo(f"  Number of miller arrays: {len(miller_arrays)}")
        
        for i, array in enumerate(miller_arrays):
            click.echo(f"  Array {i+1}:")
            click.echo(f"    Labels: {array.info().labels}")
            click.echo(f"    Type: {array.info().source}")
            click.echo(f"    Size: {array.size()} reflections")
            click.echo(f"    Data type: {array.data().__class__.__name__}")
            
            if hasattr(array, 'd_min'):
                click.echo(f"    Resolution: {array.d_min():.2f} Å")
            
            if array.is_complex_array():
                click.echo(f"    Complex array: Yes")
            elif array.is_real_array():
                click.echo(f"    Real array: Yes")
            
            # Show some statistics
            if hasattr(array, 'data'):
                data = array.data()
                if hasattr(data, 'min') and hasattr(data, 'max'):
                    click.echo(f"    Data range: {data.min():.3f} to {data.max():.3f}")
        
        return miller_arrays
        
    except ImportError:
        click.echo("❌ CCTBX not available - cannot inspect MTZ file")
        return None
    except Exception as e:
        click.echo(f"❌ Error inspecting MTZ file: {e}")
        return None


@click.command()
@click.option('--pdb-file', '-p', default='examples/input.pdb', help='Path to PDB file')
@click.option('--mtz-file', '-m', default='examples/input.mtz', help='Path to MTZ file')
def main(pdb_file, mtz_file):
    """
    Inspect downloaded PDB and MTZ data files.
    """
    
    click.echo("\n Inspecting PDB and MTZ Data Files\n")
    
    # Check if files exist
    if not os.path.exists(pdb_file):
        click.echo(f"❌ PDB file not found: {pdb_file}")
        click.echo("Download data first: python examples/download_pdb_data.py <pdb_code>")
        return
    
    if not os.path.exists(mtz_file):
        click.echo(f"❌ MTZ file not found: {mtz_file}")
        click.echo("Download data first: python examples/download_pdb_data.py <pdb_code>")
        return
    
    # Inspect PDB file
    xray_structure = inspect_pdb_file(pdb_file)
    
    # Inspect MTZ file
    miller_arrays = inspect_mtz_file(mtz_file)
    
    # Show compatibility information
    if xray_structure and miller_arrays:
        click.echo("\n✅ Data Compatibility:")
        click.echo("  Both PDB and MTZ files are readable")
        click.echo("  Ready for structure factor calculation")
        click.echo("\nYou can now run:")
        click.echo("  python examples/redis_structure_factor_example.py")


if __name__ == "__main__":
    main() 