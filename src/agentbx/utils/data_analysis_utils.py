"""
Utilities for analyzing crystallographic data and structure factors.
"""

import logging
from typing import Any
from typing import Dict

import numpy as np


logger = logging.getLogger(__name__)


def analyze_complex_data(data: Any, name: str = "data") -> Dict[str, Any]:
    """
    Analyze complex data and return comprehensive statistics.

    Args:
        data: Complex data array (CCTBX flex array or numpy array)
        name: Name of the data for logging

    Returns:
        Dictionary with analysis results
    """
    try:
        # Convert to numpy for easier analysis
        if hasattr(data, "__iter__"):
            data_array = np.array(data)
        else:
            data_array = data

        # Calculate amplitudes (magnitude of complex numbers)
        amplitudes = np.abs(data_array)

        # Calculate phases (angle of complex numbers)
        phases = np.angle(data_array, deg=True)

        # Calculate real and imaginary parts
        real_parts = np.real(data_array)
        imag_parts = np.imag(data_array)

        # Basic statistics
        analysis = {
            "name": name,
            "n_reflections": len(data_array),
            "amplitudes": {
                "min": float(amplitudes.min()),
                "max": float(amplitudes.max()),
                "mean": float(amplitudes.mean()),
                "std": float(amplitudes.std()),
                "median": float(np.median(amplitudes)),
            },
            "phases": {
                "min": float(phases.min()),
                "max": float(phases.max()),
                "mean": float(phases.mean()),
                "std": float(phases.std()),
            },
            "real_parts": {
                "min": float(real_parts.min()),
                "max": float(real_parts.max()),
                "mean": float(real_parts.mean()),
                "std": float(real_parts.std()),
            },
            "imaginary_parts": {
                "min": float(imag_parts.min()),
                "max": float(imag_parts.max()),
                "mean": float(imag_parts.mean()),
                "std": float(imag_parts.std()),
            },
            "sample_values": [
                str(data_array[0]) if len(data_array) > 0 else "N/A",
                str(data_array[1]) if len(data_array) > 1 else "N/A",
                str(data_array[2]) if len(data_array) > 2 else "N/A",
            ],
        }

        return analysis

    except Exception as e:
        logger.warning(f"Could not analyze {name} data: {e}")
        return {"name": name, "error": str(e), "n_reflections": 0}


def analyze_miller_array(
    miller_array: Any, name: str = "miller_array"
) -> Dict[str, Any]:
    """
    Analyze a CCTBX miller array and return comprehensive statistics.

    Args:
        miller_array: CCTBX miller array
        name: Name of the array for logging

    Returns:
        Dictionary with analysis results
    """
    try:
        analysis = {
            "name": name,
            "size": miller_array.size(),
            "is_complex": miller_array.is_complex_array(),
            "is_real": miller_array.is_real_array(),
            "anomalous_flag": miller_array.anomalous_flag(),
            "info": {},
        }

        # Get array info
        if hasattr(miller_array, "info"):
            info = miller_array.info()
            analysis["info"] = {
                "labels": info.labels if hasattr(info, "labels") else [],
                "source": info.source if hasattr(info, "source") else "unknown",
                "wavelength": info.wavelength if hasattr(info, "wavelength") else None,
            }

        # Get resolution info
        if hasattr(miller_array, "d_min"):
            analysis["d_min"] = miller_array.d_min()
        if hasattr(miller_array, "d_max"):
            analysis["d_max"] = miller_array.d_max()

        # Get unit cell and space group info
        if hasattr(miller_array, "unit_cell"):
            analysis["unit_cell"] = str(miller_array.unit_cell())
        if hasattr(miller_array, "space_group"):
            analysis["space_group"] = str(miller_array.space_group())

        # Analyze data if available
        if hasattr(miller_array, "data"):
            data_analysis = analyze_complex_data(miller_array.data(), f"{name}_data")
            analysis["data_analysis"] = data_analysis

        return analysis

    except Exception as e:
        logger.warning(f"Could not analyze miller array {name}: {e}")
        return {"name": name, "error": str(e), "size": 0}


def analyze_bundle(bundle: Any) -> Dict[str, Any]:
    """
    Analyze a bundle and return comprehensive information about its contents.

    Args:
        bundle: Bundle object

    Returns:
        Dictionary with bundle analysis
    """
    try:
        analysis = {
            "bundle_type": bundle.bundle_type,
            "bundle_id": getattr(bundle, "bundle_id", None),
            "created_at": getattr(bundle, "created_at", None),
            "n_assets": len(bundle.assets),
            "assets": {},
            "metadata": dict(bundle.metadata) if hasattr(bundle, "metadata") else {},
            "size_estimate": (
                bundle.get_size_estimate()
                if hasattr(bundle, "get_size_estimate")
                else None
            ),
        }

        # Analyze each asset
        for asset_name, asset in bundle.assets.items():
            if hasattr(asset, "size") and hasattr(asset, "data"):
                # It's likely a miller array
                analysis["assets"][asset_name] = analyze_miller_array(asset, asset_name)
            elif hasattr(asset, "scatterers"):
                # It's likely an xray_structure
                analysis["assets"][asset_name] = {
                    "type": "xray_structure",
                    "n_atoms": len(asset.scatterers()),
                    "n_chains": (
                        len(asset.chains()) if hasattr(asset, "chains") else None
                    ),
                    "unit_cell": str(asset.unit_cell()),
                    "space_group": str(asset.space_group()),
                }
            elif isinstance(asset, dict):
                # It's a dictionary (e.g., bulk_solvent_params)
                analysis["assets"][asset_name] = {
                    "type": "dict",
                    "keys": list(asset.keys()),
                    "values": asset,
                }
            else:
                # Generic object
                analysis["assets"][asset_name] = {
                    "type": type(asset).__name__,
                    "str_repr": (
                        str(asset)[:100] + "..."
                        if len(str(asset)) > 100
                        else str(asset)
                    ),
                }

        return analysis

    except Exception as e:
        logger.warning(f"Could not analyze bundle: {e}")
        return {
            "error": str(e),
            "bundle_type": getattr(bundle, "bundle_type", "unknown"),
        }


def print_analysis_summary(analysis: Dict[str, Any], indent: int = 0) -> None:
    """
    Print a formatted summary of analysis results.

    Args:
        analysis: Analysis results dictionary
        indent: Indentation level for formatting
    """
    prefix = "  " * indent

    if "error" in analysis:
        logger.error(f"{prefix}❌ Error: {analysis['error']}")
        return

    if "name" in analysis:
        logger.info(f"{prefix}📊 {analysis['name']}:")

    if "n_reflections" in analysis:
        logger.info(f"{prefix}  Reflections: {analysis['n_reflections']}")

    if "amplitudes" in analysis:
        amp = analysis["amplitudes"]
        logger.info(f"{prefix}  Amplitude range: {amp['min']:.3f} to {amp['max']:.3f}")
        logger.info(f"{prefix}  Mean amplitude: {amp['mean']:.3f}")

    if "phases" in analysis:
        phase = analysis["phases"]
        logger.info(
            f"{prefix}  Phase range: {phase['min']:.1f}° to {phase['max']:.1f}°"
        )
        logger.info(f"{prefix}  Mean phase: {phase['mean']:.1f}°")

    if "sample_values" in analysis:
        samples = analysis["sample_values"]
        logger.info(f"{prefix}  Sample values: {', '.join(samples[:3])}")


def compare_structure_factors(
    f_calc: Any, f_obs: Any, name_prefix: str = ""
) -> Dict[str, Any]:
    """
    Compare calculated and observed structure factors.

    Args:
        f_calc: Calculated structure factors (miller array)
        f_obs: Observed structure factors (miller array)
        name_prefix: Prefix for naming in output

    Returns:
        Dictionary with comparison results
    """
    try:
        # Ensure both arrays have the same indices
        common_set = f_calc.common_set(f_obs)
        f_calc_common = f_calc.select(common_set)
        f_obs_common = f_obs.select(common_set)

        # Calculate amplitudes
        f_calc_amp = f_calc_common.amplitudes()
        f_obs_amp = f_obs_common.amplitudes()

        # Calculate R-factors
        r_factor = f_calc_amp.r1_factor(f_obs_amp)
        r_free = None

        # Try to calculate R_free if free flags are available
        if hasattr(f_obs, "free_r_flags"):
            free_flags = f_obs.free_r_flags()
            if free_flags is not None:
                f_calc_free = f_calc_common.select(free_flags)
                f_obs_free = f_obs_common.select(free_flags)
                if f_calc_free.size() > 0:
                    f_calc_free_amp = f_calc_free.amplitudes()
                    f_obs_free_amp = f_obs_free.amplitudes()
                    r_free = f_calc_free_amp.r1_factor(f_obs_free_amp)

        comparison = {
            "n_reflections": f_calc_common.size(),
            "r_factor": r_factor,
            "r_free": r_free,
            "f_calc_analysis": analyze_miller_array(
                f_calc_common, f"{name_prefix}f_calc"
            ),
            "f_obs_analysis": analyze_miller_array(f_obs_common, f"{name_prefix}f_obs"),
        }

        return comparison

    except Exception as e:
        logger.warning(f"Could not compare structure factors: {e}")
        return {"error": str(e), "n_reflections": 0}
