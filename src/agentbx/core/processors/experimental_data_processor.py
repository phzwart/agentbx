# src/agentbx/processors/experimental_data_processor.py
"""
Processor responsible ONLY for processing experimental data.

Input: raw_experimental_data (MTZ, HKL files, etc.)
Output: experimental_data

Does NOT know about:
- Atomic models
- Structure factors
- Target functions
"""

import logging
from typing import Any
from typing import Dict
from typing import List

from agentbx.core.bundle_base import Bundle
from agentbx.schemas.generated import ExperimentalDataBundle

from .base import SinglePurposeProcessor


class ExperimentalDataProcessor(SinglePurposeProcessor):
    """
    Pure experimental data processing processor.

    Responsibility: Convert raw experimental files to clean experimental_data bundles.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

    def define_input_bundle_types(self) -> List[str]:
        """Define the input bundle types for this processor."""
        return ["raw_experimental_data"]

    def define_output_bundle_types(self) -> List[str]:
        """Define the output bundle types for this processor."""
        return ["experimental_data"]

    def process_bundles(self, input_bundles: Dict[str, Bundle]) -> Dict[str, Bundle]:
        """
        Process raw experimental data into clean experimental_data bundle.
        """
        raw_data = input_bundles["raw_experimental_data"]

        # Extract raw data information
        file_path = raw_data.get_asset("file_path")
        data_labels = raw_data.get_metadata("data_labels", {})
        data_type = raw_data.get_metadata("data_type", "amplitudes")  # or "intensities"

        # Process the reflection file
        f_obs, sigmas, r_free_flags, metadata = self._process_reflection_file(
            file_path, data_labels, data_type
        )

        # Convert intensities to amplitudes if needed
        if data_type == "intensities":
            f_obs, sigmas = self._convert_intensities_to_amplitudes(f_obs, sigmas)

        # Validate data quality
        self._validate_experimental_data(f_obs, sigmas, r_free_flags)

        # Create experimental data bundle
        exp_bundle = Bundle(bundle_type="experimental_data")
        exp_bundle.add_asset("f_obs", f_obs)
        exp_bundle.add_asset("miller_indices", f_obs.indices())

        if sigmas is not None:
            exp_bundle.add_asset("sigmas", sigmas)

        if r_free_flags is not None:
            exp_bundle.add_asset("r_free_flags", r_free_flags)

        # Add experimental metadata
        exp_bundle.add_asset("experimental_metadata", metadata)

        # Add target preferences based on data quality
        target_prefs = self._determine_target_preferences(f_obs, sigmas, metadata)
        exp_bundle.add_asset("target_preferences", target_prefs)

        # Validate with schema
        ExperimentalDataBundle(
            f_obs=f_obs,
            miller_indices=f_obs.indices(),
            sigmas=sigmas,
            r_free_flags=r_free_flags,
            experimental_metadata=metadata,
            target_preferences=target_prefs,
        )
        print("[Schema Validation] ExperimentalDataBundle validation successful.")
        return {"experimental_data": exp_bundle}

    def _process_reflection_file(
        self, file_path: str, data_labels: Dict[str, Any], data_type: str
    ) -> tuple[Any, Any, Any, Dict[str, Any]]:
        """
        Process MTZ/HKL file to extract F_obs, sigmas, R_free.
        """
        from iotbx import reflection_file_reader
        from iotbx.reflection_file_utils import reflection_file_server

        # Read reflection file
        reflection_file = reflection_file_reader.any_reflection_file(file_path)

        if reflection_file is None:
            raise ValueError(f"Could not read reflection file: {file_path}")

        # Create reflection file server
        server = reflection_file_server(
            crystal_symmetry=None,
            force_symmetry=True,
            reflection_files=[reflection_file],
            err=None,
        )

        # Extract F_obs (or I_obs)
        f_obs_label = data_labels.get("f_obs", data_labels.get("i_obs"))
        sigma_label = data_labels.get(
            "sigmas", data_labels.get("sigma_f", data_labels.get("sigma_i"))
        )
        r_free_label = data_labels.get("r_free_flags", "FreeR_flag")

        # Get Miller arrays
        f_obs = server.get_miller_array(f_obs_label)
        if f_obs is None:
            raise ValueError(f"Could not find data column: {f_obs_label}")

        # Get sigmas if available
        sigmas = None
        if sigma_label:
            sigmas = server.get_miller_array(sigma_label)

        # Get R_free flags if available
        r_free_flags = None
        if r_free_label:
            try:
                r_free_flags = server.get_miller_array(r_free_label)
                if r_free_flags is not None:
                    r_free_flags = r_free_flags.as_bool()
            except Exception:
                self.logger.warning(f"Could not read R_free flags from {r_free_label}")

        # Extract experimental metadata
        metadata = self._extract_metadata_from_file(reflection_file, file_path)

        return f_obs, sigmas, r_free_flags, metadata

    def _convert_intensities_to_amplitudes(
        self, i_obs: Any, sig_i: Any
    ) -> tuple[Any, Any]:
        """
        French-Wilson conversion of intensities to amplitudes.
        """
        from cctbx import french_wilson

        # Apply French-Wilson algorithm
        fw = french_wilson.french_wilson_scale(miller_array=i_obs, log=None)

        f_obs = fw.f_sq_as_f()
        sigmas = fw.sigmas()

        return f_obs, sigmas

    def _extract_metadata_from_file(
        self, reflection_file: Any, file_path: str
    ) -> Dict[str, Any]:
        """
        Extract metadata from reflection file.
        """
        metadata = {
            "file_path": file_path,
            "file_type": "mtz",  # Default assumption
            "space_group": str(reflection_file.space_group_info()),
            "unit_cell": str(reflection_file.unit_cell()),
            "wavelength": 1.0,  # Default
            "temperature": "unknown",
        }

        # Try to extract wavelength from file
        try:
            wavelength = reflection_file.wavelength()
            if wavelength is not None:
                metadata["wavelength"] = wavelength
        except Exception:
            pass

        # Try to extract temperature from file
        try:
            temperature = reflection_file.temperature()
            if temperature is not None:
                metadata["temperature"] = temperature
        except Exception:
            pass

        return metadata

    def _validate_experimental_data(
        self, f_obs: Any, sigmas: Any, r_free_flags: Any
    ) -> None:
        """
        Validate experimental data quality.
        """
        # Check data completeness
        if f_obs.size() == 0:
            raise ValueError("No reflections found in experimental data")

        # Check for negative amplitudes
        if (f_obs.data() < 0).count(True) > 0:
            raise ValueError("Found negative structure factor amplitudes")

        # Check sigma/F ratios if sigmas available
        if sigmas is not None:
            sigma_f_ratios = sigmas.data() / f_obs.data()
            if (sigma_f_ratios > 10).count(True) > f_obs.size() * 0.1:
                self.logger.warning("Many reflections have high sigma/F ratios")

        # Check R_free completeness if available
        if r_free_flags is not None:
            r_free_fraction = r_free_flags.data().count(True) / r_free_flags.size()
            if r_free_fraction < 0.01 or r_free_fraction > 0.2:
                self.logger.warning(
                    f"R_free fraction ({r_free_fraction:.3f}) outside normal range"
                )

    def _determine_target_preferences(
        self, f_obs: Any, sigmas: Any, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Determine optimal target function based on data quality.
        """
        preferences = {
            "default_target": "maximum_likelihood",
            "use_anomalous": False,
            "use_twinning": False,
        }

        # Check for anomalous data
        if hasattr(f_obs, "anomalous_flag") and f_obs.anomalous_flag():
            preferences["use_anomalous"] = True

        # Check data quality for target selection
        if sigmas is not None:
            sigma_f_ratios = sigmas.data() / f_obs.data()
            mean_sigma_f = sigma_f_ratios.mean()

            if mean_sigma_f > 0.5:
                # High noise - prefer least squares
                preferences["default_target"] = "least_squares"
            elif mean_sigma_f < 0.1:
                # Low noise - maximum likelihood is fine
                pass
            else:
                # Medium noise - maximum likelihood with care
                preferences["default_target"] = "maximum_likelihood"

        return preferences

    def _generate_r_free_flags(self, f_obs: Any, fraction: float = 0.05) -> Any:
        """
        Generate R_free flags if not present.
        """
        # Create random R_free flags
        import random

        from cctbx import miller

        random.seed(42)  # For reproducibility

        flags = miller.build_set(
            crystal_symmetry=f_obs.crystal_symmetry(),
            anomalous_flag=f_obs.anomalous_flag(),
            d_min=f_obs.d_min(),
            d_max=f_obs.d_max(),
        )

        # Set random fraction as R_free
        data = [random.random() < fraction for _ in range(flags.size())]
        flags = flags.array(data=data)

        return flags

    def process_mtz_file(
        self,
        mtz_file: str,
        f_obs_label: str = "FP",
        sigma_label: str = "SIGFP",
        r_free_label: str = "FreeR_flag",
    ) -> str:
        """
        Process MTZ file and return experimental_data bundle ID.
        """
        # Create raw experimental data bundle
        raw_bundle = Bundle(bundle_type="raw_experimental_data")
        raw_bundle.add_asset("file_path", mtz_file)
        raw_bundle.add_metadata("data_type", "amplitudes")
        raw_bundle.add_metadata(
            "data_labels",
            {
                "f_obs": f_obs_label,
                "sigmas": sigma_label,
                "r_free_flags": r_free_label,
            },
        )

        # Store raw bundle
        raw_bundle_id = self.store_bundle(raw_bundle)

        # Process
        output_ids = self.run({"raw_experimental_data": raw_bundle_id})
        return output_ids["experimental_data"]

    def process_intensity_file(
        self, hkl_file: str, i_obs_label: str = "I", sigma_label: str = "SIGI"
    ) -> str:
        """
        Process intensity file and return experimental_data bundle ID.
        """
        # Create raw experimental data bundle
        raw_bundle = Bundle(bundle_type="raw_experimental_data")
        raw_bundle.add_asset("file_path", hkl_file)
        raw_bundle.add_metadata("data_type", "intensities")
        raw_bundle.add_metadata(
            "data_labels",
            {
                "i_obs": i_obs_label,
                "sigma_i": sigma_label,
            },
        )

        # Store raw bundle
        raw_bundle_id = self.store_bundle(raw_bundle)

        # Process
        output_ids = self.run({"raw_experimental_data": raw_bundle_id})
        return output_ids["experimental_data"]

    def analyze_data_quality(self, exp_data_id: str) -> Dict[str, Any]:
        """
        Analyze experimental data quality.
        """
        exp_bundle = self.get_bundle(exp_data_id)
        f_obs = exp_bundle.get_asset("f_obs")
        sigmas = exp_bundle.get_asset("sigmas")
        metadata = exp_bundle.get_asset("experimental_metadata")

        analysis = {
            "total_reflections": f_obs.size(),
            "resolution_range": (f_obs.d_min(), f_obs.d_max()),
            "space_group": metadata.get("space_group", "unknown"),
            "unit_cell": metadata.get("unit_cell", "unknown"),
        }

        if sigmas is not None:
            sigma_f_ratios = sigmas.data() / f_obs.data()
            analysis.update(
                {
                    "mean_sigma_f": float(sigma_f_ratios.mean()),
                    "median_sigma_f": float(sigma_f_ratios.median()),
                    "completeness": self._calculate_completeness(f_obs),
                }
            )

        return analysis

    def _calculate_completeness(self, f_obs: Any) -> float:
        """
        Calculate data completeness.
        """
        # This is a simplified calculation
        # In practice, you'd compare against theoretical reflections
        return 1.0  # Placeholder

    def get_computation_info(self) -> Dict[str, Any]:
        """
        Get information about this processor's computational requirements.
        """
        return {
            "processor_type": "experimental_data_processor",
            "input_types": ["raw_experimental_data"],
            "output_types": ["experimental_data"],
            "memory_usage": "low",
            "cpu_usage": "medium",
            "gpu_usage": "none",
        }
