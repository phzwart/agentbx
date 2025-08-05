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
        observation_type = raw_data.get_metadata(
            "data_type", "amplitudes"
        )  # or "intensities"

        # Process the reflection file
        data_obs, sigmas, r_free_flags, metadata, observation_type = (
            self._process_reflection_file(file_path, data_labels, observation_type)
        )

        # Do not convert intensities to amplitudes; just use as-is

        # Validate data quality
        self._validate_experimental_data(
            data_obs, sigmas, r_free_flags, observation_type
        )

        # Create experimental data bundle
        exp_bundle = Bundle(bundle_type="experimental_data")
        exp_bundle.add_asset("data_obs", data_obs)
        exp_bundle.add_asset("miller_indices", data_obs.indices())

        if sigmas is not None:
            exp_bundle.add_asset("sigmas", sigmas)

        if r_free_flags is not None:
            exp_bundle.add_asset("r_free_flags", r_free_flags)

        # Add experimental metadata
        exp_bundle.add_asset("experimental_metadata", metadata)

        # Add target preferences based on data quality
        target_prefs = self._determine_target_preferences(data_obs, sigmas, metadata)
        exp_bundle.add_asset("target_preferences", target_prefs)

        # Only include sigmas and r_free_flags if they are Miller arrays
        sigmas_for_schema = (
            sigmas if hasattr(sigmas, "indices") and hasattr(sigmas, "data") else None
        )
        r_free_flags_for_schema = (
            r_free_flags
            if hasattr(r_free_flags, "indices") and hasattr(r_free_flags, "data")
            else None
        )
        data_obs_for_schema = (
            data_obs
            if hasattr(data_obs, "indices") and hasattr(data_obs, "data")
            else None
        )

        # If sigmas are part of the loaded Miller array, use them
        if hasattr(data_obs, "sigmas") and callable(getattr(data_obs, "sigmas", None)):
            sigmas_from_array = data_obs.sigmas()
            if (
                sigmas_from_array is not None
                and hasattr(sigmas_from_array, "indices")
                and hasattr(sigmas_from_array, "data")
            ):
                sigmas_for_schema = sigmas_from_array

        # Only include sigmas and r_free_flags if they are Miller arrays
        sigmas_for_schema = (
            sigmas_for_schema
            if hasattr(sigmas_for_schema, "indices")
            and hasattr(sigmas_for_schema, "data")
            else None
        )
        r_free_flags_for_schema = (
            r_free_flags
            if hasattr(r_free_flags, "indices") and hasattr(r_free_flags, "data")
            else None
        )
        data_obs_for_schema = (
            data_obs
            if hasattr(data_obs, "indices") and hasattr(data_obs, "data")
            else None
        )

        # Build kwargs for ExperimentalDataBundle
        bundle_kwargs = dict(
            data_obs=data_obs_for_schema,
            miller_indices=data_obs.indices(),
            experimental_metadata=metadata,
            target_preferences=target_prefs,
        )
        if sigmas_for_schema is not None:
            bundle_kwargs["sigmas"] = sigmas_for_schema
        if r_free_flags_for_schema is not None:
            bundle_kwargs["r_free_flags"] = r_free_flags_for_schema

        # Validate with schema
        ExperimentalDataBundle(**bundle_kwargs)
        self.logger.info(
            "[Schema Validation] ExperimentalDataBundle validation successful."
        )
        return {"experimental_data": exp_bundle}

    def _process_reflection_file(
        self, file_path: str, data_labels: Dict[str, Any], observation_type: str
    ) -> tuple[Any, Any, Any, Dict[str, Any], str]:
        """
        Process MTZ/HKL file to extract observed data, sigmas, R_free.
        """
        from iotbx import reflection_file_reader
        from iotbx.reflection_file_utils import reflection_file_server

        # Helper to print min/max or value for flex arrays or scalars
        def _print_min_max(label, arr):
            # If arr has .data(), use it; otherwise, treat arr as the data
            data = arr.data() if hasattr(arr, "data") else arr
            try:
                if hasattr(data, "min") and hasattr(data, "max"):
                    print(
                        f"[Summary] {label} min/max:",
                        float(data.min()),
                        float(data.max()),
                    )
                elif hasattr(data, "__len__") and len(data) == 1:
                    print(f"[Summary] {label} value:", float(data[0]))
                elif hasattr(data, "as_double") and hasattr(data, "__getitem__"):
                    print(f"[Summary] {label} value:", float(data.as_double()[0]))
                else:
                    print(f"[Summary] {label} value:", float(data))
            except Exception:
                print(f"[Summary] {label} value (repr):", repr(data))

        # Read reflection file
        reflection_file = reflection_file_reader.any_reflection_file(file_path)

        # Debug: print all available Miller arrays and their labels
        print("Available Miller arrays in file:")
        for arr in reflection_file.as_miller_arrays():
            print("  ", arr.info().label_string())

        if reflection_file is None:
            raise ValueError(f"Could not read reflection file: {file_path}")

        # Create reflection file server
        server = reflection_file_server(
            crystal_symmetry=None,
            force_symmetry=True,
            reflection_files=[reflection_file],
            err=None,
        )

        # Get all available labels
        available_labels = [
            arr.info().label_string() for arr in reflection_file.as_miller_arrays()
        ]

        # Helper for substring matching
        def find_label(label):
            if not label:
                return None
            if label in available_labels:
                return label
            for l in available_labels:
                if label in l:
                    return l
            return None

        # NEW: If 'data' is specified, use it directly
        data_label = data_labels.get("data")
        if data_label:
            actual_label = find_label(data_label)
            if not actual_label:
                raise ValueError(f"Could not find data column: {data_label}")
            # Get the array
            arr = server.get_miller_array(actual_label)
            # Heuristic: if 'intensity' in label, treat as intensities, else amplitudes
            if "intensity" in actual_label.lower():
                observation_type = "intensities"
            else:
                observation_type = "amplitudes"
            # If the array is a merged amplitude+sigma or intensity+sigma, split
            if hasattr(arr, "sigmas") and arr.sigmas() is not None:
                data_obs = arr
                sigmas = arr.sigmas()
            else:
                data_obs = arr
                sigmas = None
            r_free_flags = None
            # Try to find a matching R-free array in the same dataset
            dataset_prefix = actual_label.split(",")[0]
            rfree_label = next(
                (
                    l
                    for l in available_labels
                    if dataset_prefix in l and "free" in l.lower()
                ),
                None,
            )
            if rfree_label:
                try:
                    r_free_flags = server.get_miller_array(rfree_label)
                    if r_free_flags is not None:
                        r_free_flags = r_free_flags.as_bool()
                except Exception:
                    self.logger.warning(
                        f"Could not read R_free flags from {rfree_label}"
                    )
            metadata = self._extract_metadata_from_file(reflection_file, file_path)
            # Print summary for Miller arrays
            print("[Summary] Miller array type:", observation_type)
            print("[Summary] data_obs size:", data_obs.size())
            _print_min_max("data_obs", data_obs)
            if sigmas is not None:
                print("[Summary] sigmas size:", sigmas.size())
                _print_min_max("sigmas", sigmas)
            if r_free_flags is not None:
                print("[Summary] r_free_flags size:", r_free_flags.size())
                print(
                    "[Summary] r_free_flags fraction free:",
                    float(r_free_flags.data().count(True)) / r_free_flags.size(),
                )
            return data_obs, sigmas, r_free_flags, metadata, observation_type

        # Extract raw data information
        f_obs_label = data_labels.get("f_obs", data_labels.get("i_obs"))
        sigma_label = data_labels.get(
            "sigmas", data_labels.get("sigma_f", data_labels.get("sigma_i"))
        )
        r_free_label = data_labels.get("r_free_flags", "FreeR_flag")

        # Get all available labels
        available_labels = [
            arr.info().label_string() for arr in reflection_file.as_miller_arrays()
        ]

        # Helper for substring matching
        def find_label(label):
            if not label:
                return None
            # Try exact match first
            if label in available_labels:
                return label
            # Try substring match
            for l in available_labels:
                if label in l:
                    return l
            return None

        # Get Miller arrays with substring matching
        f_obs_actual_label = find_label(f_obs_label)
        f_obs = (
            server.get_miller_array(f_obs_actual_label) if f_obs_actual_label else None
        )
        if f_obs is None:
            raise ValueError(f"Could not find data column: {f_obs_label}")

        sigmas = None
        if sigma_label:
            sigma_actual_label = find_label(sigma_label)
            sigmas = (
                server.get_miller_array(sigma_actual_label)
                if sigma_actual_label
                else None
            )

        r_free_flags = None
        if r_free_label:
            r_free_actual_label = find_label(r_free_label)
            try:
                r_free_flags = (
                    server.get_miller_array(r_free_actual_label)
                    if r_free_actual_label
                    else None
                )
                if r_free_flags is not None:
                    r_free_flags = r_free_flags.as_bool()
            except Exception:
                self.logger.warning(f"Could not read R_free flags from {r_free_label}")

        # Extract experimental metadata
        metadata = self._extract_metadata_from_file(reflection_file, file_path)

        return f_obs, sigmas, r_free_flags, metadata, observation_type

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
            "space_group": None,
            "unit_cell": None,
            "wavelength": 1.0,  # Default
            "temperature": "unknown",
        }

        # Try to extract symmetry from the file object
        try:
            if hasattr(reflection_file, "space_group_info"):
                metadata["space_group"] = str(reflection_file.space_group_info())
            if hasattr(reflection_file, "unit_cell"):
                metadata["unit_cell"] = str(reflection_file.unit_cell())
        except Exception:
            pass

        # If not found, try to get from the first Miller array
        try:
            arrays = reflection_file.as_miller_arrays()
            if arrays:
                arr = arrays[0]
                if metadata["space_group"] is None and hasattr(arr, "space_group_info"):
                    metadata["space_group"] = str(arr.space_group_info())
                if metadata["unit_cell"] is None and hasattr(arr, "unit_cell"):
                    metadata["unit_cell"] = str(arr.unit_cell())
        except Exception:
            pass

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
        self,
        data_obs: Any,
        sigmas: Any,
        r_free_flags: Any,
        observation_type: str = "amplitudes",
    ) -> None:
        """
        Validate experimental data quality.
        """
        # Robustly extract data arrays
        obs_data = data_obs.data() if hasattr(data_obs, "data") else data_obs
        sigmas_data = (
            sigmas.data()
            if (sigmas is not None and hasattr(sigmas, "data"))
            else sigmas
        )

        # Check data completeness
        if hasattr(data_obs, "size") and data_obs.size() == 0:
            raise ValueError("No reflections found in experimental data")
        elif (
            not hasattr(data_obs, "size")
            and hasattr(obs_data, "__len__")
            and len(obs_data) == 0
        ):
            raise ValueError("No reflections found in experimental data")

        # Only check for negative amplitudes if observation_type is amplitudes
        if observation_type == "amplitudes":
            if hasattr(obs_data, "__lt__") and hasattr(obs_data, "count"):
                if (obs_data < 0).count(True) > 0:
                    raise ValueError("Found negative structure factor amplitudes")
            elif hasattr(obs_data, "__iter__"):
                if any(x < 0 for x in obs_data):
                    raise ValueError("Found negative structure factor amplitudes")
            elif obs_data < 0:
                raise ValueError("Found negative structure factor amplitudes")

        # Check sigma/F ratios if sigmas available
        if sigmas is not None and sigmas_data is not None:
            try:
                sigma_f_ratios = sigmas_data / obs_data
                if hasattr(sigma_f_ratios, "count") and hasattr(
                    sigma_f_ratios, "__gt__"
                ):
                    if (sigma_f_ratios > 10).count(True) > (
                        data_obs.size() if hasattr(data_obs, "size") else len(obs_data)
                    ) * 0.1:
                        self.logger.warning("Many reflections have high sigma/F ratios")
                elif hasattr(sigma_f_ratios, "__iter__"):
                    if (
                        sum(x > 10 for x in sigma_f_ratios)
                        > (
                            data_obs.size()
                            if hasattr(data_obs, "size")
                            else len(obs_data)
                        )
                        * 0.1
                    ):
                        self.logger.warning("Many reflections have high sigma/F ratios")
                elif sigma_f_ratios > 10:
                    self.logger.warning("High sigma/F ratio for single observation")
            except Exception:
                self.logger.warning("Could not compute sigma/F ratios for validation")

        # Check R_free completeness if available
        if r_free_flags is not None:
            r_free_data = (
                r_free_flags.data() if hasattr(r_free_flags, "data") else r_free_flags
            )
            r_free_size = (
                r_free_flags.size()
                if hasattr(r_free_flags, "size")
                else (len(r_free_data) if hasattr(r_free_data, "__len__") else 1)
            )
            r_free_count = (
                r_free_data.count(True)
                if hasattr(r_free_data, "count")
                else sum(1 for x in r_free_data if x)
            )
            r_free_fraction = r_free_count / r_free_size
            if r_free_fraction < 0.01 or r_free_fraction > 0.2:
                self.logger.warning(
                    f"R_free fraction ({r_free_fraction:.3f}) outside normal range"
                )

    def _determine_target_preferences(
        self, data_obs: Any, sigmas: Any, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        preferences = {
            "default_target": "maximum_likelihood",
            "use_anomalous": False,
            "use_twinning": False,
        }

        # Check for anomalous data
        if hasattr(data_obs, "anomalous_flag") and data_obs.anomalous_flag():
            preferences["use_anomalous"] = True

        # Check data quality for target selection
        if sigmas is not None:
            obs_data = data_obs.data() if hasattr(data_obs, "data") else data_obs
            sigmas_data = sigmas.data() if hasattr(sigmas, "data") else sigmas
            sigma_f_ratios = sigmas_data / obs_data
            if hasattr(sigma_f_ratios, "mean"):
                mean_sigma_f = sigma_f_ratios.mean()
            elif hasattr(sigma_f_ratios, "__len__") and len(sigma_f_ratios) > 0:
                mean_sigma_f = sum(sigma_f_ratios) / len(sigma_f_ratios)
            else:
                mean_sigma_f = float(sigma_f_ratios)

            if mean_sigma_f > 0.5:
                preferences["default_target"] = "least_squares"
            elif mean_sigma_f < 0.1:
                pass
            else:
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
