# src/agentbx/agents/experimental_data_agent.py
"""
Agent responsible ONLY for processing experimental data.

Input: raw_experimental_data (MTZ, HKL files, etc.)
Output: experimental_data

Does NOT know about:
- Atomic models
- Structure factors
- Target functions
"""

from typing import Dict, List, Optional
from pathlib import Path
from .base import SinglePurposeAgent
from ..core.bundle_base import Bundle


class ExperimentalDataAgent(SinglePurposeAgent):
    """
    Pure experimental data processing agent.
    
    Responsibility: Convert raw experimental files to clean experimental_data bundles.
    """
    
    def define_input_bundle_types(self) -> List[str]:
        return ["raw_experimental_data"]
    
    def define_output_bundle_types(self) -> List[str]:
        return ["experimental_data"]
    
    def process_bundles(self, input_bundles: Dict[str, Bundle]) -> Dict[str, Bundle]:
        """
        Process raw experimental data into clean experimental_data bundle.
        """
        raw_data = input_bundles["raw_experimental_data"]
        
        # Extract raw data information
        file_path = raw_data.get_asset("file_path")
        data_labels = raw_data.get_asset("data_labels", {})
        data_type = raw_data.get_asset("data_type", "amplitudes")  # or "intensities"
        
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
        
        return {"experimental_data": exp_bundle}
    
    def _process_reflection_file(self, file_path: str, data_labels: Dict, data_type: str):
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
            err=None
        )
        
        # Extract F_obs (or I_obs)
        f_obs_label = data_labels.get("f_obs", data_labels.get("i_obs"))
        sigma_label = data_labels.get("sigmas", data_labels.get("sigma_f", data_labels.get("sigma_i")))
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
            except:
                print(f"Warning: Could not read R_free flags from {r_free_label}")
        
        # Extract experimental metadata
        metadata = self._extract_metadata_from_file(reflection_file, file_path)
        
        return f_obs, sigmas, r_free_flags, metadata
    
    def _convert_intensities_to_amplitudes(self, i_obs, sig_i):
        """
        French-Wilson conversion of intensities to amplitudes.
        """
        from cctbx import french_wilson
        
        # Apply French-Wilson algorithm
        fw = french_wilson.french_wilson_scale(
            miller_array=i_obs,
            log=None
        )
        
        f_obs = fw.f_sq_as_f()
        sigmas = fw.sigmas()
        
        return f_obs, sigmas
    
    def _extract_metadata_from_file(self, reflection_file, file_path: str):
        """
        Extract experimental metadata from reflection file.
        """
        metadata = {
            "file_path": file_path,
            "file_type": Path(file_path).suffix.lower(),
        }
        
        # Extract crystal symmetry
        crystal_symmetry = reflection_file.file_content().crystal_symmetry()
        if crystal_symmetry:
            metadata.update({
                "space_group": str(crystal_symmetry.space_group_info()),
                "unit_cell": crystal_symmetry.unit_cell().parameters(),
            })
        
        # Extract resolution information
        try:
            miller_arrays = reflection_file.file_content().miller_arrays()
            if miller_arrays:
                d_max_min = miller_arrays[0].d_max_min()
                metadata.update({
                    "resolution_range": d_max_min,
                    "d_min": d_max_min[1],
                    "d_max": d_max_min[0]
                })
        except:
            pass
        
        # Try to extract wavelength and other experimental info
        try:
            # This would depend on file format specifics
            metadata["collection_date"] = "unknown"
            metadata["wavelength"] = "unknown"
            metadata["temperature"] = "unknown"
        except:
            pass
        
        return metadata
    
    def _validate_experimental_data(self, f_obs, sigmas, r_free_flags):
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
            mean_sigma_f = sigma_f_ratios.mean()
            if mean_sigma_f > 0.5:
                print(f"Warning: High sigma/F ratio ({mean_sigma_f:.3f}). Data quality may be poor.")
        
        # Check R_free fraction
        if r_free_flags is not None:
            free_fraction = r_free_flags.data().count(True) / r_free_flags.size()
            if free_fraction < 0.03 or free_fraction > 0.15:
                print(f"Warning: Unusual R_free fraction ({free_fraction:.3f}). Typical range is 5-10%.")
        
        # Check resolution
        d_max_min = f_obs.d_max_min()
        if d_max_min[1] > 3.0:
            print(f"Warning: Low resolution data (d_min = {d_max_min[1]:.2f} Å)")
    
    def _determine_target_preferences(self, f_obs, sigmas, metadata):
        """
        Determine recommended target function based on data quality.
        """
        target_prefs = {
            "default_target": "maximum_likelihood",
            "alternatives": ["least_squares"]
        }
        
        # Use least squares for very high resolution data
        d_min = metadata.get("d_min", 2.0)
        if d_min < 1.2:
            target_prefs["default_target"] = "least_squares"
            target_prefs["reason"] = "high_resolution_data"
        
        # Consider data quality
        if sigmas is not None:
            sigma_f_ratios = sigmas.data() / f_obs.data()
            mean_sigma_f = sigma_f_ratios.mean()
            
            if mean_sigma_f > 0.3:
                target_prefs["default_target"] = "maximum_likelihood"
                target_prefs["reason"] = "poor_data_quality"
        
        return target_prefs
    
    def _generate_r_free_flags(self, f_obs, fraction=0.05):
        """
        Generate R_free flags if not present in data.
        """
        from cctbx.array_family import flex
        import random
        
        n_reflections = f_obs.size()
        n_free = int(n_reflections * fraction)
        
        # Create random free flags
        flags = flex.bool(n_reflections, False)
        free_indices = random.sample(range(n_reflections), n_free)
        
        for i in free_indices:
            flags[i] = True
        
        r_free_flags = f_obs.array(data=flags)
        return r_free_flags
    
    def process_mtz_file(self, mtz_file: str, f_obs_label: str = "FP", 
                        sigma_label: str = "SIGFP", r_free_label: str = "FreeR_flag") -> str:
        """
        Convenience method to process an MTZ file directly.
        
        Returns:
            Bundle ID for the created experimental_data bundle
        """
        # Create raw data bundle
        raw_bundle = Bundle(bundle_type="raw_experimental_data")
        raw_bundle.add_asset("file_path", mtz_file)
        raw_bundle.add_asset("data_labels", {
            "f_obs": f_obs_label,
            "sigmas": sigma_label,
            "r_free_flags": r_free_label
        })
        raw_bundle.add_asset("data_type", "amplitudes")
        
        # Store raw bundle
        raw_bundle_id = self.store_bundle(raw_bundle)
        
        # Process it
        result = self.run({"raw_experimental_data": raw_bundle_id})
        return result["experimental_data"]
    
    def process_intensity_file(self, hkl_file: str, i_obs_label: str = "I", 
                              sigma_label: str = "SIGI") -> str:
        """
        Convenience method to process intensity data (will apply French-Wilson).
        
        Returns:
            Bundle ID for the created experimental_data bundle
        """
        # Create raw data bundle for intensities
        raw_bundle = Bundle(bundle_type="raw_experimental_data")
        raw_bundle.add_asset("file_path", hkl_file)
        raw_bundle.add_asset("data_labels", {
            "i_obs": i_obs_label,
            "sigmas": sigma_label
        })
        raw_bundle.add_asset("data_type", "intensities")
        
        # Store raw bundle
        raw_bundle_id = self.store_bundle(raw_bundle)
        
        # Process it
        result = self.run({"raw_experimental_data": raw_bundle_id})
        return result["experimental_data"]
    
    def analyze_data_quality(self, exp_data_id: str) -> Dict:
        """
        Analyze the quality of processed experimental data.
        
        Returns:
            Dict with data quality metrics
        """
        exp_bundle = self.get_bundle(exp_data_id)
        
        f_obs = exp_bundle.get_asset("f_obs")
        sigmas = exp_bundle.get_asset("sigmas")
        r_free_flags = exp_bundle.get_asset("r_free_flags")
        metadata = exp_bundle.get_asset("experimental_metadata")
        
        quality_metrics = {
            "n_reflections": f_obs.size(),
            "resolution_range": f_obs.d_max_min(),
            "completeness": self._calculate_completeness(f_obs),
        }
        
        if sigmas is not None:
            sigma_f_ratios = sigmas.data() / f_obs.data()
            quality_metrics.update({
                "mean_sigma_f_ratio": sigma_f_ratios.mean(),
                "sigma_f_cutoff_2": (sigma_f_ratios > 2.0).count(True) / f_obs.size(),
                "data_quality": "good" if sigma_f_ratios.mean() < 0.2 else "poor"
            })
        
        if r_free_flags is not None:
            free_fraction = r_free_flags.data().count(True) / r_free_flags.size()
            quality_metrics["r_free_fraction"] = free_fraction
        
        return quality_metrics
    
    def _calculate_completeness(self, f_obs):
        """
        Calculate data completeness for given resolution range.
        """
        # This is a simplified completeness calculation
        # Real implementation would consider systematic absences
        crystal_symmetry = f_obs.crystal_symmetry()
        complete_set = crystal_symmetry.build_miller_set(
            anomalous_flag=f_obs.anomalous_flag(),
            d_min=f_obs.d_min()
        )
        
        return f_obs.completeness(complete_set)
    
    def get_computation_info(self) -> Dict:
        """Return information about this agent's computation."""
        return {
            "agent_type": "ExperimentalDataAgent",
            "responsibility": "Experimental data processing",
            "supported_formats": ["MTZ", "HKL", "CIF", "SCA"],
            "algorithms": ["french_wilson", "data_validation", "completeness_analysis"],
            "cctbx_modules": ["iotbx.reflection_file_reader", "cctbx.french_wilson"],
            "convenience_methods": ["process_mtz_file", "process_intensity_file", "analyze_data_quality"]
        }
    
    def _convert_intensities_to_amplitudes(self, i_obs, sig_i):
        """
        French-Wilson conversion of intensities to amplitudes.
        """
        from cctbx import french_wilson
        
        # Apply French-Wilson algorithm
        fw = french_wilson.french_wilson_scale(
            miller_array=i_obs,
            log=None
        )
        
        f_obs = fw.f_sq_as_f()
        sigmas = fw.sigmas()
        
        return f_obs, sigmas
    
    def _extract_metadata_from_file(self, reflection_file, file_path: str):
        """
        Extract experimental metadata from reflection file.
        """
        metadata = {
            "file_path": file_path,
            "file_type": Path(file_path).suffix.lower(),
        }
        
        # Extract crystal symmetry
        crystal_symmetry = reflection_file.file_content().crystal_symmetry()
        if crystal_symmetry:
            metadata.update({
                "space_group": str(crystal_symmetry.space_group_info()),
                "unit_cell": crystal_symmetry.unit_cell().parameters(),
            })
        
        # Extract resolution information
        try:
            miller_arrays = reflection_file.file_content().miller_arrays()
            if miller_arrays:
                d_max_min = miller_arrays[0].d_max_min()
                metadata.update({
                    "resolution_range": d_max_min,
                    "d_min": d_max_min[1],
                    "d_max": d_max_min[0]
                })
        except:
            pass
        
        # Try to extract wavelength and other experimental info
        try:
            # This would depend on file format specifics
            metadata["collection_date"] = "unknown"
            metadata["wavelength"] = "unknown"
            metadata["temperature"] = "unknown"
        except:
            pass
        
        return metadata
    
    def _validate_experimental_data(self, f_obs, sigmas, r_free_flags):
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
            mean_sigma_f = sigma_f_ratios.mean()
            if mean_sigma_f > 0.5:
                print(f"Warning: High sigma/F ratio ({mean_sigma_f:.3f}). Data quality may be poor.")
        
        # Check R_free fraction
        if r_free_flags is not None:
            free_fraction = r_free_flags.data().count(True) / r_free_flags.size()
            if free_fraction < 0.03 or free_fraction > 0.15:
                print(f"Warning: Unusual R_free fraction ({free_fraction:.3f}). Typical range is 5-10%.")
        
        # Check resolution
        d_max_min = f_obs.d_max_min()
        if d_max_min[1] > 3.0:
            print(f"Warning: Low resolution data (d_min = {d_max_min[1]:.2f} Å)")
    
    def _determine_target_preferences(self, f_obs, sigmas, metadata):
        """
        Determine recommended target function based on data quality.
        """
        target_prefs = {
            "default_target": "maximum_likelihood",
            "alternatives": ["least_squares"]
        }
        
        # Use least squares for very high resolution data
        d_min = metadata.get("d_min", 2.0)
        if d_min < 1.2:
            target_prefs["default_target"] = "least_squares"
            target_prefs["reason"] = "high_resolution_data"
        
        # Consider data quality
        if sigmas is not None:
            sigma_f_ratios = sigmas.data() / f_obs.data()
            mean_sigma_f = sigma_f_ratios.mean()
            
            if mean_sigma_f > 0.3:
                target_prefs["default_target"] = "maximum_likelihood"
                target_prefs["reason"] = "poor_data_quality"
        
        return target_prefs
    
    def _generate_r_free_flags(self, f_obs, fraction=0.05):
        """
        Generate R_free flags if not present in data.
        """
        from cctbx.array_family import flex
        import random
        
        n_reflections = f_obs.size()
        n_free = int(n_reflections * fraction)
        
        # Create random free flags
        flags = flex.bool(n_reflections, False)
        free_indices = random.sample(range(n_reflections), n_free)
        
        for i in free_indices:
            flags[i] = True
        
        r_free_flags = f_obs.array(data=flags)
        return r_free_flags
    
    def get_computation_info(self) -> Dict:
        """Return information about this agent's computation."""
        return {
            "agent_type": "ExperimentalDataAgent",
            "responsibility": "Experimental data processing",
            "supported_formats": ["MTZ", "HKL", "CIF", "SCA"],
            "algorithms": ["french_wilson", "data_validation"],
            "cctbx_modules": ["iotbx.reflection_file_reader", "cctbx.french_wilson"]
        }