# Auto-generated Pydantic models from YAML schemas
# DO NOT EDIT - regenerate using SchemaGenerator

import hashlib
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator


class TargetDataBundle(BaseModel):
    """
    Target function values computed from structure factors and experimental data

    Generated from target_data.yaml
    """

    # Bundle metadata
    bundle_type: Literal["target_data"] = "target_data"
    created_at: datetime = Field(default_factory=datetime.now)
    bundle_id: Optional[str] = None
    checksum: Optional[str] = None

    # Required assets
    target_value: float = Field(description="Scalar target function value")
    target_type: str = Field(description="Type of target function used")

    # Optional assets
    r_factors: Dict[str, Any] = Field(
        default=None, description="Crystallographic R-factors"
    )
    target_per_reflection: Any = Field(
        default=None, description="Target contribution from each reflection"
    )
    likelihood_parameters: Dict[str, Any] = Field(
        default=None, description="Maximum likelihood alpha and beta parameters"
    )
    target_gradients_wrt_sf: Any = Field(
        default=None, description="Gradients of target w.r.t structure factors"
    )
    target_metadata: Dict[str, Any] = Field(
        default=None, description="Target computation metadata"
    )

    @field_validator("target_value")
    @classmethod
    def validate_target_value(cls, v):
        """Validate Scalar target function value"""
        return v

    @field_validator("r_factors")
    @classmethod
    def validate_r_factors(cls, v):
        """Validate Crystallographic R-factors"""
        return v

    @field_validator("likelihood_parameters")
    @classmethod
    def validate_likelihood_parameters(cls, v):
        """Validate Maximum likelihood alpha and beta parameters"""
        return v

    @field_validator("target_gradients_wrt_sf")
    @classmethod
    def validate_target_gradients_wrt_sf(cls, v):
        """Validate Gradients of target w.r.t structure factors"""
        if not hasattr(v, "indices"):
            raise ValueError("miller_array must have indices")
        if not hasattr(v, "data"):
            raise ValueError("miller_array must have data")
        if not v.is_complex_array():
            raise ValueError("miller_array must be complex")
        return v

    def calculate_checksum(self) -> str:
        """Calculate checksum of bundle contents."""
        # Implementation would hash all asset contents
        import json

        content = self.dict(exclude={"checksum", "created_at", "bundle_id"})
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def validate_dependencies(self, available_bundles: Dict[str, "BaseModel"]) -> bool:
        """Validate that all dependencies are satisfied."""
        required_deps: List[str] = ["structure_factor_data", "experimental_data"]
        for dep in required_deps:
            if dep not in available_bundles:
                raise ValueError(f"Missing dependency: {dep}")
        return True


class GradientDataBundle(BaseModel):
    """
    Gradients of target function w.r.t. atomic parameters via chain rule

    Generated from gradient_data.yaml
    """

    # Bundle metadata
    bundle_type: Literal["gradient_data"] = "gradient_data"
    created_at: datetime = Field(default_factory=datetime.now)
    bundle_id: Optional[str] = None
    checksum: Optional[str] = None

    # Required assets
    coordinate_gradients: Any = Field(
        description="Gradients w.r.t. atomic coordinates: dT/d(xyz)"
    )

    # Optional assets
    bfactor_gradients: Any = Field(
        default=None, description="Gradients w.r.t. B-factors: dT/d(B)"
    )
    occupancy_gradients: Any = Field(
        default=None, description="Gradients w.r.t. occupancies: dT/d(occ)"
    )
    structure_factor_gradients: Any = Field(
        default=None,
        description="Intermediate: gradients w.r.t. structure factors dT/dF",
    )
    gradient_metadata: Dict[str, Any] = Field(
        default=None, description="Gradient computation information"
    )

    @field_validator("coordinate_gradients")
    @classmethod
    def validate_coordinate_gradients(cls, v):
        """Validate Gradients w.r.t. atomic coordinates: dT/d(xyz)"""
        return v

    @field_validator("bfactor_gradients")
    @classmethod
    def validate_bfactor_gradients(cls, v):
        """Validate Gradients w.r.t. B-factors: dT/d(B)"""
        return v

    @field_validator("occupancy_gradients")
    @classmethod
    def validate_occupancy_gradients(cls, v):
        """Validate Gradients w.r.t. occupancies: dT/d(occ)"""
        return v

    def calculate_checksum(self) -> str:
        """Calculate checksum of bundle contents."""
        # Implementation would hash all asset contents
        import json

        content = self.dict(exclude={"checksum", "created_at", "bundle_id"})
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def validate_dependencies(self, available_bundles: Dict[str, "BaseModel"]) -> bool:
        """Validate that all dependencies are satisfied."""
        required_deps: List[str] = [
            "target_data",
            "structure_factor_data",
            "atomic_model_data",
        ]
        for dep in required_deps:
            if dep not in available_bundles:
                raise ValueError(f"Missing dependency: {dep}")
        return True


class AtomicModelDataBundle(BaseModel):
    """
    Atomic model data for structure factor calculations

    Generated from atomic_model_data.yaml
    """

    # Bundle metadata
    bundle_type: Literal["atomic_model_data"] = "atomic_model_data"
    created_at: datetime = Field(default_factory=datetime.now)
    bundle_id: Optional[str] = None
    checksum: Optional[str] = None

    # Required assets
    xray_structure: Any = Field(
        description="CCTBX xray.structure object with atomic model"
    )
    miller_indices: Any = Field(
        description="Miller indices for structure factor calculation"
    )

    # Optional assets
    bulk_solvent_params: Dict[str, Any] = Field(
        default=None, description="Bulk solvent correction parameters"
    )
    anisotropic_scaling_params: Dict[str, Any] = Field(
        default=None, description="Anisotropic scaling parameters"
    )
    model_metadata: Dict[str, Any] = Field(
        default=None, description="Model provenance and quality info"
    )

    @field_validator("xray_structure")
    @classmethod
    def validate_xray_structure(cls, v):
        """Validate CCTBX xray.structure object with atomic model"""
        if not hasattr(v, "scatterers"):
            raise ValueError("xray_structure must have scatterers")
        if not hasattr(v, "unit_cell"):
            raise ValueError("xray_structure must have unit_cell")
        return v

    @field_validator("miller_indices")
    @classmethod
    def validate_miller_indices(cls, v):
        """Validate Miller indices for structure factor calculation"""
        return v

    @field_validator("bulk_solvent_params")
    @classmethod
    def validate_bulk_solvent_params(cls, v):
        """Validate Bulk solvent correction parameters"""
        return v

    def calculate_checksum(self) -> str:
        """Calculate checksum of bundle contents."""
        # Implementation would hash all asset contents
        import json

        content = self.dict(exclude={"checksum", "created_at", "bundle_id"})
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def validate_dependencies(self, available_bundles: Dict[str, "BaseModel"]) -> bool:
        """Validate that all dependencies are satisfied."""
        required_deps: List[str] = []
        for dep in required_deps:
            if dep not in available_bundles:
                raise ValueError(f"Missing dependency: {dep}")
        return True


class ExperimentalDataBundle(BaseModel):
    """
    Experimental crystallographic data for refinement and validation

    Generated from experimental_data.yaml
    """

    # Bundle metadata
    bundle_type: Literal["experimental_data"] = "experimental_data"
    created_at: datetime = Field(default_factory=datetime.now)
    bundle_id: Optional[str] = None
    checksum: Optional[str] = None

    # Required assets
    f_obs: Any = Field(description="Observed structure factor amplitudes")
    miller_indices: Any = Field(
        description="Miller indices for experimental reflections"
    )

    # Optional assets
    r_free_flags: Any = Field(
        default=None, description="Free R flags for cross-validation"
    )
    sigmas: Any = Field(
        default=None, description="Uncertainties in observed structure factors"
    )
    i_obs: Any = Field(default=None, description="Observed intensities (if available)")
    anomalous_data: Dict[str, Any] = Field(
        default=None, description="Anomalous scattering data (F+, F-, or I+, I-)"
    )
    experimental_metadata: Dict[str, Any] = Field(
        default=None, description="Experimental conditions and data collection info"
    )
    target_preferences: Dict[str, Any] = Field(
        default=None, description="Preferred target function for this dataset"
    )

    @field_validator("f_obs")
    @classmethod
    def validate_f_obs(cls, v):
        """Validate Observed structure factor amplitudes"""
        if not hasattr(v, "indices"):
            raise ValueError("miller_array must have indices")
        if not hasattr(v, "data"):
            raise ValueError("miller_array must have data")
        if (v.data() < 0).count(True) > 0:
            raise ValueError("miller_array data must be positive")
        return v

    @field_validator("r_free_flags")
    @classmethod
    def validate_r_free_flags(cls, v):
        """Validate Free R flags for cross-validation"""
        if not hasattr(v, "indices"):
            raise ValueError("miller_array must have indices")
        if not hasattr(v, "data"):
            raise ValueError("miller_array must have data")
        return v

    @field_validator("sigmas")
    @classmethod
    def validate_sigmas(cls, v):
        """Validate Uncertainties in observed structure factors"""
        if not hasattr(v, "indices"):
            raise ValueError("miller_array must have indices")
        if not hasattr(v, "data"):
            raise ValueError("miller_array must have data")
        if (v.data() < 0).count(True) > 0:
            raise ValueError("miller_array data must be positive")
        return v

    def calculate_checksum(self) -> str:
        """Calculate checksum of bundle contents."""
        # Implementation would hash all asset contents
        import json

        content = self.dict(exclude={"checksum", "created_at", "bundle_id"})
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def validate_dependencies(self, available_bundles: Dict[str, "BaseModel"]) -> bool:
        """Validate that all dependencies are satisfied."""
        required_deps: List[str] = []
        for dep in required_deps:
            if dep not in available_bundles:
                raise ValueError(f"Missing dependency: {dep}")
        return True


class StructureFactorDataBundle(BaseModel):
    """
    Computed structure factors from atomic models

    Generated from structure_factor_data.yaml
    """

    # Bundle metadata
    bundle_type: Literal["structure_factor_data"] = "structure_factor_data"
    created_at: datetime = Field(default_factory=datetime.now)
    bundle_id: Optional[str] = None
    checksum: Optional[str] = None

    # Required assets
    f_calc: Any = Field(description="Calculated structure factors from atomic model")
    miller_indices: Any = Field(
        description="Miller indices corresponding to structure factors"
    )

    # Optional assets
    f_mask: Any = Field(
        default=None, description="Structure factors from bulk solvent mask"
    )
    f_model: Any = Field(
        default=None,
        description="Combined structure factors: scale * (f_calc + k_sol * f_mask)",
    )
    scale_factors: Dict[str, Any] = Field(
        default=None,
        description="Scaling parameters used in structure factor calculation",
    )
    computation_info: Dict[str, Any] = Field(
        default=None, description="Metadata about structure factor calculation"
    )

    @field_validator("f_calc")
    @classmethod
    def validate_f_calc(cls, v):
        """Validate Calculated structure factors from atomic model"""
        if not hasattr(v, "indices"):
            raise ValueError("miller_array must have indices")
        if not hasattr(v, "data"):
            raise ValueError("miller_array must have data")
        if not v.is_complex_array():
            raise ValueError("miller_array must be complex")
        import numpy as np

        if hasattr(v, "data") and not np.all(np.isfinite(v.data())):
            raise ValueError("All values must be finite")
        return v

    @field_validator("f_mask")
    @classmethod
    def validate_f_mask(cls, v):
        """Validate Structure factors from bulk solvent mask"""
        if not hasattr(v, "indices"):
            raise ValueError("miller_array must have indices")
        if not hasattr(v, "data"):
            raise ValueError("miller_array must have data")
        if not v.is_complex_array():
            raise ValueError("miller_array must be complex")
        return v

    @field_validator("f_model")
    @classmethod
    def validate_f_model(cls, v):
        """Validate Combined structure factors: scale * (f_calc + k_sol * f_mask)"""
        if not hasattr(v, "indices"):
            raise ValueError("miller_array must have indices")
        if not hasattr(v, "data"):
            raise ValueError("miller_array must have data")
        if not v.is_complex_array():
            raise ValueError("miller_array must be complex")
        return v

    @field_validator("scale_factors")
    @classmethod
    def validate_scale_factors(cls, v):
        """Validate Scaling parameters used in structure factor calculation"""
        return v

    def calculate_checksum(self) -> str:
        """Calculate checksum of bundle contents."""
        # Implementation would hash all asset contents
        import json

        content = self.dict(exclude={"checksum", "created_at", "bundle_id"})
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def validate_dependencies(self, available_bundles: Dict[str, "BaseModel"]) -> bool:
        """Validate that all dependencies are satisfied."""
        required_deps: List[str] = ["atomic_model_data"]
        for dep in required_deps:
            if dep not in available_bundles:
                raise ValueError(f"Missing dependency: {dep}")
        return True
