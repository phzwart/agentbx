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
    r_factors: Optional[Dict[str, Any]] = Field(
        default=None, description="Crystallographic R-factors"
    )
    target_per_reflection: Optional[Any] = Field(
        default=None, description="Target contribution from each reflection"
    )
    likelihood_parameters: Optional[Dict[str, Any]] = Field(
        default=None, description="Maximum likelihood alpha and beta parameters"
    )
    target_gradients_wrt_sf: Optional[Any] = Field(
        default=None, description="Gradients of target w.r.t structure factors"
    )
    target_metadata: Optional[Dict[str, Any]] = Field(
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
        required_deps: List[str] = ["structure_factor_data", "experimental_data"] or []
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
    bfactor_gradients: Optional[Any] = Field(
        default=None, description="Gradients w.r.t. B-factors: dT/d(B)"
    )
    occupancy_gradients: Optional[Any] = Field(
        default=None, description="Gradients w.r.t. occupancies: dT/d(occ)"
    )
    structure_factor_gradients: Optional[Any] = Field(
        default=None,
        description="Intermediate: gradients w.r.t. structure factors dT/dF",
    )
    gradient_metadata: Optional[Dict[str, Any]] = Field(
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
            "xray_atomic_model_data",
            "target_data",
            "structure_factor_data",
        ] or []
        for dep in required_deps:
            if dep not in available_bundles:
                raise ValueError(f"Missing dependency: {dep}")
        return True


class GeometryGradientDataBundle(BaseModel):
    """
    Geometry gradients computed from CCTBX geometry restraints

    Generated from geometry_gradient_data.yaml
    """

    # Bundle metadata
    bundle_type: Literal["geometry_gradient_data"] = "geometry_gradient_data"
    created_at: datetime = Field(default_factory=datetime.now)
    bundle_id: Optional[str] = None
    checksum: Optional[str] = None

    # Required assets
    coordinates: Any = Field(description="Atomic coordinates in Cartesian space")
    geometric_gradients: Any = Field(
        description="Gradients of geometry restraints w.r.t. coordinates"
    )

    # Optional assets
    restraint_energies: Optional[Dict[str, Any]] = Field(
        default=None, description="Individual restraint energies by type"
    )
    restraint_counts: Optional[Dict[str, Any]] = Field(
        default=None, description="Number of restraints by type"
    )
    geometry_metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Metadata about geometry restraint computation"
    )

    @field_validator("coordinates")
    @classmethod
    def validate_coordinates(cls, v):
        """Validate Atomic coordinates in Cartesian space"""
        return v

    @field_validator("geometric_gradients")
    @classmethod
    def validate_geometric_gradients(cls, v):
        """Validate Gradients of geometry restraints w.r.t. coordinates"""
        return v

    @field_validator("restraint_energies")
    @classmethod
    def validate_restraint_energies(cls, v):
        """Validate Individual restraint energies by type"""
        return v

    @field_validator("restraint_counts")
    @classmethod
    def validate_restraint_counts(cls, v):
        """Validate Number of restraints by type"""
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
        required_deps: List[str] = ["xray_atomic_model_data"] or []
        for dep in required_deps:
            if dep not in available_bundles:
                raise ValueError(f"Missing dependency: {dep}")
        return True


class AgentConfigurationBundle(BaseModel):
    """
    Agent configuration and capability definitions

    Generated from agent_configuration.yaml
    """

    # Bundle metadata
    bundle_type: Literal["agent_configuration"] = "agent_configuration"
    created_at: datetime = Field(default_factory=datetime.now)
    bundle_id: Optional[str] = None
    checksum: Optional[str] = None

    # Required assets
    agent_definition: Dict[str, Any] = Field(description="Basic agent definition")
    capabilities: Any = Field(description="Agent capabilities and their configurations")

    # Optional assets
    security_policies: Optional[Dict[str, Any]] = Field(
        default=None, description="Security policies for the agent"
    )
    resource_limits: Optional[Dict[str, Any]] = Field(
        default=None, description="Resource limits for the agent"
    )
    monitoring_config: Optional[Dict[str, Any]] = Field(
        default=None, description="Monitoring configuration for the agent"
    )

    @field_validator("agent_definition")
    @classmethod
    def validate_agent_definition(cls, v):
        """Validate Basic agent definition"""
        return v

    @field_validator("capabilities")
    @classmethod
    def validate_capabilities(cls, v):
        """Validate Agent capabilities and their configurations"""
        return v

    @field_validator("security_policies")
    @classmethod
    def validate_security_policies(cls, v):
        """Validate Security policies for the agent"""
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
        required_deps: List[str] = ["agent_security", "redis_streams"] or []
        for dep in required_deps:
            if dep not in available_bundles:
                raise ValueError(f"Missing dependency: {dep}")
        return True


class MacromoleculeDataBundle(BaseModel):
    """
    Central macromolecule representation with PDB hierarchy

    Generated from macromolecule_data.yaml
    """

    # Bundle metadata
    bundle_type: Literal["macromolecule_data"] = "macromolecule_data"
    created_at: datetime = Field(default_factory=datetime.now)
    bundle_id: Optional[str] = None
    checksum: Optional[str] = None

    # Required assets
    pdb_hierarchy: Any = Field(
        description="CCTBX PDB hierarchy with full atomic model information"
    )
    crystal_symmetry: Any = Field(
        description="Crystal symmetry (unit cell and space group)"
    )

    # Optional assets
    model_manager: Optional[Any] = Field(
        default=None, description="MMTBX model manager with geometry restraints"
    )
    restraint_manager: Optional[Any] = Field(
        default=None, description="Geometry restraints manager"
    )
    xray_structure: Optional[Any] = Field(
        default=None, description="X-ray structure derived from PDB hierarchy"
    )
    macromolecule_metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Macromolecule provenance and quality info"
    )

    @field_validator("pdb_hierarchy")
    @classmethod
    def validate_pdb_hierarchy(cls, v):
        """Validate CCTBX PDB hierarchy with full atomic model information"""
        return v

    @field_validator("crystal_symmetry")
    @classmethod
    def validate_crystal_symmetry(cls, v):
        """Validate Crystal symmetry (unit cell and space group)"""
        return v

    @field_validator("model_manager")
    @classmethod
    def validate_model_manager(cls, v):
        """Validate MMTBX model manager with geometry restraints"""
        return v

    @field_validator("xray_structure")
    @classmethod
    def validate_xray_structure(cls, v):
        """Validate X-ray structure derived from PDB hierarchy"""
        if not hasattr(v, "scatterers"):
            raise ValueError("xray_structure must have scatterers")
        if not hasattr(v, "unit_cell"):
            raise ValueError("xray_structure must have unit_cell")
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
        required_deps: List[str] = ["pdb_file"] or []
        for dep in required_deps:
            if dep not in available_bundles:
                raise ValueError(f"Missing dependency: {dep}")
        return True


class XrayAtomicModelDataBundle(BaseModel):
    """
    Atomic model data for structure factor calculations

    Generated from xray_atomic_model_data.yaml
    """

    # Bundle metadata
    bundle_type: Literal["xray_atomic_model_data"] = "xray_atomic_model_data"
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
    bulk_solvent_params: Optional[Dict[str, Any]] = Field(
        default=None, description="Bulk solvent correction parameters"
    )
    anisotropic_scaling_params: Optional[Dict[str, Any]] = Field(
        default=None, description="Anisotropic scaling parameters"
    )
    model_metadata: Optional[Dict[str, Any]] = Field(
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
        required_deps: List[str] = [] or []
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
    r_free_flags: Optional[Any] = Field(
        default=None, description="Free R flags for cross-validation"
    )
    sigmas: Optional[Any] = Field(
        default=None, description="Uncertainties in observed structure factors"
    )
    i_obs: Optional[Any] = Field(
        default=None, description="Observed intensities (if available)"
    )
    anomalous_data: Optional[Dict[str, Any]] = Field(
        default=None, description="Anomalous scattering data (F+, F-, or I+, I-)"
    )
    experimental_metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Experimental conditions and data collection info"
    )
    target_preferences: Optional[Dict[str, Any]] = Field(
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
        required_deps: List[str] = [] or []
        for dep in required_deps:
            if dep not in available_bundles:
                raise ValueError(f"Missing dependency: {dep}")
        return True


class AgentSecurityBundle(BaseModel):
    """
    Agent security and authorization configuration

    Generated from agent_security.yaml
    """

    # Bundle metadata
    bundle_type: Literal["agent_security"] = "agent_security"
    created_at: datetime = Field(default_factory=datetime.now)
    bundle_id: Optional[str] = None
    checksum: Optional[str] = None

    # Required assets
    agent_registration: Any = Field(
        description="Agent registration and validation information"
    )
    permissions: Any = Field(description="List of granted permissions for the agent")

    # Optional assets
    capabilities: Optional[Any] = Field(
        default=None, description="Agent capabilities and their schemas"
    )
    whitelisted_modules: Optional[Any] = Field(
        default=None, description="Modules the agent is allowed to import and use"
    )
    instruction_triggers: Optional[Any] = Field(
        default=None, description="Valid instruction triggers for this agent"
    )
    security_policies: Optional[Dict[str, Any]] = Field(
        default=None, description="Security policies for the agent"
    )

    @field_validator("agent_registration")
    @classmethod
    def validate_agent_registration(cls, v):
        """Validate Agent registration and validation information"""
        return v

    @field_validator("permissions")
    @classmethod
    def validate_permissions(cls, v):
        """Validate List of granted permissions for the agent"""
        return v

    @field_validator("capabilities")
    @classmethod
    def validate_capabilities(cls, v):
        """Validate Agent capabilities and their schemas"""
        return v

    @field_validator("whitelisted_modules")
    @classmethod
    def validate_whitelisted_modules(cls, v):
        """Validate Modules the agent is allowed to import and use"""
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
        required_deps: List[str] = ["agent_configuration"] or []
        for dep in required_deps:
            if dep not in available_bundles:
                raise ValueError(f"Missing dependency: {dep}")
        return True


class RedisStreamsBundle(BaseModel):
    """
    Redis stream configuration for agent communication

    Generated from redis_streams.yaml
    """

    # Bundle metadata
    bundle_type: Literal["redis_streams"] = "redis_streams"
    created_at: datetime = Field(default_factory=datetime.now)
    bundle_id: Optional[str] = None
    checksum: Optional[str] = None

    # Required assets
    stream_configuration: Dict[str, Any] = Field(
        description="Configuration for Redis streams"
    )
    consumer_groups: Any = Field(description="Consumer group configurations")

    # Optional assets
    message_schemas: Optional[Dict[str, Any]] = Field(
        default=None, description="JSON schemas for message validation"
    )
    retry_policies: Optional[Dict[str, Any]] = Field(
        default=None, description="Retry policies for failed messages"
    )
    monitoring_config: Optional[Dict[str, Any]] = Field(
        default=None, description="Monitoring configuration for streams"
    )

    @field_validator("stream_configuration")
    @classmethod
    def validate_stream_configuration(cls, v):
        """Validate Configuration for Redis streams"""
        return v

    @field_validator("consumer_groups")
    @classmethod
    def validate_consumer_groups(cls, v):
        """Validate Consumer group configurations"""
        return v

    @field_validator("message_schemas")
    @classmethod
    def validate_message_schemas(cls, v):
        """Validate JSON schemas for message validation"""
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
        required_deps: List[str] = ["redis_connection"] or []
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
    f_mask: Optional[Any] = Field(
        default=None, description="Structure factors from bulk solvent mask"
    )
    f_model: Optional[Any] = Field(
        default=None,
        description="Combined structure factors: scale * (f_calc + k_sol * f_mask)",
    )
    scale_factors: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Scaling parameters used in structure factor calculation",
    )
    computation_info: Optional[Dict[str, Any]] = Field(
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
        required_deps: List[str] = ["xray_atomic_model_data"] or []
        for dep in required_deps:
            if dep not in available_bundles:
                raise ValueError(f"Missing dependency: {dep}")
        return True


class CoordinateUpdateBundle(BaseModel):
    """
    Coordinate update bundle for geometry minimization and agent communication.

    Generated from coordinate_update.yaml
    """

    # Bundle metadata
    bundle_type: Literal["coordinate_update"] = "coordinate_update"
    created_at: datetime = Field(default_factory=datetime.now)
    bundle_id: Optional[str] = None
    checksum: Optional[str] = None

    # Required assets
    coordinates: Any = Field(
        description="Atomic coordinates after update (shape: [N, 3])"
    )
    parent_bundle_id: str = Field(description="ID of the parent macromolecule bundle")

    # Optional assets
    step: Optional[int] = Field(default=None, description="Optimization step number")
    timestamp: Optional[float] = Field(
        default=None, description="Timestamp of update (seconds since epoch)"
    )

    def calculate_checksum(self) -> str:
        """Calculate checksum of bundle contents."""
        # Implementation would hash all asset contents
        import json

        content = self.dict(exclude={"checksum", "created_at", "bundle_id"})
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def validate_dependencies(self, available_bundles: Dict[str, "BaseModel"]) -> bool:
        """Validate that all dependencies are satisfied."""
        required_deps: List[str] = [] or []
        for dep in required_deps:
            if dep not in available_bundles:
                raise ValueError(f"Missing dependency: {dep}")
        return True
