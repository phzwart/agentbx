"""
Processor responsible ONLY for geometry gradient calculations.

Input: xray_atomic_model_data
Output: geometry_gradient_data

Does NOT know about:
- Target functions
- Structure factors
- Optimization
- Experimental data
"""

import logging
import time
import os
from typing import Any
from typing import Dict
from typing import List

from ..core.bundle_base import Bundle
from .base import SinglePurposeProcessor


logger = logging.getLogger(__name__)


class CctbxGeometryProcessor(SinglePurposeProcessor):
    """
    Pure geometry gradient calculation processor.

    Responsibility: Compute geometry gradients from atomic models using CCTBX.
    """

    def define_input_bundle_types(self) -> List[str]:
        """Define the input bundle types for this processor."""
        return ["xray_atomic_model_data"]

    def define_output_bundle_types(self) -> List[str]:
        """Define the output bundle types for this processor."""
        return ["geometry_gradient_data"]

    def process_bundles(self, input_bundles: Dict[str, Bundle]) -> Dict[str, Bundle]:
        """
        Calculate geometry gradients from atomic model.
        """
        model_data = input_bundles["xray_atomic_model_data"]

        # Extract atomic model components
        xray_structure = model_data.get_asset("xray_structure")
        if xray_structure is None:
            raise ValueError("X-ray structure not found in model data")

        # Calculate geometry gradients
        geometry_gradients = self._calculate_geometry_gradients(xray_structure)

        # Create geometry gradient bundle
        geo_bundle = Bundle(bundle_type="geometry_gradient_data")
        geo_bundle.add_asset("geometry_gradients", geometry_gradients)
        geo_bundle.add_asset("gradient_norm", self._calculate_gradient_norm(geometry_gradients))

        # Add metadata
        geo_bundle.add_metadata("geometry_type", "bond_length_angle_dihedral")
        geo_bundle.add_metadata("calculation_method", "analytical")

        return {"geometry_gradient_data": geo_bundle}

    def _calculate_geometry_gradients(self, xray_structure: Any) -> Any:
        """
        Calculate geometry gradients using CCTBX.
        """
        # This is a placeholder implementation
        # In practice, this would use CCTBX's geometry restraints
        
        # For now, return a mock gradient object
        # In real implementation, this would be:
        # gradients = xray_structure.geometry_gradients()
        
        return None  # Placeholder

    def _calculate_gradient_norm(self, geometry_gradients: Any) -> float:
        """
        Calculate the norm of the geometry gradient vector.
        """
        # Placeholder implementation
        # In practice, this would calculate the L2 norm of all gradients
        return 0.0

    def calculate_geometry_gradients(self, model_bundle_id: str) -> str:
        """
        Calculate geometry gradients from atomic model bundle.
        """
        # Process the model bundle
        output_ids = self.run({"xray_atomic_model_data": model_bundle_id})
        return output_ids["geometry_gradient_data"]

    def get_computation_info(self) -> Dict[str, Any]:
        """
        Get information about this processor's computational requirements.
        """
        return {
            "processor_type": "geometry_processor",
            "input_types": ["xray_atomic_model_data"],
            "output_types": ["geometry_gradient_data"],
            "memory_usage": "medium",
            "cpu_usage": "medium",
            "gpu_usage": "none",
        }
```

Now let me create the new `__init__.py` file for the processors directory:

```python:src/agentbx/processors/__init__.py
"""Processor modules for agentbx."""

from .base import SinglePurposeProcessor
from .experimental_data_processor import ExperimentalDataProcessor
from .gradient_processor import GradientProcessor
from .structure_factor_processor import StructureFactorProcessor
from .target_processor import TargetProcessor
from .geometry_processor import CctbxGeometryProcessor


__all__ = [
    "SinglePurposeProcessor",
    "StructureFactorProcessor",
    "TargetProcessor",
    "GradientProcessor",
    "ExperimentalDataProcessor",
    "CctbxGeometryProcessor",
]
```

Now I need to update the main `__init__.py` file to import from the new processors directory:

```python:src/agentbx/__init__.py
"""AgentBX: Crystallographic data processing and analysis framework."""

from .processors.base import SinglePurposeProcessor
from .processors.experimental_data_processor import ExperimentalDataProcessor
from .processors.gradient_processor import GradientProcessor
from .processors.structure_factor_processor import StructureFactorProcessor
from .processors.target_processor import TargetProcessor
from .processors.geometry_processor import CctbxGeometryProcessor
from .core.bundle_base import Bundle
from .core.config import AgentConfig
from .core.config import RedisConfig
from .core.redis_manager import RedisManager
from .schemas.generated import XrayAtomicModelDataBundle
from .schemas.generated import ExperimentalDataBundle
from .schemas.generated import GradientDataBundle
from .schemas.generated import StructureFactorDataBundle
from .schemas.generated import TargetDataBundle
from .utils.cli import main


__version__ = "0.1.0"

__all__ = [
    "SinglePurp 