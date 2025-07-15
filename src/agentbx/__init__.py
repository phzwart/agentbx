"""AgentBX: Crystallographic data processing and analysis framework."""

from .core.bundle_base import Bundle
from .core.config import AgentConfig
from .core.config import RedisConfig
from .core.redis_manager import RedisManager
from .processors.base import SinglePurposeProcessor
from .processors.experimental_data_processor import ExperimentalDataProcessor
from .processors.geometry_processor import CctbxGeometryProcessor
from .processors.gradient_processor import GradientProcessor
from .processors.structure_factor_processor import StructureFactorProcessor
from .processors.target_processor import TargetProcessor
from .schemas.generated import ExperimentalDataBundle
from .schemas.generated import GradientDataBundle
from .schemas.generated import StructureFactorDataBundle
from .schemas.generated import TargetDataBundle
from .schemas.generated import XrayAtomicModelDataBundle
from .utils.cli import main


__version__ = "0.1.0"

__all__ = [
    "SinglePurposeProcessor",
    "StructureFactorProcessor",
    "TargetProcessor",
    "GradientProcessor",
    "ExperimentalDataProcessor",
    "CctbxGeometryProcessor",
    "Bundle",
    "RedisConfig",
    "AgentConfig",
    "RedisManager",
    "XrayAtomicModelDataBundle",
    "ExperimentalDataBundle",
    "GradientDataBundle",
    "StructureFactorDataBundle",
    "TargetDataBundle",
    "main",
]
