"""AgentBX: Crystallographic data processing and analysis framework."""

from .agents.base import SinglePurposeProcessor
from .agents.experimental_data_agent import ExperimentalDataProcessor
from .agents.gradient_agent import GradientProcessor
from .agents.structure_factor_agent import StructureFactorProcessor
from .agents.target_agent import TargetProcessor
from .core.bundle_base import Bundle
from .core.config import AgentConfig
from .core.config import RedisConfig
from .core.redis_manager import RedisManager
from .schemas.generated import AtomicModelDataBundle
from .schemas.generated import ExperimentalDataBundle
from .schemas.generated import GradientDataBundle
from .schemas.generated import StructureFactorDataBundle
from .schemas.generated import TargetDataBundle
from .utils.cli import main


__version__ = "0.1.0"

__all__ = [
    "SinglePurposeProcessor",
    "StructureFactorProcessor",
    "TargetProcessor",
    "GradientProcessor",
    "ExperimentalDataProcessor",
    "Bundle",
    "RedisConfig",
    "AgentConfig",
    "RedisManager",
    "AtomicModelDataBundle",
    "ExperimentalDataBundle",
    "GradientDataBundle",
    "StructureFactorDataBundle",
    "TargetDataBundle",
    "main",
]
