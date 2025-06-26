"""Agentbx - Crystallographic refinement with AI agents."""

from ._version import __version__
from .agents.base import SinglePurposeAgent
from .agents.experimental_data_agent import ExperimentalDataAgent
from .agents.gradient_agent import GradientAgent
from .agents.structure_factor_agent import StructureFactorAgent
from .agents.target_agent import TargetAgent
from .core.base_client import BaseClient
from .core.bundle_base import Bundle
from .core.redis_manager import RedisManager
from .schemas.generated import AtomicModelDataBundle
from .schemas.generated import ExperimentalDataBundle
from .schemas.generated import GradientDataBundle
from .schemas.generated import StructureFactorDataBundle
from .schemas.generated import TargetDataBundle


__author__ = "Petrus Zwart"
__email__ = "PHZwart@lbl.gov"

__all__ = [
    "__version__",
    "RedisManager",
    "BaseClient",
    "Bundle",
    "SinglePurposeAgent",
    "StructureFactorAgent",
    "TargetAgent",
    "GradientAgent",
    "ExperimentalDataAgent",
    "TargetDataBundle",
    "GradientDataBundle",
    "AtomicModelDataBundle",
    "ExperimentalDataBundle",
    "StructureFactorDataBundle",
]
