"""Agentbx - Crystallographic refinement with AI agents."""

from .core.redis_manager import RedisManager
from .core.base_client import BaseClient
from .core.bundle_base import Bundle
from .agents.base import SinglePurposeAgent
from .agents.structure_factor_agent import StructureFactorAgent
from .agents.target_agent import TargetAgent
from .agents.gradient_agent import GradientAgent
from .agents.experimental_data_agent import ExperimentalDataAgent
from .schemas.generated import (
    TargetDataBundle,
    GradientDataBundle,
    AtomicModelDataBundle,
    ExperimentalDataBundle,
    StructureFactorDataBundle,
)

__version__ = "0.1.0"
__author__ = "Petrus Zwart"
__email__ = "PHZwart@lbl.gov"

__all__ = [
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
