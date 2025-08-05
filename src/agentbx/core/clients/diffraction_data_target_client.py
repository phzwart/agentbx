from agentbx.core.base_client import BaseClient
from agentbx.core.bundle_base import Bundle
from agentbx.core.processors.experimental_data_processor import (
    ExperimentalDataProcessor,
)
from agentbx.core.processors.structure_factor_processor import StructureFactorProcessor
from agentbx.core.processors.target_processor import TargetProcessor


class DiffractionDataTargetClient(BaseClient):
    """
    Client for handling diffraction data and target function calculations.
    Provides methods to fetch experimental data bundles, compute targets,
    and interface with structure factor and target processors.
    """

    def __init__(self, redis_manager, client_id="diffraction_data_target_client"):
        super().__init__(redis_manager, client_id)
        self.exp_processor = ExperimentalDataProcessor(
            redis_manager, f"{client_id}_exp"
        )
        self.sf_processor = StructureFactorProcessor(redis_manager, f"{client_id}_sf")
        self.target_processor = TargetProcessor(redis_manager, f"{client_id}_target")

    def fetch_experimental_data_bundle(self, bundle_id: str) -> Bundle:
        """Fetch an experimental data bundle by ID."""
        return self.get_bundle(bundle_id)

    def compute_structure_factors(self, model_bundle_id: str) -> str:
        """Compute structure factors from an atomic model bundle."""
        return self.sf_processor.calculate_structure_factors(model_bundle_id)

    def compute_target(
        self, sf_bundle_id: str, exp_bundle_id: str, target_type: str = None
    ) -> str:
        """Compute the target function from structure factor and experimental data bundles."""
        return self.target_processor.calculate_target(
            sf_bundle_id, exp_bundle_id, target_type=target_type
        )

    # Add more methods as needed for workflow integration, validation, etc.
