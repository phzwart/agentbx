# src/agentbx/agents/base.py
"""
Base classes for agentbx agents.

This module provides the foundation for all agents in the agentbx system.
"""

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List

from ..core.base_client import BaseClient
from ..core.bundle_base import Bundle


class SinglePurposeAgent(BaseClient, ABC):
    """
    Base class for agents with single, clear responsibility.

    Design principles:
    1. One responsibility per agent
    2. Clear input/output bundle contracts
    3. No mixing of concerns (e.g., no targets in structure factor agents)
    4. Stateless operation (all state in bundles)
    """

    def __init__(self, redis_manager: Any, agent_id: str) -> None:
        super().__init__(redis_manager, agent_id)
        self._input_bundle_types = self.define_input_bundle_types()
        self._output_bundle_types = self.define_output_bundle_types()

    @abstractmethod
    def define_input_bundle_types(self) -> List[str]:
        """Define what bundle types this agent requires as input."""

    @abstractmethod
    def define_output_bundle_types(self) -> List[str]:
        """Define what bundle types this agent produces as output."""

    @abstractmethod
    def process_bundles(self, input_bundles: Dict[str, Bundle]) -> Dict[str, Bundle]:
        """
        Core processing logic. Transform input bundles to output bundles.

        Args:
            input_bundles: Dict mapping bundle_type -> Bundle

        Returns:
            Dict mapping output_bundle_type -> Bundle
        """

    def validate_inputs(self, input_bundles: Dict[str, Bundle]) -> bool:
        """Validate that required input bundles are present and valid."""
        for required_type in self._input_bundle_types:
            if required_type not in input_bundles:
                raise ValueError(f"Missing required input bundle: {required_type}")

            if not input_bundles[required_type].validate():
                raise ValueError(f"Invalid input bundle: {required_type}")

        return True

    def run(self, input_bundle_ids: Dict[str, str]) -> Dict[str, str]:
        """
        Main execution method.

        Args:
            input_bundle_ids: Dict mapping bundle_type -> bundle_id

        Returns:
            Dict mapping output_bundle_type -> new_bundle_id
        """
        # Load input bundles
        input_bundles = {}
        for bundle_type, bundle_id in input_bundle_ids.items():
            input_bundles[bundle_type] = self.get_bundle(bundle_id)

        # Validate inputs
        self.validate_inputs(input_bundles)

        # Process
        output_bundles = self.process_bundles(input_bundles)

        # Store outputs and return IDs
        output_ids = {}
        for bundle_type, bundle in output_bundles.items():
            bundle_id = self.store_bundle(bundle)
            output_ids[bundle_type] = bundle_id

        return output_ids
