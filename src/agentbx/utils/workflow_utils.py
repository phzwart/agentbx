"""
Utilities for building and managing crystallographic workflows.
"""

import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional


logger = logging.getLogger(__name__)


class WorkflowManager:
    """
    Manages crystallographic workflow execution and monitoring.
    """

    def __init__(self, redis_manager: Any) -> None:
        """
        Initialize workflow manager.

        Args:
            redis_manager: Redis manager instance
        """
        self.redis_manager = redis_manager
        self.workflow_history: List[Dict[str, Any]] = []
        self.current_workflow: Optional[Dict[str, Any]] = None

    def create_workflow(self, name: str, description: str = "") -> str:
        """
        Create a new workflow and return its ID.

        Args:
            name: Workflow name
            description: Workflow description

        Returns:
            Workflow ID
        """
        workflow_id = f"workflow_{len(self.workflow_history)}_{name}"

        workflow = {
            "id": workflow_id,
            "name": name,
            "description": description,
            "status": "created",
            "steps": [],
            "input_bundles": {},
            "output_bundles": {},
            "created_at": None,
            "completed_at": None,
            "error": None,
        }

        self.current_workflow = workflow
        self.workflow_history.append(workflow)

        logger.info(f"Created workflow: {workflow_id}")
        return workflow_id

    def add_workflow_step(
        self,
        step_name: str,
        agent_class: Any,
        agent_id: str,
        input_bundle_types: List[str],
        output_bundle_types: List[str],
    ) -> None:
        """
        Add a step to the current workflow.

        Args:
            step_name: Name of the step
            agent_class: Agent class to use
            agent_id: Agent ID
            input_bundle_types: Required input bundle types
            output_bundle_types: Expected output bundle types

        Raises:
            ValueError: If no current workflow exists.
        """
        if self.current_workflow is None:
            raise ValueError("No current workflow. Call create_workflow() first.")

        step = {
            "name": step_name,
            "agent_class": agent_class,
            "agent_id": agent_id,
            "input_bundle_types": input_bundle_types,
            "output_bundle_types": output_bundle_types,
            "status": "pending",
            "input_bundle_ids": {},
            "output_bundle_ids": {},
            "error": None,
        }

        self.current_workflow["steps"].append(step)
        logger.info(f"Added workflow step: {step_name}")

    def execute_workflow(self, input_bundle_ids: Dict[str, str]) -> Dict[str, str]:
        """
        Execute the current workflow.

        Args:
            input_bundle_ids: Initial input bundle IDs

        Returns:
            Dict[str, str]: Final output bundle IDs

        Raises:
            ValueError: If no current workflow exists.
        """
        if self.current_workflow is None:
            raise ValueError("No current workflow to execute.")

        self.current_workflow["status"] = "running"
        self.current_workflow["input_bundles"] = input_bundle_ids.copy()
        current_bundle_ids = input_bundle_ids.copy()

        try:
            for i, step in enumerate(self.current_workflow["steps"]):
                logger.info(
                    f"Executing workflow step {i+1}/{len(self.current_workflow['steps'])}: {step['name']}"
                )

                step["status"] = "running"

                # Prepare input bundles for this step
                step_input_ids = {}
                for bundle_type in step["input_bundle_types"]:
                    if bundle_type in current_bundle_ids:
                        step_input_ids[bundle_type] = current_bundle_ids[bundle_type]
                    else:
                        raise ValueError(
                            f"Missing input bundle type '{bundle_type}' for step '{step['name']}'"
                        )

                step["input_bundle_ids"] = step_input_ids.copy()

                # Execute the step
                agent = step["agent_class"](self.redis_manager, step["agent_id"])
                step_output_ids = agent.run(step_input_ids)

                step["output_bundle_ids"] = step_output_ids
                step["status"] = "completed"

                # Update current bundle IDs for next steps
                current_bundle_ids.update(step_output_ids)

                logger.info(f"Completed workflow step: {step['name']}")

            self.current_workflow["status"] = "completed"
            self.current_workflow["output_bundles"] = current_bundle_ids

            logger.info(
                f"Workflow completed successfully: {self.current_workflow['id']}"
            )
            return current_bundle_ids

        except Exception as e:
            self.current_workflow["status"] = "failed"
            self.current_workflow["error"] = str(e)
            logger.error(f"Workflow failed: {e}")
            raise

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a workflow.

        Args:
            workflow_id: Workflow ID

        Returns:
            Workflow status dictionary or None if not found
        """
        for workflow in self.workflow_history:
            if workflow["id"] == workflow_id:
                return workflow
        return None

    def list_workflows(self) -> List[Dict[str, Any]]:
        """
        List all workflows.

        Returns:
            List of workflow summaries
        """
        return [
            {
                "id": w["id"],
                "name": w["name"],
                "status": w["status"],
                "n_steps": len(w["steps"]),
                "created_at": w["created_at"],
            }
            for w in self.workflow_history
        ]


def create_simple_structure_factor_workflow(
    redis_manager: Any, pdb_file: str, mtz_file: Optional[str] = None
) -> Any:
    """
    Create a simple workflow for structure factor calculation.

    Args:
        redis_manager: Redis manager instance
        pdb_file: Path to PDB file
        mtz_file: Optional path to MTZ file

    Returns:
        Workflow manager with configured workflow
    """
    from ..agents.structure_factor_agent import StructureFactorAgent
    from .crystallographic_utils import create_atomic_model_bundle

    # Create workflow manager
    workflow_mgr = WorkflowManager(redis_manager)

    # Create workflow
    workflow_mgr.create_workflow(
        name="structure_factor_calculation",
        description="Calculate structure factors from atomic model",
    )

    # Add structure factor calculation step
    workflow_mgr.add_workflow_step(
        step_name="calculate_structure_factors",
        agent_class=StructureFactorAgent,
        agent_id="sf_agent_workflow",
        input_bundle_types=["atomic_model_data"],
        output_bundle_types=["structure_factor_data"],
    )

    # Create input bundle
    atomic_bundle = create_atomic_model_bundle(pdb_file, mtz_file)
    input_bundle_id = redis_manager.store_bundle(atomic_bundle)

    return workflow_mgr, {"atomic_model_data": input_bundle_id}


def execute_structure_factor_workflow(
    redis_manager: Any, pdb_file: str, mtz_file: Optional[str] = None
) -> Dict[str, str]:
    """
    Execute a complete structure factor calculation workflow.

    Args:
        redis_manager: Redis manager instance
        pdb_file: Path to PDB file
        mtz_file: Optional path to MTZ file

    Returns:
        Dictionary of output bundle IDs
    """
    workflow_mgr, input_bundle_ids = create_simple_structure_factor_workflow(
        redis_manager, pdb_file, mtz_file
    )

    return workflow_mgr.execute_workflow(input_bundle_ids)
