import argparse
import logging
import os

import torch

from agentbx.core.agents.sync_geometry_agent import SyncGeometryAgent
from agentbx.core.config import RedisConfig
from agentbx.core.processors.macromolecule_processor import MacromoleculeProcessor
from agentbx.core.redis_manager import RedisManager
from agentbx.utils.structures.coordinate_shaker import shake_coordinates_in_bundle


def parse_args():
    """
    This function parses the command line arguments.
    """
    parser = argparse.ArgumentParser(description="Sync geometry minimization workflow")
    parser.add_argument(
        "--pdbfile", type=str, required=True, help="Path to the input PDB file"
    )
    parser.add_argument(
        "--shake-magnitude",
        type=float,
        default=0.2,
        help="Magnitude for coordinate shaking (default: 0.2)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["lbfgs", "adam"],
        default="lbfgs",
        help="Optimizer to use: 'lbfgs' or 'adam' (default: lbfgs)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["cyclic", "step", "none"],
        default="cyclic",
        help="Learning rate scheduler to use with Adam: 'cyclic', 'step', or 'none' (default: cyclic)",
    )
    return parser.parse_args()


def main(pdbfile, shake_magnitude, optimizer_choice, scheduler_choice):
    """
    This function is the main entry point for the sync geometry workflow.

    It parses the command line arguments, sets up the Redis manager, and runs the sync geometry agent.
    """
    print(
        f"Starting sync geometry workflow with optimizer: {optimizer_choice} and scheduler: {scheduler_choice}"
    )
    logging.basicConfig(level=logging.INFO)
    redis_config = RedisConfig.from_env()
    redis_manager = RedisManager(
        host=redis_config.host,
        port=redis_config.port,
        db=redis_config.db,
        password=redis_config.password,
        max_connections=redis_config.max_connections,
        socket_timeout=redis_config.socket_timeout,
        socket_connect_timeout=redis_config.socket_connect_timeout,
        retry_on_timeout=redis_config.retry_on_timeout,
        health_check_interval=redis_config.health_check_interval,
        default_ttl=redis_config.default_ttl,
    )
    with redis_manager:
        # 1. Read in a PDB file and load as a macromolecule bundle
        pdb_path = os.path.abspath(pdbfile)
        processor = MacromoleculeProcessor(redis_manager, "sync_workflow_processor")
        macromolecule_bundle_id = processor.create_macromolecule_bundle(pdb_path)
        # 2. Optionally shake coordinates
        macromolecule_bundle = redis_manager.get_bundle(macromolecule_bundle_id)
        shaken_bundle = shake_coordinates_in_bundle(
            macromolecule_bundle, magnitude=shake_magnitude
        )
        shaken_bundle_id = redis_manager.store_bundle(shaken_bundle)
        # 3. Choose optimizer and scheduler
        if optimizer_choice == "lbfgs":
            max_iterations = 200
            optimizer_factory = torch.optim.LBFGS
            optimizer_kwargs = {
                "lr": 1.0,
                "max_iter": 20,
                "history_size": 20,
                "line_search_fn": "strong_wolfe",
            }
            scheduler_factory = None
            scheduler_kwargs = None
        else:
            max_iterations = 4000
            optimizer_factory = torch.optim.Adam
            optimizer_kwargs = {
                "lr": 0.01,
            }
            if scheduler_choice == "cyclic":
                scheduler_factory = torch.optim.lr_scheduler.CyclicLR
                scheduler_kwargs = {
                    "base_lr": 0.001,
                    "max_lr": 0.01,
                    "step_size_up": 15,
                    "step_size_down": 15,
                    "mode": "triangular",
                    "cycle_momentum": False,
                }
            elif scheduler_choice == "step":
                scheduler_factory = torch.optim.lr_scheduler.StepLR
                scheduler_kwargs = {
                    "step_size": 100,
                    "gamma": 0.5,
                }
            else:
                scheduler_factory = None
                scheduler_kwargs = None
        print(
            f"Using optimizer: {optimizer_choice.upper()} (scheduler: {scheduler_choice})"
        )
        agent = SyncGeometryAgent(
            redis_manager=redis_manager,
            macromolecule_bundle_id=shaken_bundle_id,
            optimizer_factory=optimizer_factory,
            optimizer_kwargs=optimizer_kwargs,
            max_iterations=max_iterations,
            convergence_threshold=1e-6,
            scheduler_factory=scheduler_factory,
            scheduler_kwargs=scheduler_kwargs,
        )
        results = agent.minimize()
        print(f"Final total geometry energy: {results['final_total_geometry_energy']}")
        print(f"Final gradient norm: {results['final_gradient_norm']}")
        print(f"Energy call count: {results['energy_call_count']}")


if __name__ == "__main__":
    args = parse_args()
    main(args.pdbfile, args.shake_magnitude, args.optimizer, args.scheduler)
