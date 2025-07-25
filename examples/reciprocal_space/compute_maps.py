import argparse
import os

from agentbx.core.bundle_base import Bundle
from agentbx.core.config import RedisConfig
from agentbx.core.processors.experimental_data_processor import (
    ExperimentalDataProcessor,
)
from agentbx.core.processors.macromolecule_processor import MacromoleculeProcessor
from agentbx.core.processors.structure_factor_processor import StructureFactorProcessor
from agentbx.core.redis_manager import RedisManager


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute reciprocal space maps from a macromolecule."
    )
    parser.add_argument(
        "--pdbfile", type=str, required=True, help="Path to the input PDB file"
    )
    parser.add_argument(
        "--mtzfile", type=str, required=True, help="Path to the observed data MTZ file"
    )
    parser.add_argument(
        "--f-obs-label",
        type=str,
        default="FP",
        help="Label for F_obs column in MTZ (default: FP)",
    )
    parser.add_argument(
        "--sigma-label",
        type=str,
        default="SIGFP",
        help="Label for sigma column in MTZ (default: SIGFP)",
    )
    parser.add_argument(
        "--r-free-label",
        type=str,
        default="FreeR_flag",
        help="Label for R_free column in MTZ (default: FreeR_flag)",
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Full Miller array label for amplitudes/intensities+sigmas (e.g. r3ry4sf,_refln.intensity_meas,_refln.intensity_sigma)",
    )
    return parser.parse_args()


def main(pdbfile, mtzfile, f_obs_label, sigma_label, r_free_label, data, eps=0.01):
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
        pdb_path = os.path.abspath(pdbfile)
        processor = MacromoleculeProcessor(redis_manager, "reciprocal_space_processor")

        macromolecule_bundle_id = processor.create_macromolecule_bundle(pdb_path)
        # Read and store observed data (MTZ/CIF)
        mtz_path = os.path.abspath(mtzfile)
        exp_processor = ExperimentalDataProcessor(
            redis_manager, "reciprocal_space_exp_processor"
        )

        # Build data_labels dict
        if data:
            data_labels = {"data": data}
        else:
            data_labels = {
                "f_obs": f_obs_label,
                "sigmas": sigma_label,
                "r_free_flags": r_free_label,
            }
        # Create raw bundle
        raw_bundle = Bundle(bundle_type="raw_experimental_data")
        raw_bundle.add_asset("file_path", mtz_path)
        raw_bundle.add_metadata("data_type", "amplitudes")
        raw_bundle.add_metadata("data_labels", data_labels)
        raw_bundle_id = exp_processor.store_bundle(raw_bundle)
        experimental_data_bundle_id = exp_processor.run(
            {"raw_experimental_data": raw_bundle_id}
        )["experimental_data"]

        experimental_data_bundle = redis_manager.get_bundle(experimental_data_bundle_id)
        data_obs = experimental_data_bundle.get_asset("data_obs")
        d_min = data_obs.d_min()
        print(f"[Resolution] Observed data d_min: {d_min:.3f}")
        d_min_for_model = d_min - eps

        # (You can use d_min_for_model for Miller set generation later)
        # Generate xray_atomic_model_data bundle for structure factor calculation
        macromolecule_bundle = redis_manager.get_bundle(macromolecule_bundle_id)
        xray_structure = macromolecule_bundle.get_asset("xray_structure")
        xray_atomic_model_bundle_id = processor.create_xray_atomic_model_bundle(
            xray_structure, d_min_for_model
        )


if __name__ == "__main__":
    args = parse_args()
    main(
        args.pdbfile,
        args.mtzfile,
        args.f_obs_label,
        args.sigma_label,
        args.r_free_label,
        args.data,
    )
