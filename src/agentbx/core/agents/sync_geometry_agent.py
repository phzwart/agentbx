import logging
import time

import numpy as np
import torch

from agentbx.core.clients.coordinate_translator import CoordinateTranslator
from agentbx.core.processors.geometry_processor import CctbxGeometryProcessor
from agentbx.core.processors.macromolecule_processor import MacromoleculeProcessor


class SyncGeometryAgentNonClassicalPattern:
    """
    Synchronous geometry agent with a non-classical optimization pattern.
    """

    def __init__(
        self,
        redis_manager,
        macromolecule_bundle_id,
        optimizer_factory,
        optimizer_kwargs,
        scheduler_factory=None,
        scheduler_kwargs=None,
        max_iterations=100,
        convergence_threshold=1e-6,
        device=None,
        dtype=torch.float32,
    ):
        """
        Initialize the non-classical sync geometry agent.
        """
        self.logger = logging.getLogger("SyncGeometryAgent")
        self.redis_manager = redis_manager
        self.macromolecule_bundle_id = macromolecule_bundle_id
        if device is not None and str(device) != "cpu":
            self.logger.warning(
                f"Non-CPU device '{device}' requested; only CPU is supported. Using CPU."
            )
        self.device = torch.device("cpu")  # Force CPU
        self.dtype = dtype
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.coordinate_translator = CoordinateTranslator(
            redis_manager=redis_manager,
            coordinate_system="cartesian",
            requires_grad=True,
            dtype=self.dtype,
            device=self.device,
        )
        self.macromolecule_processor = MacromoleculeProcessor(
            redis_manager, "sync_agent_processor"
        )
        self.geometry_processor = CctbxGeometryProcessor(
            redis_manager, "sync_agent_geomproc"
        )
        # Load bundle and model_manager
        bundle = self.redis_manager.get_bundle(macromolecule_bundle_id)
        self.model_manager = bundle.get_asset("model_manager")
        # Ensure model_manager has restraints set up
        if self.model_manager.get_restraints_manager() is None:
            self.logger.info("Initializing model_manager with restraints...")
            self.model_manager.process(make_restraints=True)
        # Get initial coordinates as torch tensor
        sites_cart = self.model_manager.get_sites_cart()
        self.current_coordinates = (
            self.coordinate_translator.cctbx_to_torch(sites_cart)
            .detach()
            .clone()
            .requires_grad_(True)
        )
        self.optimizer = optimizer_factory(
            [self.current_coordinates], **optimizer_kwargs
        )
        self.scheduler = None
        if scheduler_factory is not None:
            if scheduler_kwargs is None:
                scheduler_kwargs = {}
            self.scheduler = scheduler_factory(self.optimizer, **scheduler_kwargs)
        self.iteration_history = []
        self.best_coordinates = None
        self.best_gradient_norm = float("inf")
        self.latest_total_energy = None
        self.energy_call_count = 0
        self.gradient_call_count = 0

    def _update_coordinates_in_bundle(self, new_coordinates):
        """
        Update coordinates in the macromolecule bundle.
        """
        try:
            coords_numpy = new_coordinates.detach().cpu().numpy()
            coords_numpy = np.ascontiguousarray(coords_numpy, dtype=np.float64)
            from cctbx.array_family import flex

            cctbx_coordinates = flex.vec3_double(coords_numpy)
            self.macromolecule_bundle_id = (
                self.macromolecule_processor.update_coordinates(
                    self.macromolecule_bundle_id, cctbx_coordinates
                )
            )
            bundle = self.redis_manager.get_bundle(self.macromolecule_bundle_id)
            self.model_manager = bundle.get_asset("model_manager")
            return self.macromolecule_bundle_id
        except Exception as e:
            self.logger.error(f"Failed to update coordinates in bundle: {e}")
            raise

    def _compute_gradients_and_energy(self):
        """
        Compute gradients and energy using the model_manager.
        """
        self.energy_call_count += 1
        self.gradient_call_count += 1
        sites_cart = self.model_manager.get_sites_cart()
        restraints_manager = self.model_manager.get_restraints_manager()
        if restraints_manager is None:
            self.logger.info("No restraints manager found, creating one...")
            self.model_manager.process(make_restraints=True)
            restraints_manager = self.model_manager.get_restraints_manager()
        geometry_restraints = restraints_manager.geometry
        call_args = {"sites_cart": sites_cart, "compute_gradients": True}
        energies_and_gradients = geometry_restraints.energies_sites(**call_args)
        gradients = energies_and_gradients.gradients
        total_energy = energies_and_gradients.target
        return gradients, total_energy

    def minimize(self):
        """
        Run the minimization loop.
        """
        self.logger.info(
            f"Starting sync geometry minimization with {self.max_iterations} max iterations"
        )
        start_time = time.time()
        gradient_norm = None
        for iteration in range(self.max_iterations):
            coords_numpy = self.current_coordinates.detach().cpu().numpy()
            coords_numpy = np.ascontiguousarray(coords_numpy, dtype=np.float64)
            from cctbx.array_family import flex

            coords_cctbx = flex.vec3_double(coords_numpy)
            self.model_manager.set_sites_cart(coords_cctbx)
            gradients_cctbx, total_energy = self._compute_gradients_and_energy()
            gradients_tensor = self.coordinate_translator.cctbx_to_torch(
                gradients_cctbx
            )
            gradient_norm = torch.norm(gradients_tensor).item()
            self.latest_total_energy = total_energy
            iteration_info = {
                "iteration": iteration,
                "gradient_norm": gradient_norm,
                "total_geometry_energy": total_energy,
                "timestamp": time.time(),
            }
            self.iteration_history.append(iteration_info)
            if gradient_norm < self.best_gradient_norm:
                self.best_gradient_norm = gradient_norm
                self.best_coordinates = self.current_coordinates.clone()
            self.logger.info(
                f"Iteration {iteration}: gradient_norm = {gradient_norm:.6f}"
            )
            if gradient_norm < self.convergence_threshold:
                self.logger.info(f"Converged at iteration {iteration}")
                break

            def closure():
                self.optimizer.zero_grad()
                coords_numpy = self.current_coordinates.detach().cpu().numpy()
                coords_numpy = np.ascontiguousarray(coords_numpy, dtype=np.float64)
                from cctbx.array_family import flex

                coords_cctbx = flex.vec3_double(coords_numpy)
                self.model_manager.set_sites_cart(coords_cctbx)
                gradients_cctbx, total_energy = self._compute_gradients_and_energy()
                gradients_tensor = self.coordinate_translator.cctbx_to_torch(
                    gradients_cctbx
                )
                self.current_coordinates.grad = gradients_tensor
                return torch.tensor(total_energy, dtype=self.dtype, device=self.device)

            if isinstance(self.optimizer, torch.optim.LBFGS):
                self.optimizer.step(closure)
            else:
                self.optimizer.zero_grad()
                self.current_coordinates.grad = gradients_tensor
                self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
        if self.best_coordinates is not None:
            self.current_coordinates = self.best_coordinates
        self._update_coordinates_in_bundle(self.current_coordinates.detach())
        results = {
            "converged": gradient_norm is not None
            and gradient_norm < self.convergence_threshold,
            "final_gradient_norm": (
                gradient_norm if gradient_norm is not None else float("nan")
            ),
            "best_gradient_norm": self.best_gradient_norm,
            "iterations": len(self.iteration_history),
            "total_time": time.time() - start_time,
            "final_bundle_id": self.macromolecule_bundle_id,
            "iteration_history": self.iteration_history,
            "final_total_geometry_energy": self.latest_total_energy,
            "optimizer_type": type(self.optimizer).__name__,
            "energy_call_count": self.energy_call_count,
            "gradient_call_count": self.gradient_call_count,
        }
        return results

    def get_best_coordinates(self):
        """
        Get the best coordinates found during minimization.
        """
        if self.best_coordinates is not None:
            return self.best_coordinates
        else:
            return self.current_coordinates


class SyncGeometryAgent:
    """
    Classic PyTorch-style synchronous geometry agent with explicit forward/backward methods.

    .forward() computes and returns only the energy (loss) as a scalar tensor.
    .backward() computes and sets gradients in self.current_coordinates.grad (using CCTBX),
    or uses cached gradients from the last .forward() if available.
    """

    def __init__(
        self,
        redis_manager,
        macromolecule_bundle_id,
        optimizer_factory,
        optimizer_kwargs,
        scheduler_factory=None,
        scheduler_kwargs=None,
        max_iterations=100,
        convergence_threshold=1e-6,
        device=None,
        dtype=torch.float32,
    ):
        """
        Initialize the classic sync geometry agent.
        """
        self.logger = logging.getLogger("SyncGeometryAgent")
        self.redis_manager = redis_manager
        self.macromolecule_bundle_id = macromolecule_bundle_id
        if device is not None and str(device) != "cpu":
            self.logger.warning(
                f"Non-CPU device '{device}' requested; only CPU is supported. Using CPU."
            )
        self.device = torch.device("cpu")  # Force CPU
        self.dtype = dtype
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.coordinate_translator = CoordinateTranslator(
            redis_manager=redis_manager,
            coordinate_system="cartesian",
            requires_grad=True,
            dtype=self.dtype,
            device=self.device,
        )
        self.macromolecule_processor = MacromoleculeProcessor(
            redis_manager, "sync_agent_processor"
        )
        # Load bundle and model_manager
        bundle = self.redis_manager.get_bundle(macromolecule_bundle_id)
        self.model_manager = bundle.get_asset("model_manager")
        # Ensure model_manager has restraints set up
        if self.model_manager.get_restraints_manager() is None:
            self.logger.info("Initializing model_manager with restraints...")
            self.model_manager.process(make_restraints=True)
        # Get initial coordinates as torch tensor
        sites_cart = self.model_manager.get_sites_cart()
        self.current_coordinates = (
            self.coordinate_translator.cctbx_to_torch(sites_cart)
            .detach()
            .clone()
            .requires_grad_(True)
        )
        self.optimizer = optimizer_factory(
            [self.current_coordinates], **optimizer_kwargs
        )
        self.scheduler = None
        if scheduler_factory is not None:
            if scheduler_kwargs is None:
                scheduler_kwargs = {}
            self.scheduler = scheduler_factory(self.optimizer, **scheduler_kwargs)
        self.iteration_history = []
        self.best_coordinates = None
        self.best_gradient_norm = float("inf")
        self.latest_total_energy = None
        self.energy_call_count = 0
        self.gradient_call_count = 0
        self._last_gradients = None
        self._last_energy = None

    def forward(self):
        """
        Compute and return the energy (loss) as a scalar tensor for the current coordinates.

        Also caches the gradients and energy for use in backward().
        """
        self._last_gradients = None
        self._last_energy = None
        self.energy_call_count += 1
        coords_numpy = self.current_coordinates.detach().cpu().numpy()
        coords_numpy = np.ascontiguousarray(coords_numpy, dtype=np.float64)
        from cctbx.array_family import flex

        coords_cctbx = flex.vec3_double(coords_numpy)
        self.model_manager.set_sites_cart(coords_cctbx)
        if self.model_manager.get_restraints_manager() is None:
            self.model_manager.process(make_restraints=True)
        restraints_manager = self.model_manager.get_restraints_manager()
        geometry_restraints = restraints_manager.geometry
        call_args = {"sites_cart": coords_cctbx, "compute_gradients": True}
        energies_and_gradients = geometry_restraints.energies_sites(**call_args)
        gradients_cctbx = energies_and_gradients.gradients
        total_energy = energies_and_gradients.target
        gradients_tensor = self.coordinate_translator.cctbx_to_torch(gradients_cctbx)
        self._last_gradients = gradients_tensor
        self._last_energy = total_energy
        return torch.tensor(total_energy, dtype=self.dtype, device=self.device)

    def backward(self):
        """
        Compute and set gradients for current coordinates (using CCTBX).

        Uses cached gradients from the last .forward() if available, otherwise recomputes.
        """
        self.gradient_call_count += 1
        if self._last_gradients is not None:
            self.current_coordinates.grad = self._last_gradients
        else:
            coords_numpy = self.current_coordinates.detach().cpu().numpy()
            coords_numpy = np.ascontiguousarray(coords_numpy, dtype=np.float64)
            from cctbx.array_family import flex

            coords_cctbx = flex.vec3_double(coords_numpy)
            self.model_manager.set_sites_cart(coords_cctbx)
            if self.model_manager.get_restraints_manager() is None:
                self.model_manager.process(make_restraints=True)
            restraints_manager = self.model_manager.get_restraints_manager()
            geometry_restraints = restraints_manager.geometry
            call_args = {"sites_cart": coords_cctbx, "compute_gradients": True}
            energies_and_gradients = geometry_restraints.energies_sites(**call_args)
            gradients_cctbx = energies_and_gradients.gradients
            gradients_tensor = self.coordinate_translator.cctbx_to_torch(
                gradients_cctbx
            )
            self.current_coordinates.grad = gradients_tensor
            self._last_gradients = gradients_tensor
            self._last_energy = energies_and_gradients.target

    def minimize(self):
        """
        Run the minimization loop. Uses closure for LBFGS, otherwise explicit forward/backward/step.

        Returns a dict with minimization results.
        """
        self.logger.info(
            f"Starting classic sync geometry minimization with {self.max_iterations} max iterations"
        )
        start_time = time.time()
        gradient_norm = None
        for iteration in range(self.max_iterations):
            if isinstance(self.optimizer, torch.optim.LBFGS):

                def closure():
                    self.optimizer.zero_grad()
                    loss = self.forward()
                    self.backward()
                    return loss

                self.optimizer.step(closure)
                gradients = self._last_gradients
                total_energy = self._last_energy
            else:
                # --- External gradient engine pattern ---
                # We are not using PyTorch autograd for backward(). Instead, we compute
                # gradients externally (CCTBX), set self.current_coordinates.grad manually
                # in self.backward(), and then call optimizer.step().
                # This is the correct pattern for optimizers like Adam/SGD when using
                # external gradients.
                self.optimizer.zero_grad()
                total_energy = self.forward()
                self.backward()
                gradients = self._last_gradients
                self.optimizer.step()
            gradient_norm = (
                torch.norm(gradients).item() if gradients is not None else float("nan")
            )
            self.latest_total_energy = total_energy
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info(
                f"Iteration {iteration}: f = {total_energy:.6f} ; |g| = {gradient_norm:.6f} ; lr = {current_lr:.6g}"
            )
            iteration_info = {
                "iteration": iteration,
                "gradient_norm": gradient_norm,
                "total_geometry_energy": total_energy,
                "learning_rate": current_lr,
                "timestamp": time.time(),
            }
            self.iteration_history.append(iteration_info)
            if gradient_norm < self.best_gradient_norm:
                self.best_gradient_norm = gradient_norm
                self.best_coordinates = self.current_coordinates.clone()
            if gradient_norm < self.convergence_threshold:
                self.logger.info(f"Converged at iteration {iteration}")
                break
            if self.scheduler is not None:
                self.scheduler.step()
        if self.best_coordinates is not None:
            self.current_coordinates = self.best_coordinates
        self._update_coordinates_in_bundle(self.current_coordinates.detach())
        results = {
            "converged": gradient_norm is not None
            and gradient_norm < self.convergence_threshold,
            "final_gradient_norm": (
                gradient_norm if gradient_norm is not None else float("nan")
            ),
            "best_gradient_norm": self.best_gradient_norm,
            "iterations": len(self.iteration_history),
            "total_time": time.time() - start_time,
            "final_bundle_id": self.macromolecule_bundle_id,
            "iteration_history": self.iteration_history,
            "final_total_geometry_energy": self.latest_total_energy,
            "optimizer_type": type(self.optimizer).__name__,
            "energy_call_count": self.energy_call_count,
            "gradient_call_count": self.gradient_call_count,
        }
        return results

    def _update_coordinates_in_bundle(self, new_coordinates):
        """
        Update coordinates in the macromolecule bundle.
        """
        try:
            coords_numpy = new_coordinates.detach().cpu().numpy()
            coords_numpy = np.ascontiguousarray(coords_numpy, dtype=np.float64)
            from cctbx.array_family import flex

            cctbx_coordinates = flex.vec3_double(coords_numpy)
            self.macromolecule_bundle_id = (
                self.macromolecule_processor.update_coordinates(
                    self.macromolecule_bundle_id, cctbx_coordinates
                )
            )
            bundle = self.redis_manager.get_bundle(self.macromolecule_bundle_id)
            self.model_manager = bundle.get_asset("model_manager")
            return self.macromolecule_bundle_id
        except Exception as e:
            self.logger.error(f"Failed to update coordinates in bundle: {e}")
            raise

    def get_best_coordinates(self):
        """
        Get the best coordinates found during minimization.
        """
        if self.best_coordinates is not None:
            return self.best_coordinates
        else:
            return self.current_coordinates
