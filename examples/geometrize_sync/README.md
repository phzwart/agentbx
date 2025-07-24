# Synchronous Geometry Minimization Workflow (`geometrize_sync`)

This directory contains an example workflow for **synchronous geometry minimization** using the `SyncGeometryAgent`.

## Overview

- The `SyncGeometryAgent` performs geometry minimization using a classic PyTorch-style optimization loop, but with gradients and energies computed externally (via CCTBX), not by PyTorch autograd.
- All calculations (energy and gradients) are performed **inside the agent**—there is no asynchronous message passing or distributed computation.
- **Redis is still used** for bundle/file communication (input/output of macromolecule data, coordinate updates, etc.), but not for agent-to-agent messaging.

## How is this different from the async agent?

- **Async agent**: Uses Redis streams for message passing between separate processes (e.g., agent and minimizer communicate asynchronously).
- **Sync agent (this workflow)**: All computation is done in-process, in a single Python process. Redis is only used for data storage and retrieval, not for workflow coordination.
- The synchronous agent can still be run in an asynchronous context if needed, but by default, it is a classic, blocking optimization loop.

## External Gradient Pattern

- `.forward()` computes and returns only the energy (loss) as a scalar tensor.
- `.backward()` computes and sets gradients in `.grad` using the external CCTBX engine.
- The optimizer loop (for Adam/SGD) is:
  1. `optimizer.zero_grad()`
  2. `loss = agent.forward()`
  3. `agent.backward()`
  4. `optimizer.step()`
- For LBFGS, the closure uses the same split: computes loss in `.forward()`, sets gradients in `.backward()`, returns loss.
- This pattern is as close as possible to standard PyTorch, but adapted for external (non-autograd) gradient engines.

## How to Run

Activate your environment (e.g., `conda activate cctbx`), then run:

```bash
python sync_geometry_workflow.py --pdbfile <path/to/your.pdb> [--optimizer adam|lbfgs] [--scheduler cyclic|step|none] [--shake-magnitude 0.2]
```

### Command-line options

- `--pdbfile`: Path to the input PDB file (required)
- `--optimizer`: Choose between `adam` and `lbfgs` (default: `lbfgs`)
- `--scheduler`: Learning rate scheduler for Adam (`cyclic`, `step`, or `none`; default: `cyclic`)
- `--shake-magnitude`: Magnitude for coordinate shaking (default: 0.2)

### Example

Run with Adam and a step scheduler:

```bash
python sync_geometry_workflow.py --pdbfile ../data/small.pdb --optimizer adam --scheduler step
```

Run with LBFGS (default):

```bash
python sync_geometry_workflow.py --pdbfile ../data/small.pdb
```

## Requirements

- Python 3.8+
- PyTorch
- Redis server running and accessible
- CCTBX and all dependencies installed (see project setup)
- Activate your environment with `conda activate cctbx` before running

## Notes

- The synchronous agent is suitable for local, in-process optimization, but can be integrated into larger async or distributed workflows if needed.
- All data exchange (input/output) is still handled via Redis bundles for compatibility with the rest of the system.
- The agent logs energy, gradient norm, and learning rate at each iteration, and supports learning rate schedulers.

---

For more details, see the code in `sync_geometry_workflow.py` and `sync_geometry_agent.py`.
