# Geometrize: Async Geometry Minimization Workflow Example

This directory demonstrates an **asynchronous geometry minimization workflow** for macromolecular structures using CCTBX, Redis Streams, and PyTorch. It provides a practical, end-to-end example of how to set up, run, and test background geometry calculations and optimization using the agentbx framework.

## What is in this directory?

- **geometry_minimization_workflow.py**: Main async workflow script. Runs a full geometry minimization loop using an agent and a minimizer.
- **test_minimizer_consumer.py**: Test script for consumer group and forward pass.
- **test_model_manager_serialization.py**: Checks serialization/deserialization of CCTBX model managers.
- **test_refresh_functionality.py**: Demonstrates restraint refresh logic in the agent/minimizer loop.
- **README.md**: (This file) Overview and usage instructions.

## What does this workflow do?

- Loads a PDB file and creates a macromolecule bundle in Redis.
- Starts an `AsyncGeometryAgent` that listens for geometry calculation requests via Redis Streams.
- Shakes atomic coordinates (optional, for testing optimization).
- Runs a `GeometryMinimizer` that:
  - Requests geometry gradients from the agent (via Redis Streams)
  - Performs PyTorch-based optimization (Adam/SGD)
  - Sends updated coordinates back to the agent
  - Monitors convergence and gradient norms
- All communication is asynchronous and message-based (no direct function calls between agent and minimizer).

## Prerequisites

- **Redis** server running locally (default: `localhost:6379`)
- Python environment with the following packages:
  - `agentbx` (and its dependencies: CCTBX, PyTorch, numpy, etc.)
  - See the main project for installation instructions
- A PDB file for input (e.g., `input.pdb`)

## Quickstart: Running the Workflow

1. **Start Redis** (if not already running):

   ```bash
   redis-server
   ```

2. **Run the main workflow**:

   ```bash
   python geometry_minimization_workflow.py --pdbfile /path/to/your.pdb --shake-magnitude 0.5
   ```

   - This will:
     - Start the async geometry agent
     - Load your PDB as a bundle
     - Shake coordinates (optional, default magnitude 0.5)
     - Run geometry minimization (default: Adam optimizer, 100 iterations)
     - Print results and convergence info

3. **Test scripts**:

   - `test_minimizer_consumer.py`: Verifies consumer group and forward pass
   - `test_model_manager_serialization.py`: Checks bundle serialization
   - `test_refresh_functionality.py`: Shows effect of restraint refresh

   Example:

   ```bash
   python test_minimizer_consumer.py --pdbfile /path/to/your.pdb
   python test_model_manager_serialization.py
   python test_refresh_functionality.py
   ```

## Main Components

- **AsyncGeometryAgent**: Background service that listens for geometry calculation requests on a Redis stream, computes gradients using CCTBX, and responds with results.
- **GeometryMinimizer**: PyTorch-style optimizer that communicates with the agent via Redis, requests gradients, and updates coordinates.
- **MacromoleculeProcessor**: Loads PDB files and creates bundles for use in the workflow.
- **RedisManager**: Handles bundle storage and retrieval.

## How the Workflow Works

1. **System Initialization**
   - Connect to Redis
   - Start the async geometry agent (background task)
2. **Data Preparation**
   - Load a PDB file, create a macromolecule bundle
   - Optionally shake coordinates for testing
3. **Minimization Setup**
   - Create a GeometryMinimizer with stream configuration
4. **Optimization Loop**
   - Minimizer requests gradients from agent
   - Agent computes gradients and returns them
   - Minimizer updates coordinates and checks convergence

## Configuration

- Stream names, consumer groups, and other parameters are centrally configured in the workflow script (see `STREAM_CONFIG`).
- You can adjust optimizer type, learning rate, max iterations, and convergence threshold via script arguments or by editing the script.

## Troubleshooting

- **Redis connection errors**: Ensure Redis is running and accessible.
- **PDB file errors**: Check that your input PDB file exists and is valid.
- **Agent not responding**: Check logs, verify consumer group setup, and Redis health.
- **Slow performance**: Monitor Redis, check bundle sizes, and system resources.

## Further Reading

- See the main [Async Geometry Agent System Documentation](../../docs/async_geometry_agent_system.md) for architectural details.
- [Redis Streams Documentation](https://redis.io/docs/data-types/streams/)
- [CCTBX Geometry Restraints](https://cctbx.github.io/)
- [PyTorch Optimization](https://pytorch.org/docs/stable/optim.html)

## Contributing

- If you add new message types or bundle schemas, update this README.
- Add tests for new features and document new workflows.
