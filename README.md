# Agentbx: A Redis-Based Crystallographic Agent System

[![PyPI Version](https://img.shields.io/pypi/v/agentbx.svg)](https://pypi.org/project/agentbx/)
[![Python Version](https://img.shields.io/pypi/pyversions/agentbx)][pypi status]
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)][license]

[![Read the documentation at https://agentbx.readthedocs.io/](https://img.shields.io/readthedocs/agentbx/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/phzwart/agentbx/actions/workflows/tests.yml/badge.svg)][tests]
[![Coverage](https://img.shields.io/badge/coverage-120%20tests%20passing-brightgreen)][tests]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi status]: https://pypi.org/project/agentbx/
[read the docs]: https://agentbx.readthedocs.io/
[tests]: https://github.com/phzwart/agentbx/actions/workflows/tests.yml
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

## Overview

Agentbx is a Python-based system for managing crystallographic & electron microscopy computing using a Redis-backed agent architecture. It is designed for modular, distributed, and AI-integrated scientific computing, with a focus on clear separation of concerns and robust, persistent data handling.

## Key Concepts & Architecture

### Modular Separation: Agents, Clients, and Processors

- **Agents** (`src/agentbx/core/agents/`):
  - Long-running services that listen to Redis streams for requests (e.g., geometry calculation, security management).
  - Example: `AsyncGeometryAgent` processes geometry calculation requests and returns results via Redis.
- **Clients** (`src/agentbx/core/clients/`):
  - Optimizers and user-facing modules that submit requests to agents and update bundles (e.g., coordinate, B-factor, solvent optimizers).
  - Follow PyTorch conventions for optimization (separate backward/step logic).
- **Processors** (`src/agentbx/processors/`):
  - Stateless, single-responsibility modules for core scientific calculations (e.g., geometry, gradients, structure factors).

### Redis as the Central Nervous System

- **Bundles**: All data (atomic models, gradients, results) are stored as versioned bundles in Redis.
- **Streams**: Agents and clients communicate via Redis streams for robust, asynchronous, and distributed operation.

### No Internal Workflow Engine

- **Workflow orchestration is now externalized**: Instead of internal workflow management, users are encouraged to use modern orchestration tools like [Prefect](https://www.prefect.io/) or [LangGraph](https://langchain-ai.github.io/langgraph/) to coordinate multi-step pipelines and distributed jobs.
- Agentbx provides the building blocks (agents, clients, processors, bundles) for these workflows, but does not enforce or manage workflow logic internally.

## Directory Structure

```
src/agentbx/
  core/
    agents/      # Agent services (e.g., AsyncGeometryAgent, AgentSecurityManager)
    clients/     # Optimizers and user-facing modules (e.g., CoordinateOptimizer, BFactorOptimizer)
    processors/  # Stateless scientific processors (e.g., geometry, gradients)
    ...          # Redis manager, bundle base, config, etc.
```

## Example: Multi-Process Usage

1. **Start an Agent (in one shell):**
   ```bash
   python -m agentbx.core.agents.async_geometry_agent
   # or use the provided example script
   ```
2. **Run a Client Optimizer (in another shell):**
   ```bash
   python examples/optimization_clients_example.py
   # or your own script using CoordinateOptimizer, BFactorOptimizer, etc.
   ```
3. **Monitor Redis:**
   - All communication and data flow through Redis, enabling robust, distributed, and restartable computation.

## Integration with AI and External Orchestration

- **AI Models**: Easily integrate PyTorch/TensorFlow models as clients or processors.
- **Orchestration**: Use Prefect, LangGraph, or similar tools to build complex, multi-step scientific workflows using Agentbx as the computational backend.

## Features

- **Modular, single-responsibility agents and clients**
- **Persistent, versioned data bundles in Redis**
- **Stateless, testable processors for core scientific logic**
- **Seamless AI integration**
- **No internal workflow engine: bring your own orchestration**
- **Robust multi-process/multi-shell operation**

## Requirements

- **Python 3.10+**
- **Redis**
- **CCTBX**
- **Pydantic**
- **Click**
- **Poetry**
- (Optional) **PyTorch/TensorFlow** for AI integration

## Installation

You can install _Agentbx_ via [pip] from [PyPI]:

```console
$ pip install agentbx
```

Or install with Redis support:

```console
$ pip install agentbx[redis-agents]
```

## Publishing to PyPI

To publish a new version to PyPI:

1. Update the version using the sync script:
   ```bash
   python scripts/sync_version.py 1.0.4
   ```
2. Commit and push your changes.
3. Ensure all tests pass and the package builds successfully.
4. Publish to PyPI (requires credentials):
   ```bash
   poetry publish --build
   ```

## Getting Started

- See `examples/optimization_clients_example.py` for a full demonstration of agent/client interaction and optimization.
- See `whatsnext.txt` for a running development log and next steps.
- For orchestration, see Prefect or LangGraph documentation for how to build workflows using Agentbx components.

## Contributing

Contributions are welcome! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
