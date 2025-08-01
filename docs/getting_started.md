# Getting Started

## Installation

Install agentbx using pip:

```bash
pip install agentbx
```

Or install with Redis support:

```bash
pip install agentbx[redis-agents]
```

For development installation:

```bash
git clone https://github.com/phzwart/agentbx.git
cd agentbx
poetry install
```

## Quick Start

### Basic Usage

```python
from agentbx.core.redis_manager import RedisManager
from agentbx.agents.structure_factor_agent import StructureFactorProcessor
from agentbx.schemas.generated import XrayAtomicModelDataBundle

# Initialize Redis manager
redis_manager = RedisManager(host="localhost", port=6379)

# Create a structure factor processor
processor = StructureFactorProcessor(redis_manager, "sf_processor_001")

# Create a bundle
bundle = XrayAtomicModelDataBundle(
    xray_structure=your_structure,
    miller_indices=your_indices
)

# Your processor is ready to process crystallographic data!
```

### Command Line Interface

```bash
# Get help
agentbx --help

# Workflow with PDB and MTZ files
agentbx workflow examples/input.pdb examples/input.mtz

# Analyze crystallographic files
agentbx analyze examples/input.pdb
```

## Core Concepts

### Processors

Processors are single-purpose components that handle specific crystallographic tasks:

- **ExperimentalDataProcessor**: Handles experimental data processing
- **StructureFactorProcessor**: Calculates structure factors from atomic models
- **TargetProcessor**: Computes target functions for refinement
- **GradientProcessor**: Calculates gradients for optimization

### Bundles

Bundles are data containers that hold related crystallographic information:

```python
from agentbx.schemas.generated import XrayAtomicModelDataBundle

# Create a bundle for atomic model data
bundle = XrayAtomicModelDataBundle(
    xray_structure=your_structure,
    miller_indices=your_indices
)
```

### Redis Integration

Redis provides persistent storage and caching:

```python
# Store a bundle
bundle_id = redis_manager.store_bundle(bundle)

# Retrieve a bundle
retrieved_bundle = redis_manager.get_bundle(bundle_id)
```

## Next Steps

- Read the [Usage Guide](usage.md) for detailed examples
- Check the [API Reference](reference.md) for complete documentation
- Explore [Redis Integration](redis_integration.md) for advanced features
