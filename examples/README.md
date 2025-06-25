# Redis Structure Factor Agent Examples

This directory contains examples demonstrating the Redis manager with the StructureFactorAgent using real crystallographic data.

## Quick Start

### 1. Download Test Data

Download PDB and MTZ files for a test structure:

```bash
# Download data for ubiquitin (small, well-determined structure)
python examples/download_pdb_data.py 1ubq

# Or download data for lysozyme (classic test case)
python examples/download_pdb_data.py 1lyd
```

### 2. Inspect the Data

Check what was downloaded:

```bash
python examples/inspect_pdb_data.py
```

### 3. Run the Redis Example

```bash
python examples/redis_structure_factor_example.py
```

## Available PDB Codes

Use `--list-available` to see recommended test structures:

```bash
python examples/download_pdb_data.py --list-available
```

Recommended structures:

- `1ubq` - Ubiquitin (small, well-determined)
- `1lyd` - Lysozyme (classic test case)
- `1crn` - Crambin (small protein)
- `1hhb` - Hemoglobin (medium size)
- `1ake` - Adenylate kinase (medium size)

## Prerequisites

1. **Redis server running**

   ```bash
   redis-server
   ```

2. **CCTBX environment activated**

   ```bash
   conda activate cctbx_esm3
   ```

3. **Internet connection** (for downloading PDB data)

## Scripts Overview

### `download_pdb_data.py`

Downloads PDB and MTZ files from the Protein Data Bank.

**Usage:**

```bash
# Download specific PDB code
python examples/download_pdb_data.py 1ubq

# List available codes
python examples/download_pdb_data.py --list-available

# Overwrite existing files
python examples/download_pdb_data.py 1ubq --force
```

**Features:**

- Downloads PDB structure files
- Attempts to download associated MTZ data
- Generates synthetic MTZ data if none available
- Handles network errors gracefully
- Provides progress feedback

### `inspect_pdb_data.py`

Inspects downloaded PDB and MTZ files.

**Usage:**

```bash
python examples/inspect_pdb_data.py
```

**Shows:**

- Unit cell and space group information
- Number of atoms and chains
- Miller array details and statistics
- Data compatibility assessment

### `redis_structure_factor_example.py`

Main example demonstrating Redis integration.

**Usage:**

```bash
python examples/redis_structure_factor_example.py
```

**Demonstrates:**

- Reading real crystallographic data
- Redis bundle storage and retrieval
- Structure factor calculations
- Agent workflow execution

## File Structure
