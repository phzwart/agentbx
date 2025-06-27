# Agentbx: A Redis-Based Crystallographic Agent System

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

Agentbx is a Python-based system for managing crystallographic & electron microscopy computing workflows using a Redis-backed agent architecture. It's designed to handle complex calculations using best-in-class algorithms (like structure factor calculation) through modular, single-responsibility agents that communicate via persistent data bundles.

## Vision & Motivation

A key motivation for this architecture is to enable **seamless integration between AI models and traditional crystallographic/EM toolkits**. By using Redis as middleware and implementing clear separation of concerns, Agentbx allows:

- **Technology Stack Independence**: AI models (PyTorch, TensorFlow, etc.) can run independently from crystallographic toolkits (CCTBX, Phenix, etc.)
- **Event-Driven Architecture**: Agents react to status messages placed on Redis queues, enabling asynchronous processing
- **Modular Integration**: New AI models or crystallographic tools can be added as agents without modifying existing code
- **Scalable Workflows**: Complex pipelines can be orchestrated across different computational resources

This design enables researchers to combine cutting-edge AI predictions (protein structure prediction, electron density interpretation) with established crystallographic refinement workflows, all coordinated through Redis as the central nervous system.

## Core Architecture

### 1. **Processor System**

- **SinglePurposeProcessor**: Base class for processors with one clear responsibility
- **StructureFactorProcessor**: Calculates structure factors from atomic models
- **TargetProcessor**: Computes target functions for refinement
- **GradientProcessor**: Calculates gradients for optimization
- **ExperimentalDataProcessor**: Handles experimental data processing
- **AI Model Processors**: Future processors for protein structure prediction, density interpretation, etc.

### 2. **Redis Manager**

- **Persistent Storage**: Stores data bundles with TTL and metadata
- **Message Queues**: Status messages and workflow coordination
- **Serialization**: Handles complex CCTBX objects via pickle/JSON
- **Connection Pooling**: Robust Redis connections with health monitoring
- **Caching**: Built-in caching for expensive computations

### 3. **Bundle System**

- **Data Containers**: Bundles hold related crystallographic data assets
- **Metadata**: Creation time, checksums, provenance information
- **Validation**: Built-in validation for bundle contents
- **Type Safety**: Strong typing for different bundle types

### 4. **Utility Framework**

- **File Handling**: PDB/MTZ file reading and validation
- **Data Analysis**: Complex number analysis, miller array statistics
- **Workflow Management**: Multi-step workflow orchestration
- **CLI Tools**: Command-line interface for common operations

## Key Features

### **Modular Design**

Each processor does ONE thing well:

- StructureFactorProcessor: Only calculates structure factors
- TargetProcessor: Only computes target functions
- AI Model Processors: Only handle AI predictions
- No mixing of concerns between processors

### **Technology Stack Separation**

- **AI Models**: Can run on GPU clusters with PyTorch/TensorFlow
- **Crystallographic Tools**: Run on CPU clusters with CCTBX/Phenix
- **Redis Middleware**: Coordinates between different computational resources
- **Event-Driven**: Processors react to status messages, enabling asynchronous processing

### **Persistent Data Flow**

- Redis stores all intermediate results
- Processors can be restarted without losing data
- Data persists between workflow runs
- Unique bundle IDs for tracking

### **Crystallographic Integration**

- Native CCTBX support for crystallographic calculations
- Handles real PDB/MTZ files
- Structure factor calculations with bulk solvent
- R-factor computation and validation

### **Scalable Architecture**

- Redis enables distributed processing
- Multiple processors can work on same data
- Easy to add new processor types
- Workflow orchestration for complex pipelines

## Example Workflow

1. **Download Data**: PDB structure + MTZ reflections
2. **Create Bundle**: Package atomic model data
3. **Store in Redis**: Persistent storage with metadata
4. **Run Processor**: StructureFactorProcessor processes data
5. **Store Results**: Structure factors saved back to Redis
6. **Analyze**: Comprehensive data analysis and statistics

## Future AI Integration Workflows

### **AI-Assisted Structure Determination**

1. **AI Model Processor**: Predicts initial protein structure from sequence
2. **StructureFactorProcessor**: Calculates structure factors from AI prediction
3. **TargetProcessor**: Computes target function comparing to experimental data
4. **RefinementProcessor**: Optimizes structure using gradients
5. **ValidationProcessor**: Assesses structure quality

### **AI-Enhanced Electron Density Interpretation**

1. **ExperimentalDataProcessor**: Processes experimental data
2. **AI Density Processor**: Interprets electron density using deep learning
3. **StructureFactorProcessor**: Calculates structure factors
4. **ComparisonProcessor**: Validates AI interpretation against experimental data

## Use Cases

### **Crystallographic Refinement**

- Structure factor calculation from atomic models
- Target function computation for refinement
- Gradient calculation for optimization
- Experimental data integration

### **AI-Crystallography Integration**

- Protein structure prediction integration
- Electron density interpretation with AI
- Automated structure validation
- AI-assisted model building

### **Data Pipeline Management**

- Persistent storage of intermediate results
- Workflow orchestration for complex calculations
- Data validation and quality control
- Reproducible research workflows

### **Distributed Computing**

- Multiple processors working on same dataset
- Redis as central data store
- Scalable architecture for large datasets
- Fault tolerance and recovery

## Requirements

- **Python 3.10+**: Core language
- **Redis**: Data persistence, caching, and message queues
- **CCTBX**: Crystallographic calculations
- **Pydantic**: Data validation and serialization
- **Click**: Command-line interface
- **Poetry**: Dependency management
- **Future**: PyTorch/TensorFlow for AI model processors

## Installation

You can install _Agentbx_ via [pip] from [PyPI]:

```console
$ pip install agentbx
```

Or install with Redis support:

```console
$ pip install agentbx[redis-agents]
```
