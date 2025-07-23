# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Clear separation of **agents** (`src/agentbx/core/agents/`), **clients** (`src/agentbx/core/clients/`), and **processors** (`src/agentbx/processors/`).
- Abstract optimization client system with subclasses for coordinate, B-factor, and solvent parameter optimization.
- Example scripts for multi-process agent/client operation and development log (`whatsnext.txt`).

### Changed

- Refactored codebase for modularity and maintainability.
- Updated all imports and examples to use the new agent/client/processor structure.
- Updated documentation and README to reflect new architecture and usage patterns.

### Removed

- Internal workflow orchestration logic. Users are now encouraged to use external tools (e.g., Prefect, LangGraph) for workflow management.

### Fixed

- Syntax and import errors from previous refactorings.
- Improved robustness for multi-process/multi-shell operation.

## [1.0.4] - 2025-06-25

### Added

- Initial project setup with Redis-based agent architecture
- Centralized version management system
- Automated PyPI publishing via GitHub Actions

### Changed

- None

### Deprecated

- None

### Removed

- None

### Fixed

- None

### Security

- None

## [1.0.3] - 2024-06-26

### Added

- Centralized version management system with `scripts/sync_version.py`
- Automated PyPI publishing via GitHub Actions
- Comprehensive CI/CD pipeline with debugging
- Manual release workflow for testing

### Fixed

- Metadata validation issues in PyPI publishing
- Version synchronization across all files
- Twine upload configuration for reliable publishing

## [1.0.2] - 2024-06-26

### Fixed

- Version mismatch between `pyproject.toml` and `__init__.py`
- PyPI metadata generation issues

## [1.0.1] - 2024-06-26

### Fixed

- Initial version sync issues

## [1.0.0] - 2024-06-26

### Added

- Initial release of Agentbx
- Redis-based crystallographic agent system
- Structure factor calculation agents
- Target function computation
- Gradient calculation capabilities
- Experimental data processing
- Bundle system for data management
- CLI tools for workflow management
