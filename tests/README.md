# AgentBX Test Suite

This directory contains the comprehensive test suite for AgentBX.

## Quick Start

### Run All Tests with Coverage
```bash
# From the project root
python tests/run_all_tests.py
```

This will:
- Run all tests with verbose output
- Generate coverage reports (JSON, HTML, XML)
- Check per-file coverage requirements from `coverage_requirements.yaml`
- Provide a detailed summary

### Run Individual Test Files
```bash
# Run specific test file
pytest tests/core/test_base_client.py -v

# Run with coverage for specific module
pytest --cov=src/agentbx/core/base_client tests/core/test_base_client.py -v
```

## Coverage Configuration

Coverage requirements are defined in `tests/coverage_requirements.yaml`:

```yaml
# Core functionality - high coverage requirements
core:
  src/agentbx/core/base_client.py: 100    # Core client should be fully tested
  src/agentbx/core/redis_manager.py: 95   # High coverage for Redis functionality

# Schema generation - complex code, lower requirements
schemas:
  src/agentbx/schemas/generator.py: 80    # Complex generator, hard to test everything

# Main entry points
main:
  src/agentbx/__main__.py: 100            # Main entry point should be fully tested

# Overall project requirements
project:
  overall_minimum: 85                      # Minimum overall coverage for the project
  strict_mode: false                       # Whether to fail on any file below requirement

# Optional: exclude files from coverage requirements
exclude:
  - src/agentbx/__init__.py               # Simple init files
  - "*/__pycache__/*"                     # Python cache files
  - "*/tests/*"                           # Test files themselves
```

### Current Coverage Status

| File | Minimum Coverage | Current Coverage |
|------|-----------------|------------------|
| `src/agentbx/core/base_client.py` | 100% | 100% ✅ |
| `src/agentbx/core/redis_manager.py` | 95% | 98.2% ✅ |
| `src/agentbx/schemas/generator.py` | 80% | 82.1% ✅ |
| `src/agentbx/__main__.py` | 100% | 100% ✅ |
| **Overall** | **85%** | **88.2%** ✅ |

## Generated Reports

After running the test suite, you'll find:

- **`coverage.json`** - JSON format for programmatic access
- **`coverage.xml`** - XML format for CI/CD tools
- **`htmlcov/`** - Detailed HTML report (open `htmlcov/index.html`)

## Test Structure

```
tests/
├── __main__/                    # Tests for main entry point
├── core/                       # Tests for core functionality
│   ├── test_base_client.py
│   └── test_redis_manager.py
├── schemas/                    # Tests for schema generation
│   └── test_generator.py
├── run_all_tests.py           # Comprehensive test runner
├── coverage_requirements.yaml  # Coverage configuration
└── README.md                  # This file
```

## Adding New Tests

1. Create test files following the naming convention: `test_*.py`
2. Use pytest fixtures and assertions
3. Add comprehensive test cases for edge cases and error conditions
4. Update coverage requirements in `coverage_requirements.yaml` if needed

## Modifying Coverage Requirements

To change coverage requirements:

1. Edit `tests/coverage_requirements.yaml`
2. Add new files to the appropriate section (core, schemas, main)
3. Set the minimum coverage percentage
4. Optionally add files to the `exclude` list to skip coverage checking

### Configuration Options

- **`overall_minimum`**: Minimum overall project coverage
- **`strict_mode`**: If `true`, all files must meet the overall minimum (even if not explicitly listed)
- **`exclude`**: List of files/patterns to exclude from coverage checking

## Coverage Exclusions

Some lines are excluded from coverage:
- Debug/development code
- Abstract methods and protocols
- Error handling for edge cases that are hard to trigger
- Generated code sections

## CI/CD Integration

The test runner is designed to work with CI/CD pipelines:
- Exit code 0: All tests pass and coverage requirements met
- Exit code 1: Tests fail or coverage requirements not met
- XML report available for CI tools like Codecov, SonarQube, etc. 