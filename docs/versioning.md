# Version Management

This project uses a centralized versioning system to ensure consistency across all files.

## How It Works

The version is stored in a single source of truth: `src/agentbx/_version.py`

```python
"""Version information for agentbx."""

__version__ = "1.0.4"
```

All other files import the version from this central location:

- `src/agentbx/__init__.py` imports `__version__` from `_version.py`
- `pyproject.toml` is kept in sync using the sync script
- GitHub Actions workflows use the sync script to ensure consistency

## Updating the Version

### Using the Sync Script (Recommended)

```bash
# Update to a new version
python scripts/sync_version.py 1.0.4

# This will update:
# - src/agentbx/_version.py
# - pyproject.toml
```

### Manual Update

If you need to update manually:

1. Update `src/agentbx/_version.py`
2. Update `pyproject.toml`
3. Ensure both files have the same version

## Version Format

Versions must follow semantic versioning: `X.Y.Z`

- `X` - Major version (breaking changes)
- `Y` - Minor version (new features, backward compatible)
- `Z` - Patch version (bug fixes, backward compatible)

## CI/CD Integration

The GitHub Actions workflows automatically:

1. Check if the current version has been tagged
2. Sync the version across all files
3. Build and publish the package to PyPI
4. Create a Git tag for the release

## Benefits

- **Single source of truth**: No more version mismatches
- **Automatic sync**: CI/CD ensures consistency
- **Easy updates**: One command updates everything
- **Reliable releases**: No metadata errors from version mismatches
