#!/usr/bin/env python3
"""Sync version across all files."""

import re
import sys
from pathlib import Path


def update_version_file(version: str) -> None:
    """Update the central version file."""
    version_file = Path("src/agentbx/_version.py")
    content = f'''"""Version information for agentbx."""

__version__ = "{version}"
'''
    version_file.write_text(content)
    print(f"Updated {version_file}")


def update_pyproject_toml(version: str) -> None:
    """Update pyproject.toml version."""
    pyproject_file = Path("pyproject.toml")
    content = pyproject_file.read_text()

    # Replace version line
    pattern = r'version = "[^"]*"'
    replacement = f'version = "{version}"'
    new_content = re.sub(pattern, replacement, content)

    pyproject_file.write_text(new_content)
    print(f"Updated {pyproject_file}")


def main() -> None:
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python scripts/sync_version.py <version>")
        print("Example: python scripts/sync_version.py 1.0.1")
        sys.exit(1)

    version = sys.argv[1]

    # Validate version format
    if not re.match(r"^\d+\.\d+\.\d+", version):
        print("Error: Version must be in format X.Y.Z")
        sys.exit(1)

    print(f"Syncing version {version} across all files...")

    update_version_file(version)
    update_pyproject_toml(version)

    print("Version sync complete!")


if __name__ == "__main__":
    main()
