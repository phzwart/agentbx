#!/usr/bin/env python3
"""Update changelog for new releases."""

import re
import sys
from datetime import datetime
from pathlib import Path


def update_changelog(version: str, release_date: str = None) -> None:
    """Update the changelog for a new release."""
    if release_date is None:
        release_date = datetime.now().strftime("%Y-%m-%d")

    changelog_file = Path("CHANGELOG.md")
    content = changelog_file.read_text()

    # Replace [Unreleased] with the new version
    pattern = r"## \[Unreleased\]"
    replacement = f"## [Unreleased]\n\n## [{version}] - {release_date}"
    new_content = re.sub(pattern, replacement, content)

    changelog_file.write_text(new_content)
    print(f"Updated CHANGELOG.md for version {version}")


def main() -> None:
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/update_changelog.py <version> [release_date]")
        print("Example: python scripts/update_changelog.py 1.0.4")
        print("Example: python scripts/update_changelog.py 1.0.4 2024-06-27")
        sys.exit(1)

    version = sys.argv[1]
    release_date = sys.argv[2] if len(sys.argv) > 2 else None

    # Validate version format
    if not re.match(r"^\d+\.\d+\.\d+", version):
        print("Error: Version must be in format X.Y.Z")
        sys.exit(1)

    print(f"Updating changelog for version {version}...")
    update_changelog(version, release_date)
    print("Changelog update complete!")


if __name__ == "__main__":
    main()
