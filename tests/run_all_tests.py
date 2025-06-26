#!/usr/bin/env python3
"""
Comprehensive test runner for agentbx.

Runs all tests, generates coverage reports, and checks per-file coverage goals.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def load_coverage_requirements():
    """Load coverage requirements from YAML file."""
    config_file = Path(__file__).parent / "coverage_requirements.yaml"

    if not config_file.exists():
        print(f"❌ Error: Coverage requirements file not found: {config_file}")
        sys.exit(1)

    try:
        with open(config_file) as f:
            config = yaml.safe_load(f)

        # Flatten the requirements into a single dict
        requirements = {}

        # Add core requirements
        if "core" in config:
            requirements.update(config["core"])

        # Add schema requirements
        if "schemas" in config:
            requirements.update(config["schemas"])

        # Add main requirements
        if "main" in config:
            requirements.update(config["main"])

        # Store project settings
        project_settings = config.get("project", {})
        exclude_patterns = config.get("exclude", [])

        return requirements, project_settings, exclude_patterns

    except yaml.YAMLError as e:
        print(f"❌ Error parsing YAML file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading coverage requirements: {e}")
        sys.exit(1)


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def should_check_file(file_path, exclude_patterns):
    """Check if a file should be included in coverage checking."""
    for pattern in exclude_patterns:
        if pattern.endswith("/*"):
            # Directory pattern
            dir_pattern = pattern[:-2]
            if dir_pattern in file_path:
                return False
        elif pattern in file_path:
            return False
    return True


def check_per_file_coverage():
    """Check if each file meets its minimum coverage requirement."""

    # Load requirements from YAML
    per_file_minimums, project_settings, exclude_patterns = load_coverage_requirements()
    overall_minimum = project_settings.get("overall_minimum", 0)
    strict_mode = project_settings.get("strict_mode", False)

    # Check if coverage.json exists
    coverage_file = Path("coverage.json")
    if not coverage_file.exists():
        print("❌ Error: coverage.json not found.")
        return False

    # Load coverage data
    with open(coverage_file) as f:
        data = json.load(f)

    failed_files = []
    passed_files = []
    checked_files = []

    print(f"\n{'='*60}")
    print("📊 PER-FILE COVERAGE CHECK")
    print(f"{'='*60}")

    # Check each file against its minimum
    for file_path, file_data in data["files"].items():
        # Convert to relative path for comparison
        rel_path = file_path.replace(str(Path.cwd()) + "/", "")

        # Skip excluded files
        if not should_check_file(rel_path, exclude_patterns):
            print(f"⏭️  SKIP: {rel_path} (excluded)")
            continue

        checked_files.append(rel_path)
        percent_covered = file_data["summary"]["percent_covered"]

        if rel_path in per_file_minimums:
            min_required = per_file_minimums[rel_path]

            if percent_covered < min_required:
                failed_files.append(
                    {
                        "file": rel_path,
                        "coverage": percent_covered,
                        "required": min_required,
                    }
                )
                print(f"❌ FAIL: {rel_path} - {percent_covered:.1f}% < {min_required}%")
            else:
                passed_files.append(
                    {
                        "file": rel_path,
                        "coverage": percent_covered,
                        "required": min_required,
                    }
                )
                print(f"✅ PASS: {rel_path} - {percent_covered:.1f}% >= {min_required}%")
        else:
            # File not in requirements, check if strict mode is enabled
            if strict_mode and percent_covered < overall_minimum:
                failed_files.append(
                    {
                        "file": rel_path,
                        "coverage": percent_covered,
                        "required": overall_minimum,
                    }
                )
                print(
                    f"❌ FAIL: {rel_path} - {percent_covered:.1f}% < {overall_minimum}% (strict mode)"
                )
            else:
                print(
                    f"ℹ️  INFO: {rel_path} - {percent_covered:.1f}% (no requirement set)"
                )

    # Check global coverage
    total_percent = data["totals"]["percent_covered"]
    print(f"\n📈 Overall coverage: {total_percent:.1f}%")

    if overall_minimum > 0:
        if total_percent < overall_minimum:
            print(
                f"❌ Overall coverage {total_percent:.1f}% < required {overall_minimum}%"
            )
            return False
        else:
            print(
                f"✅ Overall coverage {total_percent:.1f}% >= required {overall_minimum}%"
            )

    # Summary
    print(f"\n{'='*60}")
    print("📋 COVERAGE SUMMARY")
    print(f"{'='*60}")

    if failed_files:
        print(f"❌ {len(failed_files)} files failed coverage requirements:")
        for file_info in failed_files:
            print(
                f"  - {file_info['file']}: {file_info['coverage']:.1f}% < {file_info['required']}%"
            )
        return False
    else:
        print(f"✅ All {len(passed_files)} files met coverage requirements!")
        if checked_files:
            print(f"📁 Checked {len(checked_files)} files total")
        return True


def main():
    """Main test runner function."""
    print("🚀 AGENTBX COMPREHENSIVE TEST RUNNER")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Change to project root if needed
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Step 1: Run all tests with coverage
    test_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--cov=src/agentbx",
        "--cov-report=json",
        "--cov-report=term-missing",
        "--cov-report=html",
        "-v",
    ]

    if not run_command(test_cmd, "RUNNING ALL TESTS WITH COVERAGE"):
        print("❌ Tests failed!")
        sys.exit(1)

    # Step 2: Check per-file coverage requirements
    if not check_per_file_coverage():
        print("❌ Coverage requirements not met!")
        sys.exit(1)

    # Step 3: Generate additional reports
    print(f"\n{'='*60}")
    print("📄 GENERATING ADDITIONAL REPORTS")
    print(f"{'='*60}")

    # Generate XML report for CI tools
    xml_cmd = [sys.executable, "-m", "coverage", "xml"]
    run_command(xml_cmd, "Generating XML coverage report")

    # Generate detailed HTML report
    html_cmd = [sys.executable, "-m", "coverage", "html"]
    run_command(html_cmd, "Generating detailed HTML coverage report")

    # Final success message
    print(f"\n{'='*60}")
    print("🎉 ALL TESTS PASSED AND COVERAGE REQUIREMENTS MET!")
    print(f"{'='*60}")
    print("📁 Generated reports:")
    print("  - coverage.json (JSON format)")
    print("  - coverage.xml (XML format for CI)")
    print("  - htmlcov/ (Detailed HTML report)")
    print("\n📊 View detailed coverage: open htmlcov/index.html")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
