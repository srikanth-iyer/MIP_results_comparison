#!/usr/bin/env python
"""
Convenience script to run tests with common configurations.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --quick      # Run only unit tests
    python run_tests.py --coverage   # Run with coverage report
    python run_tests.py --verbose    # Verbose output
"""
import sys
import subprocess
from pathlib import Path


def run_pytest(args: list[str]) -> int:
    """Run pytest with given arguments.

    Args:
        args: Command-line arguments for pytest

    Returns:
        Exit code from pytest
    """
    cmd = [sys.executable, "-m", "pytest"] + args
    print(f"Running: {' '.join(cmd)}")
    print("=" * 80)
    return subprocess.call(cmd)


def main():
    """Main entry point."""
    # Parse simple command-line arguments
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        return 0

    pytest_args = []

    # Quick mode: only unit tests, no slow tests
    if "--quick" in sys.argv:
        pytest_args.extend(["-m", "unit and not slow"])
        sys.argv.remove("--quick")

    # Coverage mode
    if "--coverage" in sys.argv or "--cov" in sys.argv:
        pytest_args.extend([
            "--cov=.",
            "--cov-report=html",
            "--cov-report=term-missing",
        ])
        if "--coverage" in sys.argv:
            sys.argv.remove("--coverage")
        if "--cov" in sys.argv:
            sys.argv.remove("--cov")

    # Verbose mode
    if "--verbose" in sys.argv or "-v" in sys.argv:
        pytest_args.append("-vv")
        if "--verbose" in sys.argv:
            sys.argv.remove("--verbose")

    # Parallel mode
    if "--parallel" in sys.argv:
        pytest_args.extend(["-n", "auto"])
        sys.argv.remove("--parallel")

    # Pass through any remaining arguments
    remaining_args = [arg for arg in sys.argv[1:] if arg not in ["--quick", "--coverage", "--verbose", "--parallel"]]
    pytest_args.extend(remaining_args)

    # Run pytest
    exit_code = run_pytest(pytest_args)

    # Print summary
    print("=" * 80)
    if exit_code == 0:
        print("✅ All tests passed!")
        if "--coverage" in sys.argv or "--cov" in sys.argv:
            print("📊 Coverage report: htmlcov/index.html")
    else:
        print("❌ Some tests failed!")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
