"""
Tests package for GenX Analysis project.

This package contains unit tests, integration tests, and fixtures for testing
the GenX data processing and visualization pipeline.

Test categories:
    - Unit tests: Fast, isolated tests of individual functions
    - Integration tests: Tests of multiple components working together
    - Slow tests: Long-running tests (can be skipped)

Run tests with:
    pytest                           # Run all tests
    pytest -v                        # Verbose output
    pytest -m unit                   # Run only unit tests
    pytest -m "not slow"             # Skip slow tests
    pytest --cov                     # With coverage report
    pytest tests/test_build_generators_data.py  # Single file
"""
__version__ = "0.1.0"
