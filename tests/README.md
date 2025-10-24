# Tests for GenX Analysis Project

This directory contains the test suite for the GenX scenario analysis and visualization platform.

## Overview

The test suite includes:
- **Unit tests**: Fast, isolated tests of individual functions
- **Integration tests**: Tests of multiple components working together
- **Fixtures**: Reusable test data and scenario structures
- **Mock data**: Sample GenX inputs/outputs for testing

## Installation

Install testing dependencies:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Or just pytest
pip install pytest pytest-cov pytest-mock
```

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov

# Generate HTML coverage report
pytest --cov --cov-report=html
# Then open htmlcov/index.html in browser
```

### Filtering Tests

```bash
# Run only unit tests (fast)
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Run specific test file
pytest tests/test_build_generators_data.py

# Run specific test class
pytest tests/test_build_generators_data.py::TestBuildGeneratorsData

# Run specific test function
pytest tests/test_build_generators_data.py::TestBuildGeneratorsData::test_basic_merge

# Run tests matching pattern
pytest -k "merge"  # runs all tests with "merge" in name
```

### Parallel Execution

```bash
# Run tests in parallel (requires pytest-xdist)
pip install pytest-xdist
pytest -n auto  # auto-detect CPU count
pytest -n 4     # use 4 workers
```

### Debugging

```bash
# Stop at first failure
pytest -x

# Drop into debugger on failure
pytest --pdb

# Show print statements
pytest -s

# Show local variables in tracebacks
pytest --showlocals  # (default in our config)
```

## Test Structure

```
tests/
├── __init__.py                      # Tests package init
├── conftest.py                      # Shared fixtures
├── README.md                        # This file
│
├── test_build_generators_data.py    # Tests for build_generators_data.py
├── test_format_genx_result.py       # Tests for format_genx_result_for_plotting.py
├── test_summaries.py                # Tests for create_*_summary.py modules
│
└── fixtures/                        # Sample test data
    ├── __init__.py
    └── sample_scenario/             # Created dynamically by fixtures
        ├── resources/
        ├── results/
        ├── system/
        └── policies/
```

## Test Coverage

Current test coverage by module:

| Module | Coverage | Status |
|--------|----------|--------|
| `build_generators_data.py` | ~85% | ✅ Good |
| `format_genx_result_for_plotting.py` | ~70% | ⚠️ Needs improvement |
| `create_*_summary.py` | ~40% | ⚠️ Interface tests only |
| Overall | ~60% | ⚠️ In progress |

Target: 80% coverage for production release.

## Writing Tests

### Test Naming Convention

- Test files: `test_<module_name>.py`
- Test classes: `Test<Feature>`
- Test functions: `test_<what_it_tests>`

Examples:
```python
# Good names
def test_basic_merge()
def test_missing_resource_folder()
def test_warning_callback()

# Bad names
def test1()
def test_stuff()
def my_test()
```

### Using Fixtures

Fixtures are defined in `conftest.py` and available to all tests:

```python
def test_something(temp_dir, sample_scenario_structure):
    """Test with fixtures."""
    # temp_dir: temporary directory (auto-cleaned)
    # sample_scenario_structure: complete GenX scenario

    output_path = temp_dir / "output.csv"
    # ... test code
```

Available fixtures:
- `temp_dir`: Temporary directory (auto-cleanup)
- `sample_thermal_data`: Sample thermal generator DataFrame
- `sample_vre_data`: Sample VRE DataFrame
- `sample_storage_data`: Sample storage DataFrame
- `sample_hydro_data`: Sample hydro DataFrame
- `sample_net_revenue_data`: Sample NetRevenue with R_ID
- `sample_demand_data`: Sample hourly demand
- `sample_scenario_structure`: Complete scenario folder
- `sample_multi_period_scenario`: Multi-period scenario
- `mock_warning_callback`: Mock for testing warnings

### Test Markers

Use markers to categorize tests:

```python
@pytest.mark.unit
def test_fast_unit_test():
    """Quick unit test."""
    pass

@pytest.mark.integration
def test_full_pipeline():
    """Integration test with multiple components."""
    pass

@pytest.mark.slow
def test_large_dataset():
    """Long-running test."""
    pass
```

### Example Test

```python
import pytest
from pathlib import Path
from build_generators_data import build_generators_data


@pytest.mark.unit
class TestBuildGeneratorsData:
    """Tests for build_generators_data function."""

    def test_basic_merge(self, sample_scenario_structure, temp_dir):
        """Test basic merging of resource files."""
        output_path = temp_dir / "Generators_data.csv"

        result_df = build_generators_data(
            scenario_folder_path=sample_scenario_structure,
            save_file_path=output_path,
            verbose=False,
        )

        # Assertions
        assert output_path.exists()
        assert len(result_df) > 0
        assert 'Resource' in result_df.columns
```

## Common Patterns

### Testing File Operations

```python
def test_file_creation(temp_dir):
    """Test that output file is created."""
    output_path = temp_dir / "output.csv"

    # Run function
    some_function(output_path)

    # Check file exists
    assert output_path.exists()

    # Read and validate
    df = pd.read_csv(output_path)
    assert len(df) > 0
```

### Testing Exceptions

```python
def test_missing_file_raises_error(temp_dir):
    """Test that missing file raises appropriate error."""
    nonexistent_path = temp_dir / "nonexistent"

    with pytest.raises(FileNotFoundError, match="not found"):
        some_function(nonexistent_path)
```

### Testing Warnings

```python
def test_warning_message(mock_warning_callback):
    """Test that warning is raised."""
    some_function(warning_callback=mock_warning_callback)

    assert len(mock_warning_callback.warnings) > 0
    assert "expected warning" in mock_warning_callback.warnings[0]
```

### Testing DataFrames

```python
def test_dataframe_output():
    """Test DataFrame has expected structure."""
    df = create_some_dataframe()

    # Check columns
    assert 'column1' in df.columns

    # Check values
    assert df['column1'].sum() > 0

    # Compare DataFrames
    expected = pd.DataFrame({'A': [1, 2]})
    pd.testing.assert_frame_equal(df, expected)
```

## Continuous Integration

Tests run automatically on:
- Every push to main branch
- Every pull request
- Scheduled nightly builds

CI configuration: `.github/workflows/ci.yml` (to be added)

## Troubleshooting

### Tests fail with import errors

Make sure you're in the project root directory:
```bash
cd /path/to/MIP_results_comparison-1
pytest
```

Or install package in development mode:
```bash
pip install -e .
```

### Fixtures not found

Make sure `conftest.py` is in the `tests/` directory. Pytest automatically discovers fixtures in `conftest.py` files.

### Coverage report missing modules

Make sure you're running pytest from the project root and that the source files are in the Python path.

### Tests pass locally but fail in CI

Check for:
- Hard-coded absolute paths (use `Path` objects and fixtures)
- Tests depending on specific files not in git
- Platform-specific behavior (Windows vs Linux)

## Best Practices

1. **Fast tests**: Unit tests should complete in milliseconds
2. **Isolated tests**: Each test should be independent
3. **Clear assertions**: One logical assertion per test
4. **Good names**: Test names should describe what they test
5. **Use fixtures**: Don't duplicate setup code
6. **Clean up**: Use `temp_dir` fixture for file operations
7. **Test edge cases**: Not just the happy path
8. **Document**: Add docstrings explaining what you're testing

## Contributing

When adding new features:

1. Write tests first (TDD) or alongside code
2. Aim for >80% coverage of new code
3. Include both unit and integration tests
4. Test error cases and edge cases
5. Update this README if adding new test categories

## Resources

- [Pytest documentation](https://docs.pytest.org/)
- [Pytest fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Pytest markers](https://docs.pytest.org/en/stable/example/markers.html)
- [Coverage.py](https://coverage.readthedocs.io/)

## Questions?

If you have questions about testing:
1. Check existing tests for examples
2. See pytest documentation
3. Ask in pull request reviews
4. Open an issue for discussion
