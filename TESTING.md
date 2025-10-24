# Running Tests

This guide walks through the different ways you can execute the test suite for this project from the command line. All commands assume you are in the project root (`C:\Users\Sriki\MIP_results_comparison-1`) and have the `mip_figures` Conda environment activated, or an equivalent Python 3.10+ environment with the required dependencies installed.

## 1. Quick Start (All Tests)

```powershell
pytest
```

Pytest automatically picks up the configuration in `pytest.ini`, so verbose output, warning filters, and strict markers are already enabled.

## 2. Common Variations

```powershell
# Verbose output is on by default, but you can supply it explicitly
pytest -v

# Stop on first failure
pytest -x

# Show print statements during test runs
pytest -s
```

## 3. Targeting Specific Tests

```powershell
# Run a single test file
pytest tests/test_build_generators_data.py

# Run a specific test class
pytest tests/test_build_generators_data.py::TestBuildGeneratorsData

# Run a specific test function
pytest tests/test_build_generators_data.py::TestBuildGeneratorsData::test_basic_merge

# Run tests whose names match an expression
pytest -k "merge"
```

## 4. Using Markers

Markers are defined in `pytest.ini` so you can focus on subsets of the suite.

```powershell
# Run only unit tests (fast)
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Run tests that require real GenX data
pytest -m requires_data
```

## 5. Coverage Reporting

```powershell
# Console coverage summary
pytest --cov

# HTML coverage report (outputs to htmlcov/index.html)
pytest --cov --cov-report=html

# Combine console and HTML reports
pytest --cov --cov-report=term --cov-report=html
```

## 6. Parallel Execution

Requires `pytest-xdist` (already included via `requirements-dev.txt`).

```powershell
# Let pytest decide the worker count
pytest -n auto

# Specify the worker count
pytest -n 4
```

## 7. Convenience Wrappers

Two helper scripts are available if you prefer a single command that configures the environment and runs pytest:

```powershell
# Python wrapper (used by CI)
python run_tests.py
```

Both wrappers ultimately call `pytest` with the same defaults as above.

## 8. Troubleshooting Tips

- Verify you are in the project root before running commands: `Get-Location`
- Ensure dependencies are installed: `pip install -r requirements-dev.txt` or `conda env update -f environment.yml`
- If markers are reported as unknown, double-check you ran the commands from the project root so that `pytest.ini` is discovered.
- To debug failing tests interactively, add `--pdb` to drop into the debugger when a failure occurs: `pytest --pdb`.
