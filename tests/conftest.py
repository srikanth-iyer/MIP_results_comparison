"""
Pytest configuration and shared fixtures for GenX analysis tests.

This file provides reusable fixtures for testing the GenX data processing pipeline.
"""
import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs.

    Yields:
        Path: Path to temporary directory (automatically cleaned up after test)
    """
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup after test
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def sample_thermal_data():
    """Sample thermal generator data for testing.

    Returns:
        pd.DataFrame: DataFrame with thermal generator attributes
    """
    return pd.DataFrame({
        'Resource': ['NGCC_1', 'NGCC_2', 'Coal_1'],
        'Zone': [1, 2, 1],
        'technology': ['Natural_Gas_CC', 'Natural_Gas_CC', 'Coal'],
        'Existing_Cap_MW': [500.0, 300.0, 400.0],
        'Heat_Rate_MMBTU_per_MWh': [7.5, 7.8, 10.2],
        'Fuel': ['Natural_Gas', 'Natural_Gas', 'Coal'],
        'Min_Power': [0.3, 0.3, 0.4],
        'Var_OM_Cost_per_MWh': [3.5, 3.6, 4.2],
        'Start_Cost_per_MW': [91.0, 91.0, 150.0],
    })


@pytest.fixture
def sample_vre_data():
    """Sample VRE (Variable Renewable Energy) data for testing.

    Returns:
        pd.DataFrame: DataFrame with VRE attributes
    """
    return pd.DataFrame({
        'Resource': ['Wind_1', 'Wind_2', 'Solar_1'],
        'Zone': [1, 2, 1],
        'technology': ['Onshore_Wind', 'Offshore_Wind', 'Solar_PV'],
        'Existing_Cap_MW': [200.0, 150.0, 100.0],
        'Var_OM_Cost_per_MWh': [0.0, 0.0, 0.0],
        'Inv_Cost_per_MWyr': [50000.0, 80000.0, 40000.0],
    })


@pytest.fixture
def sample_storage_data():
    """Sample storage resource data for testing.

    Returns:
        pd.DataFrame: DataFrame with storage attributes
    """
    return pd.DataFrame({
        'Resource': ['Battery_1', 'Battery_2'],
        'Zone': [1, 2],
        'technology': ['Li_Ion_Battery', 'Li_Ion_Battery'],
        'Existing_Cap_MW': [50.0, 75.0],
        'Existing_Cap_MWh': [200.0, 300.0],
        'Eff_Up': [0.92, 0.92],
        'Eff_Down': [0.92, 0.92],
        'Self_Disch': [0.01, 0.01],
    })


@pytest.fixture
def sample_hydro_data():
    """Sample hydro resource data for testing.

    Returns:
        pd.DataFrame: DataFrame with hydro attributes
    """
    return pd.DataFrame({
        'Resource': ['Hydro_1'],
        'Zone': [1],
        'technology': ['Hydro'],
        'Existing_Cap_MW': [1000.0],
        'Hydro_Energy_to_Power_Ratio': [8.0],
    })


@pytest.fixture
def sample_net_revenue_data():
    """Sample NetRevenue data with R_ID mappings for testing.

    Returns:
        pd.DataFrame: DataFrame with Resource to R_ID mappings
    """
    return pd.DataFrame({
        'R_ID': [1, 2, 3, 4, 5, 6, 7],
        'Resource': ['NGCC_1', 'NGCC_2', 'Coal_1', 'Wind_1', 'Wind_2', 'Solar_1', 'Battery_1'],
        'NetRevenue': [1000000, 800000, -200000, 500000, 400000, 300000, 200000],
    })


@pytest.fixture
def sample_demand_data():
    """Sample demand/load data for testing.

    Returns:
        pd.DataFrame: DataFrame with hourly demand by zone
    """
    # Create 24 hours of sample data
    hours = list(range(1, 25))
    return pd.DataFrame({
        'Time_Index': hours,
        'Demand_MW_z1': [1000 + i * 10 for i in hours],
        'Demand_MW_z2': [800 + i * 8 for i in hours],
        'Demand_MW_z3': [600 + i * 6 for i in hours],
    })


@pytest.fixture
def sample_power_data():
    """Sample power output data for testing.

    Returns:
        pd.DataFrame: DataFrame with power output by resource
    """
    # 24 hours, 7 resources
    hours = list(range(1, 25))
    data = {'Time_Index': hours}

    # Add columns for each resource (R1, R2, etc.)
    for r_id in range(1, 8):
        data[f'R{r_id}'] = [100.0 + r_id * 10 + h for h in hours]

    return pd.DataFrame(data)


@pytest.fixture
def sample_emissions_data():
    """Sample emissions data for testing.

    Returns:
        pd.DataFrame: DataFrame with emissions by zone
    """
    return pd.DataFrame({
        'Zone': [1, 2, 3],
        'CO2': [50000.0, 40000.0, 30000.0],
    })


@pytest.fixture
def sample_capacity_factor_data():
    """Sample capacity factor data for testing.

    Returns:
        pd.DataFrame: DataFrame with capacity factors by resource
    """
    return pd.DataFrame({
        'Resource': ['NGCC_1', 'NGCC_2', 'Wind_1', 'Solar_1'],
        'R_ID': [1, 2, 4, 6],
        'CapacityFactor': [0.65, 0.55, 0.35, 0.25],
    })


@pytest.fixture
def sample_scenario_structure(temp_dir, sample_thermal_data, sample_vre_data,
                               sample_storage_data, sample_hydro_data,
                               sample_net_revenue_data, sample_demand_data):
    """Create a complete sample scenario folder structure for testing.

    Args:
        temp_dir: Temporary directory fixture
        sample_*_data: Various sample data fixtures

    Returns:
        Path: Path to the sample scenario root folder
    """
    scenario_path = temp_dir / "sample_scenario"

    # Create directory structure
    resources_dir = scenario_path / "resources"
    results_dir = scenario_path / "results"
    system_dir = scenario_path / "system"
    policies_dir = scenario_path / "policies"

    for directory in [resources_dir, results_dir, system_dir, policies_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Write resource files
    sample_thermal_data.to_csv(resources_dir / "Thermal.csv", index=False)
    sample_vre_data.to_csv(resources_dir / "Vre.csv", index=False)
    sample_storage_data.to_csv(resources_dir / "Storage.csv", index=False)
    sample_hydro_data.to_csv(resources_dir / "Hydro.csv", index=False)

    # Write results files
    sample_net_revenue_data.to_csv(results_dir / "NetRevenue.csv", index=False)

    # Write system files
    sample_demand_data.to_csv(system_dir / "Demand_data.csv", index=False)

    # Create a simple Network.csv
    network_data = pd.DataFrame({
        'Network_Lines': ['Line1'],
        'z1': [1],
        'z2': [2],
        'Line_Max_Flow_MW': [500],
    })
    network_data.to_csv(system_dir / "Network.csv", index=False)

    return scenario_path


@pytest.fixture
def complete_scenario_for_summaries(temp_dir, sample_thermal_data, sample_vre_data,
                                     sample_storage_data, sample_demand_data):
    """Create a complete scenario with all files needed for summary tests.

    This fixture includes:
    - Generators_data.csv (output of build_generators_data)
    - emissions.csv, capacity.csv, capacityfactor.csv in results/
    - All necessary columns for summary functions

    Returns:
        Path: Path to the complete scenario folder
    """
    scenario_path = temp_dir / "complete_scenario"

    # Create directory structure
    results_dir = scenario_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create Generators_data.csv (as output from build_generators_data)
    # Must have 'Resource' column for create_resource_capacity to work
    generators_data = pd.DataFrame({
        'Resource': ['Gen1', 'Gen2', 'Solar1', 'Wind1'],
        'Zone': [1, 1, 2, 2],
        'region': ['Region1', 'Region1', 'Region2', 'Region2'],
        'category': ['Natural Gas', 'Coal', 'Solar', 'Wind'],
        'technology': ['Natural_Gas_CC', 'Coal_Steam', 'Solar_PV', 'Onshore_Wind'],
        'Existing_Cap_MW': [100, 200, 50, 150],
        'New_Build': [1, 0, 1, 1],
        'Cap_Size': [100, 200, 50, 150],
    })
    generators_data.to_csv(scenario_path / "Generators_data.csv", index=False)

    # Create emissions.csv with numeric zone columns
    emissions_data = pd.DataFrame({
        'Zone': ['Gen1', 'Gen2', 'AnnualSum'],
        '1': [500.0, 800.0, 1300.0],
        '2': [500.5, 1200.3, 1700.8],
        'Total': [1000.5, 2000.3, 3000.8],
    })
    emissions_data.to_csv(results_dir / "emissions.csv", index=False)

    # Create capacity.csv
    capacity_data = pd.DataFrame({
        'Resource': ['Gen1', 'Gen2', 'Solar1', 'Wind1'],
        'StartCap': [100.0, 200.0, 50.0, 150.0],
        'RetCap': [0.0, 0.0, 0.0, 0.0],
        'NewCap': [50.0, 0.0, 100.0, 50.0],
        'EndCap': [150.0, 200.0, 150.0, 200.0],
    })
    capacity_data.to_csv(results_dir / "capacity.csv", index=False)

    # Create capacityfactor.csv
    capacityfactor_data = pd.DataFrame({
        'Resource': ['Gen1', 'Gen2', 'Solar1', 'Wind1'],
        'Zone': [1, 1, 2, 2],
        'AnnualSum': [5000.0, 8000.0, 3000.0, 6000.0],
    })
    capacityfactor_data.to_csv(results_dir / "capacityfactor.csv", index=False)

    return scenario_path


@pytest.fixture
def sample_multi_period_scenario(temp_dir, sample_thermal_data, sample_demand_data):
    """Create a multi-period scenario structure for testing.

    Args:
        temp_dir: Temporary directory fixture
        sample_thermal_data: Sample thermal data
        sample_demand_data: Sample demand data

    Returns:
        Path: Path to multi-period scenario root
    """
    scenario_path = temp_dir / "multi_period_scenario"

    # Create 4 periods
    for period in ['p1', 'p2', 'p3', 'p4']:
        period_path = scenario_path / f"Inputs_{period}"

        resources_dir = period_path / "resources"
        results_dir = period_path / "results"
        system_dir = period_path / "system"

        for directory in [resources_dir, results_dir, system_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Write files for each period
        sample_thermal_data.to_csv(resources_dir / "Thermal.csv", index=False)
        sample_demand_data.to_csv(system_dir / "Demand_data.csv", index=False)

    return scenario_path


@pytest.fixture
def expected_generators_columns():
    """Expected columns in Generators_data.csv output.

    Returns:
        list: List of expected column names
    """
    return [
        'Resource', 'region', 'technology', 'cluster', 'R_ID', 'Zone',
        'THERM', 'VRE', 'MUST_RUN', 'STOR', 'FLEX', 'HYDRO',
        'Existing_Cap_MW', 'Existing_Cap_MWh', 'Heat_Rate_MMBTU_per_MWh',
        'Fuel', 'Min_Power', 'Var_OM_Cost_per_MWh',
    ]


@pytest.fixture
def mock_warning_callback():
    """Create a mock warning callback for testing warning messages.

    Returns:
        callable: Function that records warnings
    """
    warnings_recorded = []

    def callback(message: str):
        warnings_recorded.append(message)

    callback.warnings = warnings_recorded
    return callback


@pytest.fixture
def real_sample_scenario_1():
    """Path to real sample scenario 1 with actual multi-period GenX data.

    Returns:
        Path: Path to sample_scenario_1 folder with inputs_p1 through inputs_p4
    """
    scenario_path = Path(__file__).parent / "fixtures" / "sample_scenario_1"
    if not scenario_path.exists():
        pytest.skip("Real sample scenario 1 not found")
    return scenario_path


@pytest.fixture
def real_sample_scenario_2():
    """Path to real sample scenario 2 with actual multi-period GenX data.

    Returns:
        Path: Path to sample_scenario_2 folder with inputs_p1 through inputs_p4
    """
    scenario_path = Path(__file__).parent / "fixtures" / "sample_scenario_2"
    if not scenario_path.exists():
        pytest.skip("Real sample scenario 2 not found")
    return scenario_path


@pytest.fixture(params=["sample_scenario_1", "sample_scenario_2"])
def real_sample_scenarios(request):
    """Parametrized fixture that runs tests with both sample scenarios.

    Args:
        request: pytest request object with param

    Returns:
        Path: Path to the current scenario being tested
    """
    scenario_path = Path(__file__).parent / "fixtures" / request.param
    if not scenario_path.exists():
        pytest.skip(f"Scenario {request.param} not found")
    return scenario_path


# Mark for pytest-xdist (parallel execution)
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "real_data: marks tests that use real sample scenario data"
    )
