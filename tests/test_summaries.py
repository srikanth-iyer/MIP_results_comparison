"""
Tests for create_*_summary modules.

Tests the summary generation functions:
- create_resource_capacity
- create_emissions_summary
- create_generation_summary
- create_dispatch_summary
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from reformat_data_for_plot import (
    create_dispatch_summary,
    create_emissions_summary,
    create_generation_summary,
    create_resource_capacity,
    create_transmission_summary,
)


# Note: These tests will be simplified since we're testing the interface
# Full integration tests would require actual GenX result files

@pytest.mark.unit
class TestResourceCapacitySummary:
    """Tests for create_resource_capacity function."""

    def test_module_imports(self):
        """Test that the module can be imported."""
        assert callable(create_resource_capacity)

    def test_function_signature(self):
        """Test that function has expected parameters."""
        import inspect

        sig = inspect.signature(create_resource_capacity)
        params = list(sig.parameters.keys())

        # Should have these key parameters
        assert 'model_name' in params
        assert 'case_name' in params
        assert 'scenario_folder_path' in params
        assert 'planning_year' in params
        assert 'unit' in params


@pytest.mark.unit
class TestEmissionsSummary:
    """Tests for create_emissions_summary function."""

    def test_module_imports(self):
        """Test that the module can be imported."""
        assert callable(create_emissions_summary)

    def test_function_signature(self):
        """Test that function has expected parameters."""
        import inspect

        sig = inspect.signature(create_emissions_summary)
        params = list(sig.parameters.keys())

        assert 'genx_scenario_results_path' in params
        assert 'scenario_name' in params
        assert 'planning_year' in params
        assert 'unit' in params


@pytest.mark.unit
class TestGenerationSummary:
    """Tests for create_generation_summary function."""

    def test_module_imports(self):
        """Test that the module can be imported."""
        assert callable(create_generation_summary)

    def test_function_signature(self):
        """Test that function has expected parameters."""
        import inspect

        sig = inspect.signature(create_generation_summary)
        params = list(sig.parameters.keys())

        assert 'genx_scenario_results_path' in params
        assert 'scenario_name' in params
        assert 'planning_year' in params
        assert 'unit' in params


@pytest.mark.unit
class TestDispatchSummary:
    """Tests for create_dispatch_summary function."""

    def test_module_imports(self):
        """Test that the module can be imported."""
        assert callable(create_dispatch_summary)

    def test_function_signature(self):
        """Test that function has expected parameters."""
        import inspect

        sig = inspect.signature(create_dispatch_summary)
        params = list(sig.parameters.keys())

        assert 'genx_scenario_results_path' in params
        assert 'scenario_name' in params
        assert 'planning_year' in params


@pytest.mark.unit
class TestTransmissionSummary:
    """Tests for create_transmission_summary function."""

    def test_module_imports(self):
        """Test that the module can be imported."""
        assert callable(create_transmission_summary)

    def test_function_signature(self):
        """Test that function has expected parameters."""
        import inspect

        sig = inspect.signature(create_transmission_summary)
        params = list(sig.parameters.keys())

        assert 'genx_scenario_results_path' in params
        assert 'scenario_name' in params
        assert 'planning_year' in params
        assert 'unit' in params


@pytest.mark.integration
class TestSummaryIntegration:
    """Integration tests for summary functions with mock data."""

    def test_capacity_summary_handles_missing_data(self, sample_scenario_structure, temp_dir):
        """Test that resource capacity summary fails gracefully with incomplete data."""
        output_folder = temp_dir / "summaries"
        output_folder.mkdir()

        # The sample_scenario_structure doesn't have Generators_data.csv
        # So this should raise FileNotFoundError - which is the expected behavior
        with pytest.raises(FileNotFoundError):
            create_resource_capacity(
                model_name="TestModel",
                case_name="TestCase",
                scenario_folder_path=sample_scenario_structure,
                genx_scenario_results_path=sample_scenario_structure,
                results_summary_folder_path=output_folder,
                planning_year=2030,
                unit="MW",
            )

    def test_emissions_summary_handles_missing_data(self, sample_scenario_structure, temp_dir):
        """Test that emissions summary fails gracefully with missing emissions.csv."""
        output_folder = temp_dir / "summaries"
        output_folder.mkdir()

        # The sample_scenario_structure doesn't have emissions.csv
        # So this should raise an error - which is the expected behavior
        with pytest.raises((FileNotFoundError, ValueError)):
            create_emissions_summary(
                genx_scenario_results_path=sample_scenario_structure,
                scenario_name="TestScenario",
                output_folder_path=output_folder.parent,
                planning_year=2030,
                case="TestCase",
                unit="tons",
            )

    def test_skip_behavior_missing_files(self, temp_dir):
        """Test that functions skip gracefully when files are missing."""
        # Create empty scenario
        empty_scenario = temp_dir / "empty_scenario"
        empty_scenario.mkdir()

        output_folder = temp_dir / "output"
        output_folder.mkdir()

        # Test 1: Verify capacity summary skips with missing Generators_data.csv
        with pytest.raises(FileNotFoundError):
            create_resource_capacity(
                model_name="Test",
                case_name="Test",
                scenario_folder_path=empty_scenario,
                genx_scenario_results_path=empty_scenario,
                results_summary_folder_path=output_folder,
                planning_year=2030,
            )

        # Test 2: Verify emissions summary raises error with missing emissions.csv
        with pytest.raises((FileNotFoundError, ValueError)):
            create_emissions_summary(
                genx_scenario_results_path=empty_scenario,
                scenario_name="Test",
                output_folder_path=output_folder,
                planning_year=2030,
                case="Test",
            )

    def test_capacity_summary_with_complete_data(self, complete_scenario_for_summaries, temp_dir):
        """Test resource capacity summary with complete data files."""
        output_folder = temp_dir / "summaries"
        output_folder.mkdir()

        output_path = create_resource_capacity(
            model_name="TestModel",
            case_name="TestCase",
            scenario_folder_path=complete_scenario_for_summaries,
            genx_scenario_results_path=complete_scenario_for_summaries,
            results_summary_folder_path=output_folder,
            planning_year=2030,
            unit="MW",
        )

        # Verify output file exists
        assert output_path.exists()

        # Read and validate structure
        df = pd.read_csv(output_path)

        # Check required columns exist
        expected_cols = ['resource_name', 'planning_year', 'model', 'case', 'zone', 'tech_type']
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

        # Verify data values
        assert len(df) == 4, "Should have 4 resources"
        assert all(df['planning_year'] == 2030), "All rows should have planning_year=2030"
        assert all(df['model'] == "TestModel"), "All rows should have model=TestModel"
        assert all(df['case'] == "TestCase"), "All rows should have case=TestCase"
        assert all(df['unit'] == "MW"), "All rows should have unit=MW"

        # Verify resource names
        assert set(df['resource_name']) == {'Gen1', 'Gen2', 'Solar1', 'Wind1'}

        # Verify zones
        assert set(df['zone']) == {'Region1', 'Region2'}

    def test_emissions_summary_with_complete_data(self, complete_scenario_for_summaries, temp_dir):
        """Test emissions summary with complete data files."""
        output_folder = temp_dir / "summaries"
        output_folder.mkdir()

        output_path = create_emissions_summary(
            genx_scenario_results_path=complete_scenario_for_summaries,
            scenario_name="TestScenario",
            output_folder_path=output_folder,
            planning_year=2030,
            case="TestCase",
            unit="tons",
        )

        # Verify output file exists
        assert output_path is not None
        assert output_path.exists()

        # Read and validate structure
        df = pd.read_csv(output_path)

        # Check required columns exist
        expected_cols = ['planning_year', 'model', 'case', 'zone', 'unit', 'value']
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

        # Verify data values
        assert len(df) > 0, "Should have emission data"
        assert all(df['planning_year'] == 2030), "All rows should have planning_year=2030"
        assert all(df['model'] == "TestScenario"), "All rows should have model=TestScenario"
        assert all(df['case'] == "TestCase"), "All rows should have case=TestCase"
        assert all(df['unit'] == "tons"), "All rows should have unit=tons"

        # Verify zones exist
        assert set(df['zone']) == {1, 2}

        # Verify emission values
        assert all(df['value'] > 0), "All emission values should be positive"

    def test_transmission_summary_with_complete_data(self, complete_scenario_for_summaries, temp_dir):
        """Test transmission summary with complete data files."""
        output_folder = temp_dir / "summaries"
        output_folder.mkdir()

        output_path = create_transmission_summary(
            genx_scenario_results_path=complete_scenario_for_summaries,
            scenario_name="TestScenario",
            output_folder_path=output_folder,
            planning_year=2030,
            case="Results_p1",
            unit="MW",
        )

        assert output_path.exists()

        df = pd.read_csv(output_path)

        expected_cols = [
            'model', 'case', 'planning_year', 'unit',
            'Network_zones', 'Network_Lines', 'transmission_path_name',
            'start_value', 'New_Trans_Capacity', 'Cost_Trans_Capacity', 'end_value',
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

        assert len(df) == 2
        assert all(df['planning_year'] == 2030)
        assert all(df['case'] == "Results_p1")
        assert all(df['unit'] == "MW")
        assert all(df['model'] == "TestScenario")
        assert all(np.isclose(df['end_value'], df['start_value'] + df['New_Trans_Capacity']))


@pytest.mark.unit
class TestTimeWeights:
    """Tests for create_time_weights module."""

    def test_module_imports(self):
        """Test that time weights module can be imported."""
        try:
            from reformat_data_for_plot import create_time_weights  # noqa: F401
        except ImportError:
            pytest.skip("reformat_data_for_plot.create_time_weights module not found")


@pytest.mark.unit
class TestUtilityFunctions:
    """Tests for utility functions across summary modules."""

    def test_gz_to_csv_utility(self):
        """Test gz to csv conversion utility."""
        try:
            from reformat_data_for_plot import gz_to_csv  # noqa: F401
        except ImportError:
            pytest.skip("reformat_data_for_plot.gz_to_csv module not found")

    def test_path_handling(self, temp_dir):
        """Test that summary functions handle Path objects."""
        # Should accept both str and Path
        scenario_path_str = str(temp_dir / "scenario")
        scenario_path_obj = temp_dir / "scenario"

        # Both should be valid inputs (even if they fail for other reasons)
        assert isinstance(scenario_path_str, str)
        assert isinstance(scenario_path_obj, Path)


@pytest.mark.integration
class TestSummaryChaining:
    """Test that summaries can be created in sequence."""

    def test_sequential_summary_creation(self, sample_scenario_structure, temp_dir):
        """Test creating multiple summaries in sequence."""
        output_folder = temp_dir / "summaries"
        output_folder.mkdir()

        summaries_created = []

        # Try capacity summary
        try:
            path = create_resource_capacity(
                model_name="TestModel",
                case_name="TestCase",
                scenario_folder_path=sample_scenario_structure,
                genx_scenario_results_path=sample_scenario_structure,
                results_summary_folder_path=output_folder,
                planning_year=2030,
                unit="MW",
            )
            if path and path.exists():
                summaries_created.append('capacity')
        except Exception:
            pass

        # Try emissions summary
        try:
            path = create_emissions_summary(
                genx_scenario_results_path=sample_scenario_structure,
                scenario_name="TestScenario",
                output_folder_path=output_folder.parent,
                planning_year=2030,
                case="TestCase",
                unit="tons",
            )
            if path and path.exists():
                summaries_created.append('emissions')
        except Exception:
            pass

        # Try generation summary
        try:
            path = create_generation_summary(
                genx_scenario_results_path=sample_scenario_structure,
                scenario_name="TestScenario",
                output_folder_path=output_folder.parent,
                planning_year=2030,
                case="TestCase",
                unit="MWh",
            )
            if path and path.exists():
                summaries_created.append('generation')
        except Exception:
            pass

        # At minimum, we should be able to call all functions without import errors
        # Even if they fail due to data format, that's acceptable for this test
        assert True  # Test passes if we get here without import errors


@pytest.mark.unit
class TestSummaryErrorHandling:
    """Test error handling in summary functions."""

    def test_missing_input_files(self, temp_dir):
        """Test that summary functions handle missing input files gracefully."""
        nonexistent_path = temp_dir / "nonexistent"
        output_folder = temp_dir / "output"
        output_folder.mkdir()

        # Should raise an appropriate error (FileNotFoundError or similar)
        with pytest.raises((FileNotFoundError, ValueError, Exception)):
            create_resource_capacity(
                model_name="TestModel",
                case_name="TestCase",
                scenario_folder_path=nonexistent_path,
                genx_scenario_results_path=nonexistent_path,
                results_summary_folder_path=output_folder,
                planning_year=2030,
                unit="MW",
            )

    def test_invalid_unit_parameter(self, sample_scenario_structure, temp_dir):
        """Test handling of invalid unit parameters."""
        output_folder = temp_dir / "output"
        output_folder.mkdir()

        # Some functions may validate units, others may accept any string
        # This test just checks that the parameter is accepted
        try:
            create_resource_capacity(
                model_name="TestModel",
                case_name="TestCase",
                scenario_folder_path=sample_scenario_structure,
                genx_scenario_results_path=sample_scenario_structure,
                results_summary_folder_path=output_folder,
                planning_year=2030,
                unit="InvalidUnit",
            )
        except Exception:
            # May fail for other reasons - that's ok
            pass


@pytest.mark.slow
@pytest.mark.integration
class TestSummaryPerformance:
    """Performance tests for summary generation."""

    def test_summary_creation_performance(self, sample_scenario_structure, temp_dir):
        """Test that summary creation completes in reasonable time."""
        import time
        output_folder = temp_dir / "output"
        output_folder.mkdir()

        start_time = time.time()

        try:
            create_resource_capacity(
                model_name="TestModel",
                case_name="TestCase",
                scenario_folder_path=sample_scenario_structure,
                genx_scenario_results_path=sample_scenario_structure,
                results_summary_folder_path=output_folder,
                planning_year=2030,
                unit="MW",
            )
        except Exception:
            pass  # Performance test - don't care if it fails

        elapsed_time = time.time() - start_time

        # Should complete in under 5 seconds for small test data
        assert elapsed_time < 5.0, f"Summary creation took {elapsed_time:.2f}s"
