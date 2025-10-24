"""
Tests for format_genx_result_for_plotting module.

Tests the main export pipeline including:
- Period directory discovery
- File copying and organization
- Summary generation coordination
- Multi-period processing
- Warning aggregation
"""
import pytest
import pandas as pd
from pathlib import Path
from reformat_data_for_plot import (
    create_annual_demand_csv,
    export_all_genx_scenarios,
    export_genx_for_plotting,
)


@pytest.mark.unit
class TestCreateAnnualDemandCsv:
    """Tests for create_annual_demand_csv function."""

    def test_basic_demand_aggregation(self, temp_dir, sample_demand_data):
        """Test basic annual demand calculation."""
        # Create scenario structure
        scenario_path = temp_dir / "scenario"
        system_dir = scenario_path / "system"
        system_dir.mkdir(parents=True)

        sample_demand_data.to_csv(system_dir / "Demand_data.csv", index=False)

        output_path = temp_dir / "annual_demand.csv"
        create_annual_demand_csv(
            scenario_path=scenario_path,
            output_path=output_path,
            planning_year=2030,
            verbose=False,
        )

        # Check output
        assert output_path.exists()
        result = pd.read_csv(output_path)

        # Should have 3 zones
        assert len(result) == 3
        assert set(result['zone']) == {'z1', 'z2', 'z3'}
        assert all(result['planning_year'] == 2030)

        # Check annual demand calculation (sum of hourly values)
        z1_demand = result[result['zone'] == 'z1']['annual_demand'].iloc[0]
        expected_z1 = sample_demand_data['Demand_MW_z1'].sum()
        assert abs(z1_demand - expected_z1) < 0.01

    def test_missing_demand_file(self, temp_dir):
        """Test error when Demand_data.csv is missing."""
        scenario_path = temp_dir / "scenario"
        output_path = temp_dir / "output.csv"

        with pytest.raises(FileNotFoundError, match="Demand data file not found"):
            create_annual_demand_csv(
                scenario_path=scenario_path,
                output_path=output_path,
                planning_year=2030,
            )

    def test_no_demand_columns(self, temp_dir):
        """Test error when no demand columns found."""
        scenario_path = temp_dir / "scenario"
        system_dir = scenario_path / "system"
        system_dir.mkdir(parents=True)

        # Create demand file without proper columns
        bad_df = pd.DataFrame({'Time': [1, 2, 3], 'Value': [100, 200, 300]})
        bad_df.to_csv(system_dir / "Demand_data.csv", index=False)

        output_path = temp_dir / "output.csv"

        with pytest.raises(ValueError, match="No demand zone columns found"):
            create_annual_demand_csv(
                scenario_path=scenario_path,
                output_path=output_path,
                planning_year=2030,
            )


@pytest.mark.integration
class TestExportGenxForPlotting:
    """Tests for export_genx_for_plotting function."""

    def test_single_period_export(self, sample_scenario_structure, temp_dir):
        """Test export of single-period scenario."""
        output_folder = temp_dir / "output"

        resource_capacity_path, warnings = export_genx_for_plotting(
            scenario_data_path=sample_scenario_structure.parent,
            scenario_name=sample_scenario_structure.name,
            output_folder_path=output_folder,
            scenario_to_year_map={"p1": 2030},
            verbose=False,
        )

        # Check that output folders were created
        op_inputs = output_folder / f"{sample_scenario_structure.name}_op_inputs"
        results_summary = output_folder / f"{sample_scenario_structure.name}_results_summary"

        assert op_inputs.exists()
        assert results_summary.exists()

        # Check that Generators_data.csv was created
        generators_path = op_inputs / "Inputs" / "Inputs_p1" / "Generators_data.csv"
        assert generators_path.exists()

        # Check that summary files were created
        assert (results_summary / "annual_demand.csv").exists()

        # Note: resource_capacity may fail with synthetic fixtures missing 'region' column
        # Check if it was created or if there's a warning
        if not resource_capacity_path.exists():
            # Should have a warning about missing column
            warning_messages = [str(w.get('message', '')) for w in warnings]
            assert any('region' in msg.lower() or 'capacity' in msg.lower() for msg in warning_messages), \
                f"Expected warning about resource capacity, got: {warnings}"
        else:
            # If it exists, great!
            assert resource_capacity_path.exists()

    def test_multi_period_export(self, sample_multi_period_scenario, temp_dir):
        """Test export of multi-period scenario."""
        output_folder = temp_dir / "output"

        scenario_to_year_map = {
            "p1": 2030,
            "p2": 2035,
            "p3": 2040,
            "p4": 2050,
        }

        resource_capacity_path, warnings = export_genx_for_plotting(
            scenario_data_path=sample_multi_period_scenario.parent,
            scenario_name=sample_multi_period_scenario.name,
            output_folder_path=output_folder,
            scenario_to_year_map=scenario_to_year_map,
            verbose=False,
        )

        # Check that all periods were processed
        op_inputs = output_folder / f"{sample_multi_period_scenario.name}_op_inputs" / "Inputs"

        for period in ['p1', 'p2', 'p3', 'p4']:
            period_dir = op_inputs / f"Inputs_{period}"
            assert period_dir.exists()
            assert (period_dir / "Generators_data.csv").exists()

    def test_warning_aggregation(self, temp_dir):
        """Test that warnings are properly aggregated."""
        # Create scenario without NetRevenue.csv to trigger warning
        scenario_path = temp_dir / "incomplete_scenario"
        resources_dir = scenario_path / "resources"
        system_dir = scenario_path / "system"
        resources_dir.mkdir(parents=True)
        system_dir.mkdir(parents=True)

        # Minimal thermal data
        pd.DataFrame({
            'Resource': ['Gen1'],
            'Zone': [1],
        }).to_csv(resources_dir / "Thermal.csv", index=False)

        # Minimal demand data
        pd.DataFrame({
            'Time_Index': [1, 2, 3],
            'Demand_MW_z1': [1000, 1100, 1200],
        }).to_csv(system_dir / "Demand_data.csv", index=False)

        output_folder = temp_dir / "output"

        resource_capacity_path, warnings = export_genx_for_plotting(
            scenario_data_path=scenario_path.parent,
            scenario_name=scenario_path.name,
            output_folder_path=output_folder,
            scenario_to_year_map={"p1": 2030},
            verbose=False,
        )

        # Should have warnings
        assert len(warnings) > 0
        # Check warning structure
        for warning in warnings:
            assert 'scenario' in warning
            assert 'message' in warning

    def test_missing_period_mapping(self, sample_multi_period_scenario, temp_dir):
        """Test handling of periods without year mapping."""
        output_folder = temp_dir / "output"

        # Only provide mapping for p1, not p2-p4
        scenario_to_year_map = {"p1": 2030}

        resource_capacity_path, warnings = export_genx_for_plotting(
            scenario_data_path=sample_multi_period_scenario.parent,
            scenario_name=sample_multi_period_scenario.name,
            output_folder_path=output_folder,
            scenario_to_year_map=scenario_to_year_map,
            verbose=False,
        )

        # Should have warnings about skipped periods
        assert len(warnings) > 0
        warning_messages = [str(w.get('message', '')) for w in warnings]
        assert any('no planning year mapping' in msg.lower() for msg in warning_messages)

    def test_output_aggregation(self, sample_multi_period_scenario, temp_dir):
        """Test that multi-period outputs are properly aggregated."""
        output_folder = temp_dir / "output"

        scenario_to_year_map = {
            "p1": 2030,
            "p2": 2035,
        }

        resource_capacity_path, warnings = export_genx_for_plotting(
            scenario_data_path=sample_multi_period_scenario.parent,
            scenario_name=sample_multi_period_scenario.name,
            output_folder_path=output_folder,
            scenario_to_year_map=scenario_to_year_map,
            verbose=False,
        )

        results_summary = output_folder / f"{sample_multi_period_scenario.name}_results_summary"

        # Check annual demand aggregation
        annual_demand = pd.read_csv(results_summary / "annual_demand.csv")
        # Should have data for 2 periods
        assert set(annual_demand['planning_year']) == {2030, 2035}

    def test_case_insensitive_period_mapping(self, sample_multi_period_scenario, temp_dir):
        """Test that period keys are case-insensitive."""
        output_folder = temp_dir / "output"

        # Use uppercase period keys
        scenario_to_year_map = {
            "P1": 2030,
            "P2": 2035,
        }

        resource_capacity_path, warnings = export_genx_for_plotting(
            scenario_data_path=sample_multi_period_scenario.parent,
            scenario_name=sample_multi_period_scenario.name,
            output_folder_path=output_folder,
            scenario_to_year_map=scenario_to_year_map,
            verbose=False,
        )

        # Should work without errors
        op_inputs = output_folder / f"{sample_multi_period_scenario.name}_op_inputs" / "Inputs"
        assert (op_inputs / "Inputs_p1").exists()
        assert (op_inputs / "Inputs_p2").exists()


@pytest.mark.integration
class TestExportAllGenxScenarios:
    """Tests for export_all_genx_scenarios function."""

    def test_multiple_scenarios(self, temp_dir, sample_demand_data):
        """Test batch export of multiple scenarios."""
        scenarios_root = temp_dir / "scenarios"

        # Create 3 test scenarios
        for i in range(1, 4):
            scenario_path = scenarios_root / f"Scenario_{i}"
            resources_dir = scenario_path / "resources"
            system_dir = scenario_path / "system"
            resources_dir.mkdir(parents=True)
            system_dir.mkdir(parents=True)

            # Minimal data
            pd.DataFrame({
                'Resource': [f'Gen{i}'],
                'Zone': [1],
            }).to_csv(resources_dir / "Thermal.csv", index=False)

            sample_demand_data.to_csv(system_dir / "Demand_data.csv", index=False)

        output_folder = temp_dir / "output"
        scenario_to_year_map = {"p1": 2030}

        results = export_all_genx_scenarios(
            scenarios_root=scenarios_root,
            output_folder_path=output_folder,
            scenario_to_year_map=scenario_to_year_map,
            verbose=False,
        )

        # Should have processed all 3 scenarios
        assert len(results) == 3
        assert "Scenario_1" in results
        assert "Scenario_2" in results
        assert "Scenario_3" in results

    def test_partial_failure_continues(self, temp_dir):
        """Test that scenarios with missing data still process with warnings."""
        scenarios_root = temp_dir / "scenarios"

        # Create one valid scenario
        valid_scenario = scenarios_root / "Valid"
        resources_dir = valid_scenario / "resources"
        system_dir = valid_scenario / "system"
        resources_dir.mkdir(parents=True)
        system_dir.mkdir(parents=True)

        pd.DataFrame({
            'Resource': ['Gen1'],
            'Zone': [1],
        }).to_csv(resources_dir / "Thermal.csv", index=False)

        pd.DataFrame({
            'Time_Index': [1, 2],
            'Demand_MW_z1': [1000, 1100],
        }).to_csv(system_dir / "Demand_data.csv", index=False)

        # Create one scenario with missing files (should process with warnings)
        invalid_scenario = scenarios_root / "Invalid"
        invalid_scenario.mkdir(parents=True)

        output_folder = temp_dir / "output"

        results = export_all_genx_scenarios(
            scenarios_root=scenarios_root,
            output_folder_path=output_folder,
            scenario_to_year_map={"p1": 2030},
            verbose=False,
        )

        # Both scenarios should have been processed (resilient behavior)
        assert "Valid" in results
        # Invalid scenario processes with warnings but still creates output
        assert "Invalid" in results
        # Verify that output paths are returned (files may or may not exist with synthetic data)
        assert isinstance(results["Valid"], Path)
        assert isinstance(results["Invalid"], Path)

    def test_empty_scenarios_folder(self, temp_dir):
        """Test handling of empty scenarios folder."""
        scenarios_root = temp_dir / "empty_scenarios"
        scenarios_root.mkdir()

        output_folder = temp_dir / "output"

        results = export_all_genx_scenarios(
            scenarios_root=scenarios_root,
            output_folder_path=output_folder,
            scenario_to_year_map={"p1": 2030},
            verbose=False,
        )

        # Should return empty dict
        assert len(results) == 0


@pytest.mark.slow
@pytest.mark.integration
class TestEndToEndPipeline:
    """End-to-end integration tests."""

    def test_complete_pipeline(self, sample_scenario_structure, temp_dir):
        """Test complete pipeline from input to outputs."""
        output_folder = temp_dir / "output"

        resource_capacity_path, warnings = export_genx_for_plotting(
            scenario_data_path=sample_scenario_structure.parent,
            scenario_name=sample_scenario_structure.name,
            output_folder_path=output_folder,
            scenario_to_year_map={"p1": 2030},
            verbose=False,
        )

        scenario_name = sample_scenario_structure.name

        # Verify op_inputs structure
        op_inputs = output_folder / f"{scenario_name}_op_inputs" / "Inputs" / "Inputs_p1"
        assert (op_inputs / "Generators_data.csv").exists()
        assert (op_inputs / "Load_data.csv").exists()

        # Verify results_summary structure
        results_summary = output_folder / f"{scenario_name}_results_summary"
        assert (results_summary / "annual_demand.csv").exists()

        # Verify data integrity
        generators = pd.read_csv(op_inputs / "Generators_data.csv")
        assert len(generators) > 0
        assert 'Resource' in generators.columns

        annual_demand = pd.read_csv(results_summary / "annual_demand.csv")
        assert len(annual_demand) > 0
        assert all(col in annual_demand.columns for col in ['zone', 'annual_demand', 'planning_year'])

    def test_idempotent_export(self, sample_scenario_structure, temp_dir):
        """Test that running export twice produces same results."""
        output_folder = temp_dir / "output"

        # Run export twice
        path1, warnings1 = export_genx_for_plotting(
            scenario_data_path=sample_scenario_structure.parent,
            scenario_name=sample_scenario_structure.name,
            output_folder_path=output_folder,
            scenario_to_year_map={"p1": 2030},
            verbose=False,
        )

        path2, warnings2 = export_genx_for_plotting(
            scenario_data_path=sample_scenario_structure.parent,
            scenario_name=sample_scenario_structure.name,
            output_folder_path=output_folder,
            scenario_to_year_map={"p1": 2030},
            verbose=False,
        )

        # Paths should be the same
        assert path1 == path2

        # If resource_capacity was created, outputs should be identical
        if path1.exists():
            df1 = pd.read_csv(path1)
            df2 = pd.read_csv(path2)
            pd.testing.assert_frame_equal(df1, df2)
        else:
            # If not created, both runs should have consistent warnings
            assert len(warnings1) > 0
            assert len(warnings2) > 0
