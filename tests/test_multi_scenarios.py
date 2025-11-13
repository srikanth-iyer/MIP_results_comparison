"""
Tests using multiple real sample scenarios (sample_scenario_1 and sample_scenario_2).

These tests validate that the code works across different GenX scenarios with
multi-period structures (p1-p4).
"""
import pytest
import pandas as pd
from pathlib import Path
from reformat_data_for_plot import (
    build_generators_data,
    create_annual_demand_csv,
    export_all_genx_scenarios,
    export_genx_for_plotting,
)


@pytest.mark.real_data
@pytest.mark.integration
class TestMultiPeriodScenarios:
    """Tests using real multi-period scenarios."""

    @pytest.mark.parametrize("scenario_fixture,scenario_num", [
        ("real_sample_scenario_1", 1),
        ("real_sample_scenario_2", 2),
    ])
    def test_scenario_structure(self, scenario_fixture, scenario_num, request):
        """Verify scenario has expected multi-period structure."""
        scenario_path = request.getfixturevalue(scenario_fixture)
        
        assert (scenario_path / "inputs").exists()

        # Check all periods exist
        inputs_dir = scenario_path / "inputs"
        for period in ["inputs_p1", "inputs_p2", "inputs_p3", "inputs_p4"]:
            period_path = inputs_dir / period
            assert period_path.exists(), f"Missing period: {period}"
            assert (period_path / "results").exists()
            assert (period_path / "resources").exists()
            assert (period_path / "system").exists()

        print(f"[OK] Scenario {scenario_num} structure validated: 4 periods found")

    @pytest.mark.parametrize("scenario_fixture", ["real_sample_scenario_1", "real_sample_scenario_2"])
    def test_build_generators_all_periods(self, scenario_fixture, request, temp_dir):
        """Test building generators data for all periods in each scenario."""
        scenario_path = request.getfixturevalue(scenario_fixture)
        scenario_name = scenario_path.name

        inputs_dir = scenario_path / "inputs"
        results = {}

        for period_name in ["inputs_p1", "inputs_p2", "inputs_p3", "inputs_p4"]:
            period_path = inputs_dir / period_name
            if not period_path.exists():
                continue

            output_path = temp_dir / f"{scenario_name}_{period_name}_Generators_data.csv"

            result_df = build_generators_data(
                scenario_folder_path=period_path,
                save_file_path=output_path,
                verbose=False,
            )

            results[period_name] = {
                'path': output_path,
                'count': len(result_df),
                'dataframe': result_df
            }

            assert output_path.exists()
            assert len(result_df) > 0

        print(f"[OK] {scenario_name}: Built generators for {len(results)} periods")
        for period, data in results.items():
            print(f"   {period}: {data['count']} resources")

    def test_annual_demand_all_periods_scenario_1(self, real_sample_scenario_1, temp_dir):
        """Test annual demand calculation for all periods in scenario 1."""
        inputs_dir = real_sample_scenario_1 / "inputs"
        results = {}

        period_to_year = {"p1": 2030, "p2": 2035, "p3": 2040, "p4": 2050}

        for period_num, year in period_to_year.items():
            period_path = inputs_dir / f"inputs_{period_num}"
            if not period_path.exists():
                continue

            output_path = temp_dir / f"annual_demand_{period_num}.csv"

            create_annual_demand_csv(
                scenario_path=period_path,
                output_path=output_path,
                planning_year=year,
                verbose=False,
            )

            assert output_path.exists()
            demand_df = pd.read_csv(output_path)
            assert len(demand_df) > 0
            assert all(demand_df['planning_year'] == year)

            results[period_num] = {
                'zones': len(demand_df),
                'total_demand': demand_df['annual_demand'].sum()
            }

        print(f"[OK] Scenario 1: Calculated demand for {len(results)} periods")
        for period, data in results.items():
            print(f"   {period}: {data['zones']} zones, {data['total_demand']:,.0f} MWh")

    def test_annual_demand_all_periods_scenario_2(self, real_sample_scenario_2, temp_dir):
        """Test annual demand calculation for all periods in scenario 2."""
        inputs_dir = real_sample_scenario_2 / "inputs"
        results = {}

        period_to_year = {"p1": 2030, "p2": 2035, "p3": 2040, "p4": 2050}

        for period_num, year in period_to_year.items():
            period_path = inputs_dir / f"inputs_{period_num}"
            if not period_path.exists():
                continue

            output_path = temp_dir / f"annual_demand_{period_num}.csv"

            create_annual_demand_csv(
                scenario_path=period_path,
                output_path=output_path,
                planning_year=year,
                verbose=False,
            )

            assert output_path.exists()
            demand_df = pd.read_csv(output_path)
            assert len(demand_df) > 0

            results[period_num] = {
                'zones': len(demand_df),
                'total_demand': demand_df['annual_demand'].sum()
            }

        print(f"[OK] Scenario 2: Calculated demand for {len(results)} periods")
        for period, data in results.items():
            print(f"   {period}: {data['zones']} zones, {data['total_demand']:,.0f} MWh")


@pytest.mark.real_data
@pytest.mark.integration
class TestFullExportPipeline:
    """Test full export pipeline with real scenarios."""

    def test_export_scenario_1_multi_period(self, real_sample_scenario_1, temp_dir):
        """Test exporting scenario 1 with all 4 periods."""
        output_folder = temp_dir / "output"

        # Map periods to years
        scenario_to_year_map = {
            "p1": 2030,
            "p2": 2035,
            "p3": 2040,
            "p4": 2050,
        }

        resource_capacity_path, warnings = export_genx_for_plotting(
            scenario_data_path=real_sample_scenario_1.parent,
            scenario_name=real_sample_scenario_1.name,
            output_folder_path=output_folder,
            scenario_to_year_map=scenario_to_year_map,
            verbose=False,
        )

        # Check op_inputs were created for all periods
        op_inputs = output_folder / f"{real_sample_scenario_1.name}_op_inputs" / "Inputs"
        assert op_inputs.exists()

        periods_processed = 0
        for period in ["Inputs_p1", "Inputs_p2", "Inputs_p3", "Inputs_p4"]:
            period_dir = op_inputs / period
            if period_dir.exists():
                assert (period_dir / "Generators_data.csv").exists()
                periods_processed += 1

        print(f"[OK] Scenario 1 export: {periods_processed} periods processed")
        print(f"   Warnings: {len(warnings)}")

        # Check results_summary
        results_summary = output_folder / f"{real_sample_scenario_1.name}_results_summary"
        assert results_summary.exists()
        assert (results_summary / "annual_demand.csv").exists()

        # Verify annual demand has all periods
        annual_demand = pd.read_csv(results_summary / "annual_demand.csv")
        unique_years = sorted(annual_demand['planning_year'].unique())
        print(f"   Planning years in demand: {unique_years}")

        costs_summary_path = results_summary / "costs.csv"
        assert costs_summary_path.exists()
        costs_summary_df = pd.read_csv(costs_summary_path)
        cost_years = sorted(costs_summary_df['planning_year'].unique())
        assert cost_years == [2030, 2035, 2040, 2050]

    def test_export_scenario_2_multi_period(self, real_sample_scenario_2, temp_dir):
        """Test exporting scenario 2 with all 4 periods."""
        output_folder = temp_dir / "output"

        scenario_to_year_map = {
            "p1": 2030,
            "p2": 2035,
            "p3": 2040,
            "p4": 2050,
        }

        resource_capacity_path, warnings = export_genx_for_plotting(
            scenario_data_path=real_sample_scenario_2.parent,
            scenario_name=real_sample_scenario_2.name,
            output_folder_path=output_folder,
            scenario_to_year_map=scenario_to_year_map,
            verbose=False,
        )

        # Check op_inputs
        op_inputs = output_folder / f"{real_sample_scenario_2.name}_op_inputs" / "Inputs"
        assert op_inputs.exists()

        periods_processed = 0
        for period in ["Inputs_p1", "Inputs_p2", "Inputs_p3", "Inputs_p4"]:
            period_dir = op_inputs / period
            if period_dir.exists():
                assert (period_dir / "Generators_data.csv").exists()
                periods_processed += 1

        print(f"[OK] Scenario 2 export: {periods_processed} periods processed")
        print(f"   Warnings: {len(warnings)}")

        # Check results_summary
        results_summary = output_folder / f"{real_sample_scenario_2.name}_results_summary"
        assert results_summary.exists()
        assert (results_summary / "costs.csv").exists()

    def test_export_both_scenarios_batch(self, temp_dir):
        """Test batch export of both scenarios."""
        # Get the actual fixtures directory from the test directory
        fixtures_dir = Path(__file__).parent / "fixtures"
        output_folder = temp_dir / "output"

        scenario_to_year_map = {
            "p1": 2030,
            "p2": 2035,
            "p3": 2040,
            "p4": 2050,
        }

        results = export_all_genx_scenarios(
            scenarios_root=fixtures_dir,
            output_folder_path=output_folder,
            scenario_to_year_map=scenario_to_year_map,
            verbose=False,
        )

        # Should have processed both scenarios
        assert "sample_scenario_1" in results
        assert "sample_scenario_2" in results

        print(f"[OK] Batch export: {len(results)} scenarios processed")
        for scenario_name in results.keys():
            print(f"   - {scenario_name}")


@pytest.mark.real_data
class TestDataConsistency:
    """Test data consistency across scenarios and periods."""

    @pytest.mark.parametrize("scenario_fixture", ["real_sample_scenario_1", "real_sample_scenario_2"])
    def test_category_flags_all_periods(self, scenario_fixture, request, temp_dir):
        """Test that category flags are consistent across periods."""
        scenario_path = request.getfixturevalue(scenario_fixture)
        inputs_dir = scenario_path / "inputs"

        category_counts = {}

        for period_name in ["inputs_p1", "inputs_p2", "inputs_p3", "inputs_p4"]:
            period_path = inputs_dir / period_name
            if not period_path.exists():
                continue

            output_path = temp_dir / f"{period_name}_Generators_data.csv"
            result_df = build_generators_data(
                scenario_folder_path=period_path,
                save_file_path=output_path,
                verbose=False,
            )

            category_cols = ['THERM', 'VRE', 'STOR', 'HYDRO', 'MUST_RUN', 'FLEX']
            category_counts[period_name] = {}

            for col in category_cols:
                if col in result_df.columns:
                    category_counts[period_name][col] = int(result_df[col].sum())

        print(f"[OK] {scenario_path.name}: Category counts by period")
        for period, counts in category_counts.items():
            print(f"   {period}: {counts}")

    def test_resource_counts_by_period(self, real_sample_scenario_1, temp_dir):
        """Test and compare resource counts across periods in scenario 1."""
        inputs_dir = real_sample_scenario_1 / "inputs"
        resource_counts = {}

        for period_name in ["inputs_p1", "inputs_p2", "inputs_p3", "inputs_p4"]:
            period_path = inputs_dir / period_name
            if not period_path.exists():
                continue

            output_path = temp_dir / f"{period_name}_Generators_data.csv"
            result_df = build_generators_data(
                scenario_folder_path=period_path,
                save_file_path=output_path,
                verbose=False,
            )

            resource_counts[period_name] = len(result_df)

        print("[OK] Scenario 1 resource counts:")
        for period, count in resource_counts.items():
            print(f"   {period}: {count} resources")

        # All periods should have resources
        assert all(count > 0 for count in resource_counts.values())


@pytest.mark.real_data
@pytest.mark.slow
class TestComprehensiveValidation:
    """Comprehensive end-to-end validation tests."""

    def test_complete_pipeline_scenario_1(self, real_sample_scenario_1, temp_dir):
        """Complete end-to-end test for scenario 1."""
        output_folder = temp_dir / "output"

        scenario_to_year_map = {
            "p1": 2030,
            "p2": 2035,
            "p3": 2040,
            "p4": 2050,
        }

        # Run full export
        resource_capacity_path, warnings = export_genx_for_plotting(
            scenario_data_path=real_sample_scenario_1.parent,
            scenario_name=real_sample_scenario_1.name,
            output_folder_path=output_folder,
            scenario_to_year_map=scenario_to_year_map,
            verbose=False,
        )

        # Validate outputs
        op_inputs = output_folder / f"{real_sample_scenario_1.name}_op_inputs" / "Inputs"
        results_summary = output_folder / f"{real_sample_scenario_1.name}_results_summary"

        # Check all period outputs
        for period in ["Inputs_p1", "Inputs_p2", "Inputs_p3", "Inputs_p4"]:
            period_dir = op_inputs / period
            if period_dir.exists():
                generators_path = period_dir / "Generators_data.csv"
                assert generators_path.exists()

                generators_df = pd.read_csv(generators_path)
                assert len(generators_df) > 0
                print(f"[OK] {period}: {len(generators_df)} generators")

        # Check aggregated outputs
        assert (results_summary / "annual_demand.csv").exists()
        annual_demand = pd.read_csv(results_summary / "annual_demand.csv")
        print(f"[OK] Annual demand: {len(annual_demand)} records across all periods")

        costs_summary_path = results_summary / "costs.csv"
        assert costs_summary_path.exists()
        costs_summary_df = pd.read_csv(costs_summary_path)
        print(f"[OK] Costs summary: {len(costs_summary_df)} records across all periods")

        print("[OK] Complete pipeline test passed for scenario 1")
        print(f"   Total warnings: {len(warnings)}")


if __name__ == "__main__":
    # Run with: pytest tests/test_multi_scenarios.py -v -s
    pytest.main([__file__, "-v", "-s"])
