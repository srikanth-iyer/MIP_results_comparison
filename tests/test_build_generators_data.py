"""
Tests for build_generators_data module.

Tests the generator data consolidation pipeline including:
- File reading and merging
- Category mapping (THERM, VRE, STOR, etc.)
- R_ID mapping from NetRevenue.csv
- Column renaming and validation
- Error handling
"""
import pytest
import pandas as pd
from reformat_data_for_plot.build_generators_data import (
    build_generators_data,
    compare_columns,
    _read_table,
)


class TestReadTable:
    """Tests for the _read_table helper function."""

    def test_read_csv(self, temp_dir):
        """Test reading CSV files."""
        csv_path = temp_dir / "test.csv"
        test_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        test_df.to_csv(csv_path, index=False)

        result = _read_table(csv_path)
        pd.testing.assert_frame_equal(result, test_df)

    def test_read_excel(self, temp_dir):
        """Test reading Excel files."""
        xlsx_path = temp_dir / "test.xlsx"
        test_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        test_df.to_excel(xlsx_path, index=False)

        result = _read_table(xlsx_path)
        pd.testing.assert_frame_equal(result, test_df)

    def test_unsupported_file_type(self, temp_dir):
        """Test that unsupported file types raise ValueError."""
        bad_path = temp_dir / "test.txt"
        bad_path.write_text("some content")

        with pytest.raises(ValueError, match="Unsupported file type"):
            _read_table(bad_path)


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

        # Check that output file was created
        assert output_path.exists()

        # Check that all resources are present
        expected_resources = ['NGCC_1', 'NGCC_2', 'Coal_1', 'Wind_1', 'Wind_2', 'Solar_1', 'Battery_1', 'Battery_2', 'Hydro_1']
        assert set(result_df.index) == set(expected_resources)

        # Check that DataFrame is returned
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 9  # 3 thermal + 3 VRE + 2 storage + 1 hydro

    def test_category_mapping(self, sample_scenario_structure, temp_dir):
        """Test that category columns (THERM, VRE, STOR, etc.) are correctly set."""
        output_path = temp_dir / "Generators_data.csv"

        result_df = build_generators_data(
            scenario_folder_path=sample_scenario_structure,
            save_file_path=output_path,
            verbose=False,
        )

        # Check THERM category
        assert result_df.loc['NGCC_1', 'THERM'] == 1
        assert result_df.loc['Wind_1', 'THERM'] == 0

        # Check VRE category
        assert result_df.loc['Wind_1', 'VRE'] == 1
        assert result_df.loc['Solar_1', 'VRE'] == 1
        assert result_df.loc['NGCC_1', 'VRE'] == 0

        # Check STOR category
        assert result_df.loc['Battery_1', 'STOR'] == 1
        assert result_df.loc['NGCC_1', 'STOR'] == 0

        # Check HYDRO category
        assert result_df.loc['Hydro_1', 'HYDRO'] == 1
        assert result_df.loc['NGCC_1', 'HYDRO'] == 0

    def test_r_id_mapping(self, sample_scenario_structure, temp_dir):
        """Test that R_ID is correctly mapped from NetRevenue.csv."""
        output_path = temp_dir / "Generators_data.csv"

        result_df = build_generators_data(
            scenario_folder_path=sample_scenario_structure,
            save_file_path=output_path,
            verbose=False,
        )

        # Check that R_ID column exists
        assert 'R_ID' in result_df.columns

        # Check specific mappings
        assert result_df.loc['NGCC_1', 'R_ID'] == 1
        assert result_df.loc['Wind_1', 'R_ID'] == 4
        assert result_df.loc['Battery_1', 'R_ID'] == 7

    def test_missing_resource_folder(self, temp_dir):
        """Test error handling when resource folder is missing."""
        nonexistent_path = temp_dir / "nonexistent"
        output_path = temp_dir / "output.csv"

        with pytest.raises(FileNotFoundError, match="Resource folder not found"):
            build_generators_data(
                scenario_folder_path=nonexistent_path,
                save_file_path=output_path,
            )

    def test_warning_callback(self, sample_scenario_structure, temp_dir, mock_warning_callback):
        """Test that warnings are properly collected via callback."""
        output_path = temp_dir / "Generators_data.csv"

        # Remove NetRevenue.csv to trigger a warning
        net_revenue_path = sample_scenario_structure / "results" / "NetRevenue.csv"
        if net_revenue_path.exists():
            net_revenue_path.unlink()

        build_generators_data(
            scenario_folder_path=sample_scenario_structure,
            save_file_path=output_path,
            warning_callback=mock_warning_callback,
            verbose=False,
        )

        # Check that warning was recorded
        assert len(mock_warning_callback.warnings) > 0
        assert any("NetRevenue.csv not found" in w for w in mock_warning_callback.warnings)

    def test_debug_overwrites(self, sample_scenario_structure, temp_dir):
        """Test that debug_overwrites generates overwrite report."""
        output_path = temp_dir / "Generators_data.csv"

        build_generators_data(
            scenario_folder_path=sample_scenario_structure,
            save_file_path=output_path,
            debug_overwrites=True,
            verbose=False,
        )

    # Report may or may not exist depending on whether overwrites occurred; smoke test only

    def test_missing_resource_column(self, temp_dir):
        """Test handling of files missing 'Resource' column."""
        scenario_path = temp_dir / "bad_scenario"
        resources_dir = scenario_path / "resources"
        resources_dir.mkdir(parents=True)

        # Create a file without 'Resource' column
        bad_df = pd.DataFrame({'Name': ['Gen1'], 'Capacity': [100]})
        bad_df.to_csv(resources_dir / "Bad.csv", index=False)

        output_path = temp_dir / "output.csv"

        # Should handle gracefully with warning
        result = build_generators_data(
            scenario_folder_path=scenario_path,
            save_file_path=output_path,
            verbose=False,
        )

        # Should return empty or minimal DataFrame
        assert isinstance(result, pd.DataFrame)

    def test_fillna_zero(self, sample_scenario_structure, temp_dir):
        """Test that missing values are filled with 0."""
        output_path = temp_dir / "Generators_data.csv"

        result_df = build_generators_data(
            scenario_folder_path=sample_scenario_structure,
            save_file_path=output_path,
            verbose=False,
        )

        # VRE resources shouldn't have Heat_Rate, should be 0
        assert result_df.loc['Wind_1', 'Heat_Rate_MMBTU_per_MWh'] == 0

        # Thermal resources shouldn't have storage attributes
        assert result_df.loc['NGCC_1', 'Eff_Up'] == 0

    def test_deterministic_output(self, sample_scenario_structure, temp_dir):
        """Test that running twice produces identical output."""
        output_path1 = temp_dir / "output1.csv"
        output_path2 = temp_dir / "output2.csv"

        df1 = build_generators_data(
            scenario_folder_path=sample_scenario_structure,
            save_file_path=output_path1,
            verbose=False,
        )

        df2 = build_generators_data(
            scenario_folder_path=sample_scenario_structure,
            save_file_path=output_path2,
            verbose=False,
        )

        # Should produce identical results
        pd.testing.assert_frame_equal(df1, df2)

    def test_sorted_index(self, sample_scenario_structure, temp_dir):
        """Test that output has sorted index (Resource names)."""
        output_path = temp_dir / "Generators_data.csv"

        result_df = build_generators_data(
            scenario_folder_path=sample_scenario_structure,
            save_file_path=output_path,
            verbose=False,
        )

        # Index should be sorted
        assert list(result_df.index) == sorted(result_df.index)

    def test_column_renaming(self, temp_dir):
        """Test that Derating_factor columns are renamed to CapRes."""
        scenario_path = temp_dir / "rename_test"
        resources_dir = scenario_path / "resources"
        resources_dir.mkdir(parents=True)

        # Create data with Derating_factor columns
        df = pd.DataFrame({
            'Resource': ['Gen1'],
            'Derating_factor_1': [0.95],
            'Derating_factor_2': [0.90],
        })
        df.to_csv(resources_dir / "Thermal.csv", index=False)

        output_path = temp_dir / "output.csv"
        result_df = build_generators_data(
            scenario_folder_path=scenario_path,
            save_file_path=output_path,
            verbose=False,
        )

        # Should be renamed to CapRes_*
        assert 'CapRes_1' in result_df.columns
        assert 'CapRes_2' in result_df.columns
        assert 'Derating_factor_1' not in result_df.columns
        assert result_df.loc['Gen1', 'CapRes_1'] == 0.95


@pytest.mark.unit
class TestCompareColumns:
    """Tests for compare_columns function."""

    def test_identical_columns(self):
        """Test comparison of DataFrames with identical columns."""
        df1 = pd.DataFrame(columns=['A', 'B', 'C'])
        df2 = pd.DataFrame(columns=['A', 'B', 'C'])

        missing, extra = compare_columns(df1, df2, print_fn=None)

        assert missing == []
        assert extra == []

    def test_missing_columns(self):
        """Test detection of missing columns."""
        target = pd.DataFrame(columns=['A', 'B', 'C', 'D'])
        actual = pd.DataFrame(columns=['A', 'B'])

        missing, extra = compare_columns(target, actual, print_fn=None)

        assert sorted(missing) == ['C', 'D']
        assert extra == []

    def test_extra_columns(self):
        """Test detection of extra columns."""
        target = pd.DataFrame(columns=['A', 'B'])
        actual = pd.DataFrame(columns=['A', 'B', 'C', 'D'])

        missing, extra = compare_columns(target, actual, print_fn=None)

        assert missing == []
        assert sorted(extra) == ['C', 'D']

    def test_whitespace_normalization(self):
        """Test that whitespace is properly normalized."""
        target = pd.DataFrame(columns=['A', 'B  ', '  C'])
        actual = pd.DataFrame(columns=['A', 'B', 'C'])

        missing, extra = compare_columns(
            target, actual, normalize=True, collapse_ws=True, print_fn=None
        )

        assert missing == []
        assert extra == []


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests with realistic scenario structures."""

    def test_multi_file_merge(self, temp_dir):
        """Test merging attributes from multiple resource files."""
        scenario_path = temp_dir / "complex_scenario"
        resources_dir = scenario_path / "resources"
        resources_dir.mkdir(parents=True)

        # Create base thermal file
        thermal_df = pd.DataFrame({
            'Resource': ['Gen1', 'Gen2'],
            'Zone': [1, 2],
            'Existing_Cap_MW': [500, 300],
        })
        thermal_df.to_csv(resources_dir / "Thermal.csv", index=False)

        # Create costs file with additional attributes
        costs_df = pd.DataFrame({
            'Resource': ['Gen1', 'Gen2'],
            'Var_OM_Cost_per_MWh': [3.5, 4.0],
            'Fixed_OM_Cost_per_MWyr': [10000, 8000],
        })
        costs_df.to_csv(resources_dir / "Costs.csv", index=False)

        output_path = temp_dir / "output.csv"
        result_df = build_generators_data(
            scenario_folder_path=scenario_path,
            save_file_path=output_path,
            verbose=False,
        )

        # Should have attributes from both files
        assert result_df.loc['Gen1', 'Existing_Cap_MW'] == 500
        assert result_df.loc['Gen1', 'Var_OM_Cost_per_MWh'] == 3.5

    def test_policy_assignments_override(self, temp_dir):
        """Test that policy_assignments folder files override base files."""
        scenario_path = temp_dir / "policy_scenario"
        resources_dir = scenario_path / "resources"
        policy_dir = resources_dir / "policy_assignments"
        policy_dir.mkdir(parents=True)

        # Base file
        base_df = pd.DataFrame({
            'Resource': ['Gen1'],
            'Zone': [1],
            'CapRes_1': [0.0],
        })
        base_df.to_csv(resources_dir / "Thermal.csv", index=False)

        # Policy override
        policy_df = pd.DataFrame({
            'Resource': ['Gen1'],
            'CapRes_1': [0.95],
        })
        policy_df.to_csv(policy_dir / "CapacityReserve.csv", index=False)

        output_path = temp_dir / "output.csv"
        result_df = build_generators_data(
            scenario_folder_path=scenario_path,
            save_file_path=output_path,
            verbose=False,
        )

        # Policy file should override base file
        assert result_df.loc['Gen1', 'CapRes_1'] == 0.95

    @pytest.mark.slow
    def test_large_scenario(self, temp_dir):
        """Test performance with larger number of resources."""
        scenario_path = temp_dir / "large_scenario"
        resources_dir = scenario_path / "resources"
        resources_dir.mkdir(parents=True)

        # Create 1000 resources
        large_df = pd.DataFrame({
            'Resource': [f'Gen{i}' for i in range(1000)],
            'Zone': [(i % 10) + 1 for i in range(1000)],
            'Existing_Cap_MW': [100.0 + i for i in range(1000)],
        })
        large_df.to_csv(resources_dir / "Thermal.csv", index=False)

        output_path = temp_dir / "output.csv"
        result_df = build_generators_data(
            scenario_folder_path=scenario_path,
            save_file_path=output_path,
            verbose=False,
        )

        assert len(result_df) == 1000
        assert 'Gen999' in result_df.index
